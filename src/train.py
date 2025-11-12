"""
Training script for in-context learning on signal convolution tasks.

Compares time-domain vs frequency-domain transformers on their ability to learn
convolution from in-context examples.
"""

import os
import uuid
from random import randint

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import wandb
from quinine import QuinineArgumentParser

from schema import schema
from models import build_model
from sampler import get_data_sampler
from tasks import get_task_sampler
from curriculum import Curriculum

torch.backends.cudnn.benchmark = True


# ============================================================================
# Loss Functions
# ============================================================================

def _interleaved_to_complex(vec_2p: torch.Tensor) -> torch.Tensor:
    """
    Convert interleaved real/imaginary to complex tensor.
    
    Args:
        vec_2p: (..., 2p) with [Re0, Im0, Re1, Im1, ...]
    Returns:
        (..., p) complex tensor
    """
    re = vec_2p[..., 0::2]
    im = vec_2p[..., 1::2]
    return torch.complex(re, im)


def _to_time_domain(x: torch.Tensor, p: int, fft_norm: str = "ortho") -> torch.Tensor:
    """
    Convert signal to time domain if needed.
    
    Args:
        x: Input tensor
        p: Signal period (time-domain length)
        fft_norm: FFT normalization ('ortho' recommended)
    
    Returns:
        Time-domain signal of shape (..., p)
    
    Behavior:
        - If last dim == p: Already time-domain, return as-is
        - If last dim == 2p: Frequency-domain (interleaved), apply IFFT
        - Otherwise: Raise error
    """
    D = x.shape[-1]
    
    if D == p:
        # Already in time domain
        return x
    elif D == 2 * p:
        # Frequency domain -> convert to time
        Z = _interleaved_to_complex(x)
        return torch.fft.ifft(Z, dim=-1, norm=fft_norm).real
    else:
        raise ValueError(f"Unexpected dimension {D}; expected p={p} or 2p={2*p}")


def make_time_domain_mse(p: int, fft_norm: str = "ortho"):
    """
    Create MSE loss function that operates in time domain.
    
    This ensures fair comparison: both time and frequency models are evaluated
    on the same metric (time-domain reconstruction error).
    
    Args:
        p: Signal period
        fft_norm: FFT normalization
    
    Returns:
        Loss function: (preds, targets) -> scalar loss
    """
    def loss_fn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds_time = _to_time_domain(preds, p, fft_norm)
        targets_time = _to_time_domain(targets, p, fft_norm)
        return F.mse_loss(preds_time, targets_time)
    
    return loss_fn


def make_frequency_domain_mse(p: int):
    """
    Create MSE loss function that operates in frequency domain.
    
    Only valid when both predictions and targets are in frequency domain (2p dims).
    
    Args:
        p: Signal period
    
    Returns:
        Loss function: (preds, targets) -> scalar loss
    """
    def loss_fn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        expected_dim = 2 * p
        if preds.shape[-1] != expected_dim:
            raise ValueError(f"Frequency MSE expects dim={expected_dim}, got {preds.shape[-1]}")
        return F.mse_loss(preds, targets)
    
    return loss_fn


# ============================================================================
# Training
# ============================================================================

def train_step(model, xs, ys, optimizer, loss_func):
    """
    Single training step.
    
    Args:
        model: Transformer model
        xs: Input signals (B, n_points, n_dims)
        ys: Target signals (B, n_points, n_dims)
        optimizer: PyTorch optimizer
        loss_func: Loss function
    
    Returns:
        loss: Scalar loss value
        preds: Model predictions (detached)
    """
    model.train()
    optimizer.zero_grad()
    
    preds = model(xs, ys)
    loss = loss_func(preds, ys)
    
    loss.backward()
    optimizer.step()
    
    return loss.detach().item(), preds.detach()


def sample_deterministic_seeds(pool_size: int, batch_size: int):
    """
    Sample unique seeds from a fixed pool for deterministic training.
    
    Args:
        pool_size: Total number of possible seeds
        batch_size: Number of seeds to sample
    
    Returns:
        List of unique seed integers
    """
    seeds = set()
    while len(seeds) < batch_size:
        seeds.add(randint(0, pool_size - 1))
    return list(seeds)


def compute_pointwise_losses(preds, targets, p, loss_space):
    """
    Compute per-position losses for logging.
    
    Args:
        preds: Model predictions (B, n_points, n_dims)
        targets: Ground truth (B, n_points, n_dims)
        p: Signal period
        loss_space: 'time' or 'freq'
    
    Returns:
        Array of shape (n_points,) with average loss per position
    """
    if loss_space == "time":
        # Convert to time for consistent comparison
        preds_time = _to_time_domain(preds, p)
        targets_time = _to_time_domain(targets, p)
        squared_errors = (preds_time - targets_time) ** 2
    else:
        # Frequency space
        squared_errors = (preds - targets) ** 2
    
    # Average over batch and signal dimensions, keep position dimension
    pointwise_loss = squared_errors.mean(dim=(0, 2)).cpu().numpy()
    return pointwise_loss


def compute_baseline_loss(curriculum):
    """
    Compute baseline loss for curriculum-adjusted metrics.
    
    This is a heuristic baseline based on the curriculum difficulty.
    """
    if curriculum.n_dims_truncated is None:
        return 1.0
    
    # Average "information deficit" across positions
    total = sum(
        max(curriculum.n_dims_truncated - i, 0) 
        for i in range(curriculum.n_points)
    )
    return total / curriculum.n_points


# ============================================================================
# Main Training Loop
# ============================================================================

def train(model, args):
    """Main training loop with curriculum and logging."""
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)
    
    # Resume from checkpoint if exists
    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        
        # Advance curriculum to match
        for _ in range(starting_step + 1):
            curriculum.update()
    
    # Build data and task samplers
    n_dims = model.n_dims
    batch_size = args.training.batch_size
    
    data_sampler = get_data_sampler(
        args.training.data,
        n_dims=n_dims,
        **args.training.data_kwargs,
    )
    
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        batch_size,
        **args.training.task_kwargs,
    )
    
    # Determine signal period for loss functions
    p = _get_signal_period(args)
    
    # Create loss function
    loss_space = getattr(args.training, "loss_space", "time")
    if loss_space == "time":
        loss_func = make_time_domain_mse(p)
    elif loss_space == "freq":
        loss_func = make_frequency_domain_mse(p)
    else:
        raise ValueError(f"loss_space must be 'time' or 'freq', got '{loss_space}'")
    
    # Training loop
    num_training_examples = args.training.num_training_examples
    pbar = tqdm(range(starting_step, args.training.train_steps))
    
    for step in pbar:
        # Prepare sampler arguments
        data_sampler_args = {}
        task_sampler_args = {}
        
        # Deterministic sampling if using fixed pool
        if num_training_examples is not None:
            assert num_training_examples >= batch_size
            seeds = sample_deterministic_seeds(num_training_examples, batch_size)
            data_sampler_args["seeds"] = seeds
            # Offset task seeds to keep inputs and FIRs independent but paired
            task_sampler_args["seeds"] = [s + 1 for s in seeds]
        
        # Sample inputs
        xs = data_sampler.sample_xs(
            curriculum.n_points,
            batch_size,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        ).to(device)
        
        # Generate targets via task
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs).to(device)
        
        # Training step
        loss, preds = train_step(model, xs, ys, optimizer, loss_func)
        
        # Logging
        if step % args.wandb.log_every_steps == 0 and not args.test_run:
            with torch.no_grad():
                pointwise_loss = compute_pointwise_losses(preds, ys, p, loss_space)
                baseline_loss = compute_baseline_loss(curriculum)
                
                wandb.log({
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(enumerate(pointwise_loss)),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated or n_dims,
                }, step=step)
        
        # Update curriculum
        curriculum.update()
        
        # Progress bar
        pbar.set_description(f"loss {loss:.6f}")
        
        # Save checkpoint
        if step % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": step,
            }
            torch.save(training_state, state_path)
        
        # Save model snapshots
        if (args.training.keep_every_steps > 0 and 
            step % args.training.keep_every_steps == 0 and 
            not args.test_run and 
            step > 0):
            torch.save(
                model.state_dict(), 
                os.path.join(args.out_dir, f"model_{step}.pt")
            )


def _get_signal_period(args):
    """Extract signal period p from config."""
    if "p" in args.training.task_kwargs:
        return int(args.training.task_kwargs["p"])
    elif "p" in args.training.data_kwargs:
        return int(args.training.data_kwargs["p"])
    else:
        raise ValueError(
            "Signal period 'p' must be specified in training.task_kwargs "
            "or training.data_kwargs"
        )


# ============================================================================
# Entry Point
# ============================================================================

def main(args):
    """Main entry point."""
    
    # Test run: collapse curriculum for quick smoke test
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        # Initialize wandb
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )
    
    # Build and train model
    model = build_model(args.model)
    train(model, args)
    
    # Evaluate
    if not args.test_run:
        from eval import get_run_metrics
        _ = get_run_metrics(args.out_dir)


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    
    assert args.model.family == "gpt2", "Only 'gpt2' model family is supported"
    
    print(f"Running with config: {args}")
    
    # Setup output directory
    if not args.test_run:
        run_id = args.training.resume_id or str(uuid.uuid4())
        out_dir = os.path.join(args.out_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)
        args.out_dir = out_dir
        
        # Save config
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.dump(args.__dict__, f, default_flow_style=False)
    
    main(args)