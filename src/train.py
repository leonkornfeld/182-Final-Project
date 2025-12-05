"""
Training script for in-context learning on signal convolution tasks.

Compares time-domain vs frequency-domain transformers on their ability to learn
convolution from in-context examples.

Config is read directly from config.yaml as a plain dict.
"""

import os
import uuid
from random import randint

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import wandb

from models import build_model
from samplers import get_data_sampler
from tasks import get_task_sampler
from curriculum import Curriculum
from transformers import get_linear_schedule_with_warmup

torch.backends.cudnn.benchmark = True

CONFIG_PATH = "src/config_freq.yaml"  # your YAML above, saved as config.yaml in this dir

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


# ============================================================================
# Loss Functions
# ============================================================================

def _interleaved_to_complex(vec_2p: torch.Tensor) -> torch.Tensor:
    re = vec_2p[..., 0::2]
    im = vec_2p[..., 1::2]
    return torch.complex(re, im)


def _magphase_to_complex(vec_2p: torch.Tensor) -> torch.Tensor:
    """
    Interpret the last dimension as interleaved [magnitude, phase] and
    reconstruct a complex spectrum.
    """
    mag = vec_2p[..., 0::2]
    phase = vec_2p[..., 1::2]
    return torch.polar(mag, phase)


def _to_time_domain(x: torch.Tensor, p: int, freq_representation: str = "complex") -> torch.Tensor:
    """
    Convert tensors that may live in time domain (dim=p) or frequency domain
    (dim=2*p_fft) back to time domain.

    freq_representation:
        - \"complex\": interleaved Re/Im (default, backwards compatible)
        - \"mag_phase\": interleaved magnitude/phase
    """
    D = x.shape[-1]
    p_fft = p // 2 + 1
    if D == p:
        return x
    elif D == 2 * p_fft:
        if freq_representation == "mag_phase":
            Z = _magphase_to_complex(x)
        else:
            Z = _interleaved_to_complex(x)
        return torch.fft.irfft(Z, n=p, dim=-1, norm="ortho").real
    else:
        raise ValueError(f"Unexpected dimension {D}; expected p={p} or 2*p_fft={2 * p_fft}")


def make_time_domain_mse(p: int, freq_representation: str = "complex"):
    def loss_fn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds_time = _to_time_domain(preds, p, freq_representation)
        targets_time = _to_time_domain(targets, p, freq_representation)
        return F.mse_loss(preds_time, targets_time)
    return loss_fn


def make_frequency_domain_mse(p: int):
    p_fft = p // 2 + 1
    expected_dim = 2 * p_fft

    def loss_fn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.shape[-1] != expected_dim:
            raise ValueError(
                f"Frequency MSE expects dim={expected_dim}, got {preds.shape[-1]}"
            )
        return F.mse_loss(preds, targets)

    return loss_fn


# ============================================================================
# Training helpers
# ============================================================================

def train_step(model, xs, ys, optimizer, loss_func):
    model.train()
    optimizer.zero_grad()
    preds = model(xs, ys)
    loss = loss_func(preds, ys)
    loss.backward()
    # current_norm = get_grad_norm(model)
    # print(f"Gradient Size: {current_norm}")
    optimizer.step()
    return loss.detach().item(), preds.detach()


def sample_deterministic_seeds(pool_size: int, batch_size: int):
    seeds = set()
    while len(seeds) < batch_size:
        seeds.add(randint(0, pool_size - 1))
    return list(seeds)


def compute_pointwise_losses(preds, targets, p, loss_space, cfg):
    train_cfg = cfg["training"]
    task_kwargs = train_cfg.get("task_kwargs", {})
    freq_representation = task_kwargs.get("freq_representation", "complex")

    if loss_space == "time":
        preds_time = _to_time_domain(preds, p, freq_representation)
        targets_time = _to_time_domain(targets, p, freq_representation)
        squared_errors = (preds_time - targets_time) ** 2
    else:
        squared_errors = (preds - targets) ** 2
    return squared_errors.mean(dim=(0, 2)).cpu().numpy()



def get_signal_period(cfg):
    train = cfg["training"]
    if "p" in train["task_kwargs"]:
        return int(train["task_kwargs"]["p"])
    if "p" in train["data_kwargs"]:
        return int(train["data_kwargs"]["p"])
    raise ValueError(
        "Signal period 'p' must be specified in training.task_kwargs or training.data_kwargs"
    )


# ============================================================================
# Main Training Loop
# ============================================================================

def train(model, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_cfg = cfg["training"]

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    num_warmup_steps = train_cfg.get("warmup_steps", 2000)
    total_steps = train_cfg["train_steps"]
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_steps
    )
    # ---------------------------------

    # Curriculum expects an object with .dims.start etc.
    # Build a tiny shim from the dict.
    cur_dict = train_cfg["curriculum"]
    from types import SimpleNamespace
    cur_args = SimpleNamespace(
        points=SimpleNamespace(**cur_dict["points"]),
    )
    curriculum = Curriculum(cur_args)

    # Resume from checkpoint if exists
    starting_step = 0
    state_path = os.path.join(cfg["out_dir"], "state.pt")
    if os.path.exists(state_path):
        print("STOPP")
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for _ in range(starting_step + 1):
            curriculum.update()

    n_dims = model.n_dims
    batch_size = train_cfg["batch_size"]

    data_sampler = get_data_sampler(
        train_cfg["data"],
        n_dims=n_dims,
        **train_cfg["data_kwargs"],
    )

    task_sampler = get_task_sampler(
        train_cfg["task"],
        n_dims,
        batch_size,
        **train_cfg["task_kwargs"],
    )

    p = get_signal_period(cfg)

    loss_space = train_cfg.get("loss_space", "time")
    task_kwargs = train_cfg.get("task_kwargs", {})
    freq_representation = task_kwargs.get("freq_representation", "complex")

    if loss_space == "time":
        loss_func = make_time_domain_mse(p, freq_representation)
    elif loss_space == "freq":
        loss_func = make_frequency_domain_mse(p)
    else:
        raise ValueError(f"loss_space must be 'time' or 'freq', got '{loss_space}'")

    # num_training_examples = train_cfg["num_training_examples"]
    pbar = tqdm(range(starting_step, train_cfg["train_steps"]))

    for step in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        # if num_training_examples is not None:
        #     assert num_training_examples >= batch_size
        #     seeds = sample_deterministic_seeds(num_training_examples, batch_size)
        #     data_sampler_args["seeds"] = seeds
        #     task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            batch_size,
            **data_sampler_args,
        ).to(device)

        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs).to(device)

        loss, preds = train_step(model, xs, ys, optimizer, loss_func)
        scheduler.step()
        if step % cfg["wandb"]["log_every_steps"] == 0 and not cfg["test_run"]:
            with torch.no_grad():
                pointwise_loss = compute_pointwise_losses(preds, ys, p, loss_space, cfg)
                # baseline_loss = compute_baseline_loss(curriculum)
                wandb.log(
                    {
                        "overall_loss": loss,
                        # "excess_loss": loss / baseline_loss,
                        "pointwise/loss": dict(enumerate(pointwise_loss)),
                        "n_points": curriculum.n_points,
                        "n_dims": n_dims,
                    },
                    step=step,
                )

        curriculum.update()
        pbar.set_description(f"loss {loss:.6f}")

        if step % train_cfg["save_every_steps"] == 0 and not cfg["test_run"]:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": step,
            }
            torch.save(training_state, state_path)

        if (
            train_cfg["keep_every_steps"] > 0
            and step % train_cfg["keep_every_steps"] == 0
            and not cfg["test_run"]
            and step > 0
        ):
            torch.save(
                model.state_dict(),
                os.path.join(cfg["out_dir"], f"model_{step}.pt"),
            )


# ============================================================================
# Entry point
# ============================================================================

def main():
    # Load YAML directly
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Sanity: model family
    assert cfg["model"]["family"] == "gpt2", "Only 'gpt2' model family is supported"

    print(f"Running with config: {CONFIG_PATH}")

    # Setup output dir
    if not cfg["test_run"]:
        run_id = cfg["training"].get("resume_id") or str(uuid.uuid4())
        out_dir = os.path.join(cfg["out_dir"], run_id)
        os.makedirs(out_dir, exist_ok=True)
        cfg["out_dir"] = out_dir
        cfg["training"]["resume_id"] = run_id

        # Save resolved config
        with open(os.path.join(out_dir, "config.yaml"), "w") as f_out:
            yaml.dump(cfg, f_out, default_flow_style=False)
    else:
        # Collapse curriculum for quick test
        cur = cfg["training"]["curriculum"]
        cur["points"]["start"] = cur["points"]["end"]
        cfg["training"]["train_steps"] = 100

    # Init wandb (skip if test_run)
    if not cfg["test_run"]:
        wandb.init(
            dir=cfg["out_dir"],
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            config=cfg,
            notes=cfg["wandb"]["notes"],
            name=cfg["wandb"]["name"],
            resume=True,
        )

    # Build model: build_model still expects an object with attributes,
    # so we give it exactly what it wants using a tiny shim.
    from types import SimpleNamespace
    model_conf = SimpleNamespace(**cfg["model"])
    model = build_model(model_conf)

    train(model, cfg)

    if not cfg["test_run"]:
        from eval import get_run_metrics
        _ = get_run_metrics(cfg["out_dir"])


if __name__ == "__main__":
    main()



# Usage inside training loop:
# ... loss.backward()
# current_norm = get_grad_norm(model)
# print(f"Gradient Size: {current_norm}")