# Configuration Guide

## Overview

This project uses [Quinine](https://github.com/krandiash/quinine) for configuration management. Configs are defined in YAML files and validated against a schema.

## File Structure

```
configs/
├── base.yaml              # Base config with common settings
├── time_domain.yaml       # Time-domain experiment (inherits base)
└── freq_domain.yaml       # Frequency-domain experiment (inherits base)

schema.py                  # Configuration schema definition
```

## Running Experiments

### Time Domain
```bash
python train.py --config configs/time_domain.yaml
```

### Frequency Domain
```bash
python train.py --config configs/freq_domain.yaml
```

### Quick Test Run
```bash
python train.py --config configs/time_domain.yaml --test_run
```

## Key Configuration Parameters

### Model (`model`)

| Parameter | Description | Time Domain | Freq Domain |
|-----------|-------------|-------------|-------------|
| `n_dims` | Input/output dimension | p (e.g., 64) | 2p (e.g., 128) |
| `out_dim` | Output dimension | Same as n_dims | Same as n_dims |
| `n_positions` | Max context length | 101 | 101 |
| `n_embd` | Embedding dimension | 256 | 256 |
| `n_layer` | Transformer layers | 12 | 12 |
| `n_head` | Attention heads | 4 | 4 |

### Task Configuration (`training.task_kwargs`)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `p` | Signal period (time-domain length) | 64 |
| `fir_len` | FIR filter length | 16 |
| `domain` | Domain: "time" or "freq" | "time" |
| `fir_dist` | FIR distribution: "normal" or "uniform" | "normal" |
| `normalize_fir` | Normalize FIR to unit L1 norm | true |

### Data Configuration (`training.data_kwargs`)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `p` | Signal period (must match task) | 64 |
| `domain` | Domain: "time" or "freq" | "time" |
| `amp_dist` | Amplitude distribution | "normal" |
| `amp_std` | Std dev for normal distribution | 1.0 |
| `amp_max` | Max for uniform distribution | 1.0 |
| `num_freqs` | Number of frequency harmonics | 32 |

### Training (`training`)

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `batch_size` | Batch size | 64 |
| `learning_rate` | Learning rate | 0.0001 |
| `train_steps` | Total training steps | 500000 |
| `loss_space` | Loss domain: "time" or "freq" | **"time"** for fair comparison |
| `num_training_examples` | Fixed training pool size | 10000 |

### Curriculum (`training.curriculum`)

The curriculum gradually increases task difficulty during training.

**Points curriculum** (context length):
- `start`: Initial number of in-context examples (e.g., 5)
- `end`: Final number of in-context examples (e.g., 41)
- `inc`: Increment amount (e.g., 1)
- `interval`: Steps between increments (e.g., 2000)

**Dims curriculum** (dimension truncation):
- Not used for signal tasks - keep constant
- Set `start = end` and `inc = 0`

## Important Notes for Fair Comparison

### 1. Loss Space
**Always use `loss_space: time`** when comparing time vs. frequency models. This ensures both are evaluated on the same metric (time-domain reconstruction error).

### 2. Dimension Consistency
- **Time domain**: `n_dims = p`
- **Frequency domain**: `n_dims = 2p` (interleaved real/imaginary)

Make sure:
- `model.n_dims == model.out_dim`
- `training.task_kwargs.domain == training.data_kwargs.domain`
- `training.curriculum.dims.start == model.n_dims`

### 3. Same Training Conditions
For fair comparison, keep these identical:
- `batch_size`
- `learning_rate`
- `train_steps`
- `num_training_examples`
- `curriculum.points` (context length schedule)
- `task_kwargs.p`, `task_kwargs.fir_len`

### 4. Model Capacity
The frequency model has 2x input dimensions but same embedding size. This is by design - the experiment tests whether the easier problem structure (linear in freq) outweighs the increased dimensionality.

## Example: Varying Signal Period

To test different signal periods, create new configs:

```yaml
# configs/time_p128.yaml
inherit:
    - base.yaml

model:
    n_dims: 128
    out_dim: 128

training:
    task: signal_conv
    task_kwargs:
        p: 128
        fir_len: 32
        domain: time
        # ... rest same
    data_kwargs:
        p: 128
        domain: time
        # ... rest same
    curriculum:
        dims:
            start: 128
            end: 128

wandb:
    name: time_p128_fir32
```

## Monitoring Training

Weights & Biases logs:
- `overall_loss`: MSE loss (in configured loss_space)
- `excess_loss`: Loss normalized by curriculum baseline
- `pointwise/loss`: Loss at each in-context position
- `n_points`: Current context length from curriculum
- `n_dims`: Current dimension from curriculum

## Troubleshooting

**Error: "n_dims must equal p for time domain"**
- Check `model.n_dims == training.data_kwargs.p`

**Error: "n_dims must equal 2*p for freq domain"**
- Check `model.n_dims == 2 * training.data_kwargs.p`

**Error: "p must be specified"**
- Ensure both `task_kwargs` and `data_kwargs` have `p` parameter

**Models not comparable:**
- Verify `loss_space: time` in both configs
- Ensure same `p`, `fir_len`, and training hyperparameters
