"""
Configuration schema for signal convolution in-context learning experiments.
"""

from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


# ============================================================================
# Model Schema
# ============================================================================

model_schema = {
    "family": merge(tstring, allowed(["gpt2"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),       # p (time) or 2p (freq)
    "out_dim": merge(tinteger, required),      # same as n_dims
    "n_embd": merge(tinteger, required),       # embedding dimension
    "n_layer": merge(tinteger, required),      # number of transformer layers
    "n_head": merge(tinteger, required),       # number of attention heads
}


# ============================================================================
# Curriculum Schema
# ============================================================================

curriculum_base_schema = {
    "start": merge(tinteger, required),    # initial parameter value
    "end": merge(tinteger, required),      # final parameter value
    "inc": merge(tinteger, required),      # increment amount
    "interval": merge(tinteger, required), # steps between increments
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),    # dimension curriculum
    "points": stdict(curriculum_base_schema),  # context length curriculum
}


# ============================================================================
# Training Schema
# ============================================================================

# Supported tasks
TASK_LIST = [
    "signal_conv",  # Signal convolution task
]

# Supported data samplers
DATA_LIST = [
    "signal",  # Signal sampler
]

# Domain types
DOMAIN_LIST = ["time", "freq"]

# Distribution types
DIST_LIST = ["normal", "uniform"]

training_schema = {
    # Task configuration
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    
    # Data configuration
    "data": merge(tstring, allowed(DATA_LIST)),
    "data_kwargs": merge(tdict, required),
    
    # Training parameters
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(100000)),
    
    # Checkpointing
    "save_every_steps": merge(tinteger, default(1000)),
    "keep_every_steps": merge(tinteger, default(-1)),
    "resume_id": merge(tstring, nullable, default(None)),
    
    # Training pool size (for deterministic experiments)
    "num_training_examples": merge(tinteger, nullable, default(None)),
    
    # Loss configuration
    "loss_space": merge(tstring, allowed(["time", "freq"]), default("time")),
    
    # Curriculum
    "curriculum": stdict(curriculum_schema),
}


# ============================================================================
# Weights & Biases Schema
# ============================================================================

wandb_schema = {
    "project": merge(tstring, default("signal-incontext")),
    "entity": merge(tstring, default("your-entity")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}


# ============================================================================
# Top-Level Schema
# ============================================================================

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}