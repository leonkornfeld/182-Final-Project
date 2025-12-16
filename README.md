## Project Overview
This repository is forked from https://github.com/dtsip/in-context-learning
This project trains GPT-style transformers to perform **circular convolution** on periodic signals from in-context examples, and compares **time-domain** vs **frequency-domain** parameterizations.

- **Time-domain experiment**: inputs and outputs are length-\(p\) real signals.
- **Frequency-domain experiment**: inputs and outputs are encoded Fourier spectra with **interleaved real and imaginary parts**.

The core logic for these experiments lives in `src/config_time.yaml`, `src/config_freq.yaml`, `src/samplers.py`, `src/tasks.py`, and `src/train.py`.

---

## Repository Layout (relevant parts)

- **`src/config_time.yaml`**: configuration for the time-domain convolution experiment.
- **`src/config_freq.yaml`**: configuration for the frequency-domain convolution experiment.
- **`src/train.py`**: main training script that reads a YAML config and runs training.
- **`src/samplers.py`**: data generation for periodic signals in time or frequency domain.
- **`src/tasks.py`**: definition of the in-context signal convolution task.
- **`src/models.py`**: model construction (GPT‑2 style transformer).
- **`src/eval.py` / `src/eval.ipynb`**: utilities and notebook for evaluating trained models.

---

## Running the Experiments

Training is controlled by the `CONFIG_PATH` constant at the top of `src/train.py`:

- **Current default**:
  - **`CONFIG_PATH = "src/config_freq.yaml"`** → runs the **frequency-domain** experiment.

To switch experiments:

- **Time-domain run**
  - **Edit** `CONFIG_PATH` in `src/train.py` to:
    ```python
    CONFIG_PATH = "src/config_time.yaml"
    ```
  - Then, from the project root:
    ```bash
    cd 182-Final-Project
    python src/train.py
    ```

- **Frequency-domain run**
  - **Ensure** `CONFIG_PATH` in `src/train.py` is:
    ```python
    CONFIG_PATH = "src/config_freq.yaml"
    ```
  - Then run:
    ```bash
    cd 182-Final-Project
    python src/train.py
    ```

Each run:

- Creates a fresh output directory under `./outputs/<run_id>/`.
- Saves the resolved config to `outputs/<run_id>/config.yaml`.
- Logs to Weights & Biases if `test_run: false`.

---

## Time-Domain Configuration (`src/config_time.yaml`)

This config trains a transformer to learn convolution **directly in the time domain** for periodic signals of period \(p = 30\).

- **Output directory**
  - **`out_dir`**: `./outputs`

- **Model (`model`)**
  - **`family`**: `gpt2`
  - **`n_dims`**: `20` (equal to the signal period \(p\))
  - **`out_dim`**: `20` (matches `n_dims`)
  - **`n_positions`**: `101` (max number of in-context (x, y) pairs)
  - **`n_embd`**: `256`
  - **`n_layer`**: `12`
  - **`n_head`**: `8`

- **Training (`training`)**
  - **Task (`training.task`, `training.task_kwargs`)**
    - **`task`**: `signal_conv`
    - **`p`**: `20` (signal period / time-domain length)
    - **`fir_len`**: `20` (FIR filter length)
    - **`domain`**: `time`
  - **Data (`training.data`, `training.data_kwargs`)**
    - **`data`**: `signal`
    - **`p`**: `20` (must match task)
    - **`domain`**: `time`
    - **`amp_dist`**: `normal`
    - **`amp_std`**: `1.0`
    - **`num_freqs`**: `20` (number of harmonics in the random Fourier series)
    - **`device`**: `cuda`
  - **Hyperparameters**
    - **`batch_size`**: `512`
    - **`learning_rate`**: `0.0001`
    - **`train_steps`**: `150001`
  - **Checkpointing**
    - **`save_every_steps`**: `1000`
    - **`keep_every_steps`**: `20000`
  - **Loss**
    - **`loss_space`**: `time` (loss computed in the time domain)
  - **Curriculum (`training.curriculum.points`)**
    - **`start`**: `5` in-context examples
    - **`end`**: `55` in-context examples
    - **`inc`**: `1`
    - **`interval`**: `2000` steps between increments

- **Weights & Biases (`wandb`)**
  - **`project`**: `signal-incontext`
  - **`entity`**: ``
  - **`name`**: `time_p20_fir20_b512_dec14_1530`
  - **`notes`**: `Time-domain transformer learning convolution with p=20, FIR length=20`
  - **`log_every_steps`**: `10`

---

## Frequency-Domain Configuration (`src/config_freq.yaml`)

This config trains a transformer to learn convolution **in the frequency domain**, while still evaluating loss in the time domain.

- **Output directory**
  - **`out_dir`**: `./outputs`

- **Model (`model`)**
  - **`family`**: `gpt2`
  - **`n_dims`**: `22`  
    - For `p = 20`, the real FFT has \(p/2 + 1 = 16\) frequency bins, and the model uses **interleaved real and imaginary parts**, so \(2 \times 16 = 32\).
  - **`out_dim`**: `22`
  - **`n_positions`**: `101`
  - **`n_embd`**: `256`
  - **`n_layer`**: `12`
  - **`n_head`**: `8`

- **Training (`training`)**
  - **Task (`training.task`, `training.task_kwargs`)**
    - **`task`**: `signal_conv`
    - **`p`**: `20` (signal period)
    - **`fir_len`**: `5` (FIR filter length)
    - **`domain`**: `freq`
    - **`device`**: `cuda`
    - **`freq_representation`**: `complex` (interleaved real/imaginary encoding)
  - **Data (`training.data`, `training.data_kwargs`)**
    - **`data`**: `signal`
    - **`p`**: `20`
    - **`domain`**: `freq`
    - **`amp_dist`**: `normal`
    - **`amp_std`**: `1.0`
    - **`num_freqs`**: `20`
    - **`device`**: `cuda`
    - **`freq_representation`**: `complex`
  - **Hyperparameters**
    - **`batch_size`**: `512`
    - **`learning_rate`**: `0.0001`
    - **`train_steps`**: `100001`
  - **Checkpointing**
    - **`save_every_steps`**: `40000`
    - **`keep_every_steps`**: `10000`
  - **Loss**
    - **`loss_space`**: `time` (predictions and targets are converted back to time domain before computing MSE)
  - **Curriculum (`training.curriculum.points`)**
    - **`start`**: `5`
    - **`end`**: `55`
    - **`inc`**: `1`
    - **`interval`**: `2000`

- **Weights & Biases (`wandb`)**
  - **`project`**: `signal-incontext`
  - **`entity`**: `leonkornfeld-uc-berkeley-electrical-engineering-computer`
  - **`name`**: `freq_p20_fir5_b512_dec14_19:17`
  - **`notes`**: `Frequency-domain transformer learning convolution with p=20, FIR length=5`
  - **`log_every_steps`**: `10`

---

## Data Generation (`src/samplers.py`)

The data pipeline is implemented via the `SignalSampler` class.

- **Domain options**
  - **`domain="time"`**:
    - Returns `xs` with shape `(B, T, p)` where `p` is the signal period.
  - **`domain="freq"`**:
    - Generates time-domain signals, transforms them with `torch.fft.rfft`, and encodes the result as interleaved real and imaginary parts, giving shape `(B, T, 2 * p_fft)` where `p_fft = p // 2 + 1`.

- **Signal construction**
  - **`p`**: signal period (matches the config).
  - **`num_freqs`**: number of harmonics in the random Fourier series (defaults to `p` if `None`).
  - **Amplitudes**:
    - **`amp_dist="normal"`**: Gaussian amplitudes with standard deviation `amp_std`.
    - **`amp_dist="uniform"`**: amplitudes uniformly in \([-amp_max, amp_max]\).
  - **Phases**:
    - Drawn uniformly from \([0, 2\pi)\).
  - Each time-domain signal is normalized per sample (zero mean, unit variance) before encoding.

- **Shapes**
  - **Time domain**: `(batch_size, n_points, p)`
  - **Frequency domain (complex encoding)**: `(batch_size, n_points, 2 * p_fft)`

The `get_data_sampler("signal", ...)` helper in `src/train.py` instantiates `SignalSampler` with parameters from `training.data_kwargs`.

---

## Task Definition (`src/tasks.py`)

The `SignalConvolutionTask` defines the learning problem: **in-context inference of a latent FIR filter**.

- **Setup**
  - For each batch row `i`:
    - Sample a random FIR filter \(h_i\) of length `fir_len` with Gaussian entries scaled by \(1 / \sqrt{L}\), where \(L =\) `fir_len`.
    - Construct a sequence of pairs \((x_t, y_t)\), where:
      - \(x_t\) is a random periodic input signal.
      - \(y_t = h_i * x_t\) (circular convolution).
  - At test positions, the model must predict \(y_t\) for new \(x_t\) given only in-context examples for that same \(h_i\).

- **Domain handling**
  - **Time-domain task (`domain="time"`)**
    - Inputs `xs` have shape `(B, T, p)`.
    - Internally:
      - Compute `X = rfft(xs)` and `H = rfft(h)`.
      - Multiply in the frequency domain: `Y = X * H`.
      - Transform back with `irfft` to obtain time-domain outputs `ys` with shape `(B, T, p)`.
  - **Frequency-domain task (`domain="freq"`)**
    - Inputs `xs` have shape `(B, T, 2 * p_fft)` and are interpreted as interleaved real/imaginary parts.
    - Internally:
      - Reconstruct complex spectra from the interleaved encoding.
      - Compute `H = rfft(h)` and `Y = X * H`.
      - Return outputs in the same interleaved complex format `(B, T, 2 * p_fft)`.

- **Metrics**
  - **`SignalConvolutionTask.get_metric()`** and **`get_training_metric()`** both return mean squared error (MSE).

In `src/train.py`, `get_task_sampler("signal_conv", ...)` returns a factory that creates a fresh `SignalConvolutionTask` each step using the keyword arguments from `training.task_kwargs`.

---

## Training Loop (`src/train.py`)

The main training logic is implemented in the `train(model, cfg)` function.

- **Configuration loading**
  - Reads the YAML config from `CONFIG_PATH` using `yaml.safe_load`.
  - Checks that `cfg["model"]["family"] == "gpt2"`.
  - Creates an output directory and writes a resolved `config.yaml` for reproducibility.

- **Model and optimization**
  - Builds the transformer with `build_model` from `src/models.py`, using `cfg["model"]`.
  - Uses Adam with learning rate `cfg["training"]["learning_rate"]`.

- **Curriculum**
  - Uses a `Curriculum` object (from `src/curriculum.py`) configured by `training.curriculum.points` to control the number of in-context examples `n_points` over training steps.

- **Data and task sampling**
  - Instantiates `data_sampler` via `get_data_sampler` with `training.data` and `training.data_kwargs`.
  - Instantiates a `task_sampler` factory via `get_task_sampler` with `training.task` and `training.task_kwargs`.
  - Each training step:
    - Samples `xs` with shape `(batch_size, n_points, n_dims)`.
    - Creates a `SignalConvolutionTask` and computes `ys = task.evaluate(xs)`.

- **Loss**
  - For `loss_space: "time"` (used in both provided configs):
    - Predictions and targets are converted (if needed) to time domain before computing MSE, ensuring a common evaluation space for both time- and frequency-domain runs.

- **Logging and checkpoints**
  - Logs to Weights & Biases (if `test_run: false`):
    - **`overall_loss`**
    - **`n_points`**
  - Saves:
    - Training state (`state.pt`) every `save_every_steps` to allow resuming.
    - Model snapshots every `keep_every_steps`.

---

## Practical Tips

- **Matching dimensions**
  - **Time domain**:
    - Ensure `model.n_dims == training.data_kwargs.p`.
  - **Frequency domain**:
    - Ensure `model.n_dims == 2 * (p // 2 + 1)` to match the interleaved real/imaginary output of `torch.fft.rfft`.

- **Fair comparisons**
  - When comparing time vs frequency experiments, keep the following aligned:
    - `p`, `batch_size`, `learning_rate`, `train_steps`
    - Curriculum (`training.curriculum.points`)
    - Filter length `fir_len` (if you want directly comparable tasks)

Use the descriptions above of `config_time.yaml`, `config_freq.yaml`, `samplers.py`, `tasks.py`, and `train.py` as the main reference when modifying or extending experiments.

---

## Regenerating Plots

To regenerate the plots, run the cells in `src/eval.ipynb` and read the comments provided in the notebook.

