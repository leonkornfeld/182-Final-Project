import math

import torch

from typing import Optional, Iterable, Literal

Domain = Literal["time", "freq"]  # "time" or "freq"

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()



class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError

def get_task_sampler(
    task_name: str,
    n_dims: int,
    bsize: int,
    **kwargs,
):
    names_to_classes = {
        "signal_conv": SignalConvolutionTask,
    }
    if task_name not in names_to_classes:
        raise NotImplementedError(f"Unknown task: {task_name}")
    cls = names_to_classes[task_name]
    # Return a callable that instantiates a task (keeps parity with your original code)
    return lambda **task_kwargs: cls(n_dims, bsize, **{**kwargs, **task_kwargs})


# ---------- Helpers ----------
def interleaved_to_complex(vec_2p: torch.Tensor) -> torch.Tensor:
    """
    vec_2p: (B, 2p) real, [Re0,Im0,Re1,Im1,...]
    Returns: (B, p) complex
    """
    B, two_p = vec_2p.shape
    p = two_p // 2
    re = vec_2p[:, 0::2]
    im = vec_2p[:, 1::2]
    return torch.complex(re, im)


def complex_to_interleaved(z: torch.Tensor) -> torch.Tensor:
    """
    z: (B, p) complex
    Returns: (B, 2p) real, interleaved [Re0,Im0,...]
    """
    B, p = z.shape
    out = torch.empty(B, 2 * p, device=z.device, dtype=z.real.dtype)
    out[:, 0::2] = z.real
    out[:, 1::2] = z.imag
    return out

# ---------- The signal convolution task ----------
class SignalConvolutionTask(Task):
    """
    One task per batch row:
      - Sample a single FIR h_i for row i (deterministically if seeds provided).
      - For each prompt point t, compute y_{i,t} = (h_i ⊛ x_{i,t})_circ (length p).
      - Return ys with SAME shape as xs.
    Domain:
      - time: xs, ys shapes (B, points, p)
      - freq: xs, ys shapes (B, points, 2p) interleaved
    """
    def __init__(
        self,
        n_dims: int,
        batch_size: int,
        *,
        p: int,
        fir_len: int,
        domain: Domain = "time",
        fir_dist: str = "normal",      # "normal" | "uniform"
        normalize_fir: bool = True,
        seeds: Optional[Iterable[int]] = None,
        device: str = "cuda",
    ):
        super().__init__(n_dims, batch_size, None, seeds)
        assert domain in ("time", "freq")
        self.p = int(p)
        self.fir_len = int(fir_len)
        self.domain = domain
        self.fir_dist = fir_dist
        self.normalize_fir = normalize_fir
        self.device = device

        # Consistency check
        if domain == "time":
            assert n_dims == self.p
        else:
            assert n_dims == 2 * self.p

        # Build FIRs deterministically if seeds are provided
        self.h = self._sample_firs_deterministic(batch_size, seeds)

    def _sample_firs_deterministic(self, B: int, seeds: Optional[Iterable[int]]) -> torch.Tensor:
        if seeds is None:
            # Use current RNG state
            if self.fir_dist == "uniform":
                h = 2 * torch.rand(B, self.fir_len, device=self.device) - 1.0
            else:
                h = torch.randn(B, self.fir_len, device=self.device)
        else:
            seeds = list(seeds)
            assert len(seeds) == B
            rows = []
            for s in seeds:
                g = torch.Generator(device=self.device).manual_seed(int(s) + 1)  # +1 offset vs data seeds
                if self.fir_dist == "uniform":
                    rows.append(2 * torch.rand((1, self.fir_len), generator=g, device=self.device) - 1.0)
                else:
                    rows.append(torch.randn((1, self.fir_len), generator=g, device=self.device))
            h = torch.cat(rows, dim=0)  # (B, L)

        if self.normalize_fir:
            denom = h.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
            h = h / denom
        return h  # (B, L)

    @torch.no_grad()
    def evaluate(self, xs: torch.Tensor) -> torch.Tensor:
        """
        xs:
          time: (B, points, p)
          freq: (B, points, 2p) interleaved
        Returns ys with the SAME shape as xs.
        """
        B, T, D = xs.shape
        assert B == self.b_size, "Batch size mismatch."

        if self.domain == "time":
            # For each row, same h across points
            xs_flat = xs.reshape(B * T, self.p)
            # Convolve row-wise by grouping
            # Strategy: FFT per row
            X = torch.fft.fft(xs_flat, n=self.p, dim=-1).reshape(B, T, self.p)  # (B,T,p)
            H = torch.fft.fft(self.h, n=self.p, dim=-1).unsqueeze(1)            # (B,1,p)
            Y = torch.fft.ifft(X * H, dim=-1).real                               # (B,T,p)
            return Y
        else:
            # Decode interleaved X -> complex, convolve, then re-encode
            p = self.p
            re = xs[:, :, 0::2]
            im = xs[:, :, 1::2]
            X = torch.complex(re, im)                                           # (B,T,p)
            H = torch.fft.fft(self.h, n=p, dim=-1).unsqueeze(1)                 # (B,1,p)
            Y = X * H                                                            # (B,T,p) (still complex)
            # Back to interleaved real/imag
            Y_real = torch.empty(B, T, 2 * p, device=xs.device, dtype=re.dtype)
            Y_real[:, :, 0::2] = Y.real
            Y_real[:, :, 1::2] = Y.imag
            return Y_real

    @staticmethod
    def get_metric():
        return mean_squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error