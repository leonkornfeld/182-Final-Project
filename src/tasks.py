import math

import torch

from typing import Optional, Iterable, Literal

Domain = Literal["time", "freq"]  # "time" or "freq"

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

class Task:
    def __init__(self, n_dims, batch_size, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.seeds = seeds


def get_task_sampler(task_name: str, n_dims: int, bsize: int, **kwargs):
    if task_name != "signal_conv":
        raise NotImplementedError
    return lambda **task_kwargs: SignalConvolutionTask(n_dims, bsize, **{**kwargs, **task_kwargs})


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
    Each batch row has its own random FIR filter h_i.
    Model must infer h_i from input/output pairs (in-context learning).
    Filters are Gaussian with variance scaled by 1/L.
    """

    def __init__(
        self,
        n_dims: int,
        batch_size: int,
        *,
        p: int,
        fir_len: int,
        domain: Domain = "time",
        seeds: Optional[Iterable[int]] = None,
        device = "cuda"
    ):
        super().__init__(n_dims, batch_size, seeds)

        assert domain in ("time", "freq")
        self.p = p
        self.fir_len = fir_len
        self.domain = domain

        # consistency check
        if domain == "time":
            assert n_dims == p
        else:
            assert n_dims == 2 * p

        self.device = device
        self.h = self._sample_firs(batch_size, seeds).to(self.device)
        

    def _sample_firs(self, B, seeds):
        L = self.fir_len

        if seeds is None:
            h = torch.randn(B, L, device=self.device)
        else:
            raise ValueError("This version of SignalConvolutionTask does not support seeds.")
        # else:
        #     rows = []
        #     for s in seeds:
        #         g = torch.Generator(device=self.device).manual_seed(int(s) + 1)
        #         rows.append(torch.randn((1, L), generator=g, device=self.device))
            # h = torch.cat(rows, 0)

        # Normalize each FIR
        h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        return h

    @torch.no_grad()
    def evaluate(self, xs: torch.Tensor) -> torch.Tensor:
        """
        xs:
            time → (B, T, p)
            freq → (B, T, 2p) interleaved
        return y with SAME shape.
        """
        B, T, _ = xs.shape
        assert B == self.b_size

        # print(self.h)

        if self.domain == "time":
            xs_flat = xs.reshape(B * T, self.p).to(self.device)
            # print(xs_flat)
            X = torch.fft.fft(xs_flat, n=self.p, dim=-1).reshape(B, T, self.p)
            H = torch.fft.fft(self.h, n=self.p, dim=-1).unsqueeze(1)
            Y = torch.fft.ifft(X * H, dim=-1).real
            return Y

        else:  # frequency domain
            re = xs[:, :, 0::2].to(self.device)
            im = xs[:, :, 1::2].to(self.device)
            X = torch.complex(re, im)

            H = torch.fft.fft(self.h, n=self.p, dim=-1).unsqueeze(1)
            Y = X * H

            out = torch.empty(B, T, 2 * self.p, device=self.device, dtype=re.dtype)
            out[:, :, 0::2] = Y.real
            out[:, :, 1::2] = Y.imag
            return out

    @staticmethod
    def get_metric():
        return mean_squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    


   # def _sample_firs(self, B: int, seeds: Optional[Iterable[int]]):
    #     """
    #     Gaussian FIR:
    #         h[n] ~ N(0, 1/L)
    #     So:
    #         E[||h||^2] ≈ 1
    #     No L1 / L2 normalization.
    #     """
    #     L = self.fir_len
    #     scale = 1.0 / math.sqrt(L)

    #     if seeds is None:
    #         return torch.randn(B, L, device=self.device) * scale

    #     rows = []
    #     for s in seeds:
    #         g = torch.Generator(device=self.device).manual_seed(int(s) + 1)
    #         rows.append(torch.randn((1, L), generator=g, device=self.device) * scale)

    #     return torch.cat(rows, 0)