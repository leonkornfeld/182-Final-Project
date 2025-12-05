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
# def interleaved_to_complex(vec_2p: torch.Tensor) -> torch.Tensor:
#     """
#     vec_2p: (B, 2p) real, [Re0,Im0,Re1,Im1,...]
#     Returns: (B, p) complex
#     """
#     B, two_p = vec_2p.shape
#     p = two_p // 2
#     re = vec_2p[:, 0::2]
#     im = vec_2p[:, 1::2]
#     return torch.complex(re, im)


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
        device: str = "cuda",
        freq_representation: Literal["complex", "mag_phase"] = "complex",
    ):
        """
        Args:
            n_dims: input/output dimensionality (p for time, 2*p_fft for freq)
            batch_size: number of sequences in a batch
            p: signal period
            fir_len: FIR filter length
            domain: \"time\" or \"freq\" (input/output domain)
            freq_representation:
                - \"complex\": frequency domain uses interleaved Re/Im (default, backwards compatible)
                - \"mag_phase\": frequency domain uses interleaved magnitude/phase
        """
        super().__init__(n_dims, batch_size, seeds)

        assert domain in ("time", "freq")
        self.p = p
        self.p_fft = self.p // 2 + 1
        self.fir_len = fir_len
        self.domain = domain
        self.freq_representation = freq_representation

        # consistency check
        if domain == "time":
            assert n_dims == p
        else:
            assert n_dims == 2 * self.p_fft

        self.device = device
        self.h = self._sample_firs(batch_size, seeds).to(self.device)

    def _sample_firs(self, B, seeds):
        L = self.fir_len

        if seeds is None:
            h = torch.randn(B, L, device=self.device)
        else:
            raise ValueError("This version of SignalConvolutionTask does not support seeds.")

        # Normalize each FIR
        # h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        h = h / math.sqrt(L)
        return h

    @torch.no_grad()
    def evaluate(self, xs: torch.Tensor) -> torch.Tensor:
        """
        xs:
            time → (B, T, p)
            freq → (B, T, 2p_fft) interleaved

        For frequency-domain tasks we support two output encodings:
            - \"complex\": interleaved [Re0, Im0, Re1, Im1, ...]
            - \"mag_phase\": interleaved [|Y0|, arg(Y0), |Y1|, arg(Y1), ...]

        Returns:
            y with the SAME shape as xs.
        """
        B, T, _ = xs.shape
        assert B == self.b_size

        if self.domain == "time":
            xs_flat = xs.reshape(B * T, self.p).to(self.device)
            X = torch.fft.rfft(xs_flat, n=self.p, dim=-1, norm="ortho").reshape(B, T, self.p_fft)
            H = torch.fft.rfft(self.h, n=self.p, dim=-1, norm="ortho").unsqueeze(1)
            Y = torch.fft.irfft(X * H, dim=-1, norm="ortho").real
            return Y

        # frequency-domain case
        if self.freq_representation == "mag_phase":
            # Treat xs as interleaved [magnitude, phase] with phase in (-π, π]
            x_mag = xs[:, :, 0::2].to(self.device)
            x_phase = xs[:, :, 1::2].to(self.device)

            H = torch.fft.rfft(self.h, n=self.p, dim=-1, norm="ortho").unsqueeze(1)
            h_mag = torch.abs(H)
            h_phase = torch.angle(H)  # natural range (-π, π]

            y_mag = x_mag * h_mag
            y_phase = x_phase + h_phase

            out = torch.empty(B, T, 2 * self.p_fft, device=self.device, dtype=x_mag.dtype)
            out[:, :, 0::2] = y_mag
            out[:, :, 1::2] = y_phase
            return out

        # Default: complex interleaved representation (backwards compatible)
        re = xs[:, :, 0::2].to(self.device)
        im = xs[:, :, 1::2].to(self.device)
        X = torch.complex(re, im)

        H = torch.fft.rfft(self.h, n=self.p, dim=-1, norm="ortho").unsqueeze(1)
        Y = X * H

        out = torch.empty(B, T, 2 * self.p_fft, device=self.device, dtype=re.dtype)
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