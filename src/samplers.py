import math

# sampler.py
import math
from typing import Optional, Iterable, List, Literal

import torch

Domain = Literal["time", "freq"]  # "time" => dim=p, "freq" => dim=2p


class DataSampler:
    def __init__(self, n_dims: int):
        self.n_dims = n_dims

    def sample_xs(
        self,
        n_points: int,
        b_size: int,
        n_dims_truncated: Optional[int] = None,
        seeds: Optional[Iterable[int]] = None,
    ):
        raise NotImplementedError


def get_data_sampler(data_name: str, n_dims: int, **kwargs):
    names_to_classes = {
        "signal": SignalSampler,  # <- new
    }
    if data_name not in names_to_classes:
        raise NotImplementedError(f"Unknown sampler: {data_name}")
    return names_to_classes[data_name](n_dims, **kwargs)


def fft_to_interleaved(z: torch.Tensor) -> torch.Tensor:
    """
    z: (B, p) complex tensor (torch.complex64/128) or real with last dim length p (after fft).
    Returns: (B, 2p) real tensor interleaving [Re0, Im0, Re1, Im1, ...]
    """
    if z.is_complex():
        re, im = z.real, z.imag
    else:
        # if someone passes real by mistake, treat imag as zeros
        re, im = z, torch.zeros_like(z)
    B, p = re.shape
    out = torch.empty(B, 2 * p, device=z.device, dtype=re.dtype)
    out[:, 0::2] = re
    out[:, 1::2] = im
    return out


def fft_to_magphase_interleaved(z: torch.Tensor) -> torch.Tensor:
    """
    z: (B, p) complex tensor from rfft.
    Returns: (B, 2p) real tensor interleaving [|Z0|, arg(Z0), |Z1|, arg(Z1), ...]
    with phase in the natural range (-π, π].
    """
    assert z.is_complex()
    mag = torch.abs(z)
    phase = torch.angle(z)

    B, p = mag.shape
    out = torch.empty(B, 2 * p, device=z.device, dtype=mag.dtype)
    out[:, 0::2] = mag
    out[:, 1::2] = phase
    return out


class SignalSampler(DataSampler):
    """
    Produces x prompts as *full signals*:
      - domain=="time":  xs shape (B, n_points, p)
      - domain=="freq":  xs shape (B, n_points, 2p) with either:
            * [Re0, Im0, Re1, Im1, ...]       if freq_representation == "complex"
            * [|X0|, arg(X0), |X1|, arg(X1), ...] if freq_representation == "mag_phase"

    The *dimension passed in n_dims must match* p (time) or 2p (freq) in your config.
    """

    def __init__(
        self,
        n_dims: int,
        *,
        p: int,  # signal period
        domain: Domain = "time",
        amp_dist: str = "normal",  # "normal" | "uniform"
        amp_std: float = 1.0,
        amp_max: float = 1.0,
        num_freqs: Optional[int] = None,  # number of harmonics in inputs; None => p
        device: str = "cuda",
        freq_representation: Literal["complex", "mag_phase"] = "complex",
    ):
        super().__init__(n_dims)
        assert domain in ("time", "freq")
        self.p = int(p)
        self.p_fft = self.p // 2 + 1  # rfft length
        self.domain = domain
        self.amp_dist = amp_dist
        self.amp_std = float(amp_std)
        self.amp_max = float(amp_max)
        self.num_freqs = int(num_freqs) if (num_freqs is not None) else None
        self.device = device
        self.freq_representation = freq_representation

        # Consistency check: n_dims must be p (time) or 2p (freq)
        if domain == "time":
            assert n_dims == self.p, f"n_dims({n_dims}) must equal p({self.p}) for time domain."
        else:
            # rfft → length p_fft, interleaved → 2 * p_fft
            assert n_dims == 2 * self.p_fft, (
                f"n_dims({n_dims}) must equal 2 * p_fft({2 * self.p_fft}) for freq domain."
            )
        # Precompute frequency/time grids
        K = self.p if (self.num_freqs is None or self.num_freqs > self.p) else self.num_freqs
        self.K = K
        self.k_vals = torch.arange(0, K, dtype=torch.float32, device=self.device).view(1, K, 1)
        self.omega = 2 * torch.pi * self.k_vals / self.p
        self.t = torch.arange(self.p, dtype=torch.float32, device=self.device).view(1, 1, self.p)

    def _sample_time_signal_batch(self, B: int) -> torch.Tensor:
        """
        Returns x_batch in time domain: (B, p)
        """
        phases = 2 * torch.pi * torch.rand((B, self.K, 1), device=self.device)  # (B,K,1)

        if self.amp_dist == "uniform":
            amplitudes = (2 * self.amp_max) * torch.rand((B, self.K, 1), device=self.device) - self.amp_max
        else:
            amplitudes = self.amp_std * torch.randn((B, self.K, 1), device=self.device)

        sines = torch.sin(self.omega * self.t + phases)  # (B,K,p) via broadcast
        x_time = (amplitudes * sines).sum(dim=1)  # (B,p)
        return x_time

    def _encode(self, x_time: torch.Tensor) -> torch.Tensor:
        """
        Encode time-domain signals to configured domain.
        """
        if self.domain == "time":
            return x_time
        # freq domain: choose representation for FFT output
        X = torch.fft.rfft(x_time, n=self.p, dim=-1, norm="ortho")
        if self.freq_representation == "mag_phase":
            return fft_to_magphase_interleaved(X)
        return fft_to_interleaved(X)

    def sample_xs(
        self,
        n_points: int,
        b_size: int,
        seeds: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        """
        Returns xs_b with shape:
          time: (B, n_points, p)
          freq: (B, n_points, 2p)
        Seeding:
          - If seeds is provided, len(seeds)==B and each batch row is reproducible.
          - Each point uses a deterministic offset to the given seed.
        """
        B, T = int(b_size), int(n_points)
        xs_b = torch.zeros(B, T, self.n_dims, device=self.device)

        if seeds is None:
            # single global RNG; still deterministic if user set torch.manual_seed outside
            # g0 = make_gen(torch.seed(), self.device)  # use current RNG state
            scale = math.sqrt(self.p)
            for t in range(T):
                x_time = self._sample_time_signal_batch(B)  # (B,p)
                # norms = x_time.norm(dim=1, keepdim=True) + 1e-8
                # x_time = (x_time / norms) #* scale
                means = x_time.mean(dim=1, keepdim=True)
                stds = x_time.std(dim=1, keepdim=True) + 1e-8
                x_time = (x_time - means) / stds
                xs_b[:, t, :] = self._encode(x_time)

        # norms = xs_b.norm(dim=2, keepdim=True) + 1e-8
        # xs_normed = xs_b / norms
        
        return xs_b
    

# else:
        #     seeds = list(seeds)
        #     assert len(seeds) == B, f"len(seeds) must equal batch size ({B})."
        #     for i, s in enumerate(seeds):
        #         for t in range(T):
        #             # each (i,t) gets its own substream
        #             gen = make_gen(int(s) * 1000003 + t, self.device)
        #             x_time = self._sample_time_signal_batch(1, gen)  # (1,p)
        #             x_enc = self._encode(x_time)                         # (1,p) or (1,2p)
        #             xs_b[i, t, :] = x_enc[0] 


# def make_gen(seed: int, device: str = "cuda") -> torch.Generator:
#     g = torch.Generator(device=device)
#     g.manual_seed(int(seed) & 0xFFFFFFFF)
#     return g
