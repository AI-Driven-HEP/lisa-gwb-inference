"""Amplitude schedule for spectral separator training.

A multiplicative factor scales the injected signal power. Its minimum
decreases over the first 100 epochs, shifting the training distribution
from strong signals toward weaker ones. The training pool reuses that
schedule when calculating each epoch's sample weights.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CurriculumConfig:
    E_curr: int = 100          # duration of the curriculum phase, in epochs
    A0: float = 1e2            # starting minimum amplitude scale factor
    A_m: float = 1e-2          # final (target) minimum amplitude scale factor
    A_max: float = 1e2         # global maximum amplitude scale factor
    f_sigma: float = 0.35      # sigma as a fraction of the dynamic range
    f_mu0: float = 1.0         # mu offset (in units of sigma) from the max, at epoch 1
    f_mu1: float = 0.5         # mu offset (in units of sigma) from the min, at epoch E_curr
    eps_ln_sigma: float = 1e-6  # floor on sigma_ln to avoid distribution collapse


class CurriculumAmplitudeSampler:
    """
    Usage:
        curriculum = CurriculumAmplitudeSampler(CurriculumConfig())
        for epoch in range(1, n_epochs + 1):
            for batch in loader:
                A = curriculum.sample(epoch)      # scalar, or call once per sample
                signal *= A
                ...
    """

    def __init__(self, config: CurriculumConfig = CurriculumConfig(), rng: np.random.Generator = None):
        self.cfg = config
        self.rng = rng or np.random.default_rng()

    def tau(self, epoch: int) -> float:
        """Normalized curriculum progression, Eq. 18. epoch is 1-indexed."""
        cfg = self.cfg
        return min(1.0, (epoch - 1) / max(1, cfg.E_curr - 1))

    def a_min(self, epoch: int) -> float:
        """Effective minimum amplitude for this epoch, Eq. 19-20."""
        cfg = self.cfg
        t = self.tau(epoch)
        a_interp_log10 = (1 - t) * math.log10(cfg.A0) + t * math.log10(cfg.A_m)
        a_interp = 10 ** a_interp_log10
        return min(cfg.A_max, max(cfg.A_m, a_interp))

    def sigma_ln(self, epoch: int) -> float:
        """Eq. 21."""
        cfg = self.cfg
        a_ln_min = math.log(self.a_min(epoch))
        a_ln_max = math.log(cfg.A_max)
        delta_ln = a_ln_max - a_ln_min
        return max(cfg.eps_ln_sigma, cfg.f_sigma * delta_ln)

    def mu_ln(self, epoch: int) -> float:
        """Eq. 22."""
        cfg = self.cfg
        t = self.tau(epoch)
        a_ln_min = math.log(self.a_min(epoch))
        a_ln_max = math.log(cfg.A_max)
        sigma = self.sigma_ln(epoch)

        mu_start = a_ln_max - cfg.f_mu0 * sigma
        mu_end = a_ln_min + cfg.f_mu1 * sigma
        return (1 - t) * mu_start + t * mu_end

    def sample(self, epoch: int, size: int = None):
        """
        Draw the final amplitude scale factor A for this epoch, Eq. 23.
        Returns a python float if size is None, else a numpy array of shape (size,).
        """
        cfg = self.cfg
        mu = self.mu_ln(epoch)
        sigma = self.sigma_ln(epoch)
        a_min = self.a_min(epoch)

        a_ln_sampled = self.rng.normal(mu, sigma, size=size)
        a_sampled = np.exp(a_ln_sampled)
        A = np.clip(a_sampled, a_min, cfg.A_max)
        return float(A) if size is None else A
