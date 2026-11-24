"""
LISA adaptation of the multi-scale, multi-headed autoencoder of Einsle et al.,
Phys. Rev. D 112, 063056 (2025), Sec. III.A and App. A.
https://doi.org/10.1103/hs9b-drwx
The follow-up study also uses the architecture: https://arxiv.org/abs/2608.00281.

Tensor shapes (TDI channels share the frequency axis):
    x            in R^{B x (M n_ch) x N}  (M log-spectra per TDI channel,
                                           TDI channel varying SLOWEST)
    z^l          in R^{B x c_l x n_l}     (latent at scale l = 1..L)
    n_hat, s_hat in R^{B x n_ch x N}      (predicted noise / signal log-spectra)

Scale bookkeeping (Appendix A):
    n_1 = N,          n_{l+1} = floor(n_l / 2)
    c_1 = C_base,     c_{l+1} = 2 * c_l
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lisa_gwb.simulation import (
    check_normalisation_stamp,
    denormalize_log_psd,
    normalisation_stamp,
    normalize_log_psd,
)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

class InceptionBlock1D(nn.Module):
    """
    Multi-scale feature extractor (paper Eq. A1-A2): parallel 1D conv branches
    with different kernel sizes, concatenated and projected back to
    `out_channels` via a 1x1 convolution.

    in_channels -> [branch_k for k in kernel_sizes] -> concat -> 1x1 conv -> out_channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        mid_channels: Optional[int] = None,
    ):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_channels, mid_channels, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.project = nn.Conv1d(mid_channels * len(kernel_sizes), out_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, n)
        feats = [branch(x) for branch in self.branches]   # each (B, mid, n)
        feats = torch.cat(feats, dim=1)                     # (B, mid*|K|, n)
        return self.act(self.project(feats))                # (B, out, n)


class DownsampleBlock1D(nn.Module):
    """
    Down() operator (paper Eq. A3-A5): max-pool by 2 along the frequency axis,
    then a 1x1 conv that doubles the channel count.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert out_channels == 2 * in_channels, "paper doubles channels at each scale"
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.expand = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(self.pool(x))


class MergeBlock(nn.Module):
    """
    MergeBlock (paper Eq. A10): two Conv1D(k=3) with BatchNorm+ReLU in between,
    applied to the concatenation of (skip feature, upsampled decoder feature,
    [optional noise-conditioning feature]).

        Conv1D_3 -> BatchNorm1d -> ReLU -> Conv1D_3
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act(self.bn(x))
        return self.conv2(x)


class UpsampleBlock1D(nn.Module):
    """
    Up() operator (paper Eq. A8): a 1D transposed convolution (kernel=2, stride=2)
    that simultaneously halves the channel count and doubles the frequency
    resolution. If the resulting length does not match the target scale's n_l
    (this happens whenever N is not exactly divisible by 2^L, e.g. N=1005),
    it is fixed with linear interpolation (Interp(), Eq. A9).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != target_len:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x


class CorrectorBlock(nn.Module):
    """
    Corrector module phi^l (paper Eq. A13-A15): concatenates the total and
    noise latents at scale l along the channel axis, then processes them with
    a two-layer 1x1-conv MLP to produce a learned correction map of the same
    shape as z^l. Used to refine the naive latent subtraction
    z_total^l - z_noise^l (Eq. A16), since the paper reports that subtraction
    alone does not reliably isolate the signal.
    """

    def __init__(self, channels: int, hidden_channels: Optional[int] = None):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.fc1 = nn.Conv1d(2 * channels, hidden_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(hidden_channels, channels, kernel_size=1)

    def forward(self, z_total: torch.Tensor, z_noise: torch.Tensor) -> torch.Tensor:
        u = torch.cat([z_total, z_noise], dim=1)   # (B, 2c_l, n_l)
        return self.fc2(self.act(self.fc1(u)))      # (B, c_l, n_l)


# --------------------------------------------------------------------------- #
# Multi-scale encoder
# --------------------------------------------------------------------------- #

class MultiScaleEncoder(nn.Module):
    """
    Encoder_total / Encoder_noise (paper Eq. 7, 9 and Appendix A "Encoders").

    Produces a dict {1: z^1, ..., L: z^L} of latent tensors at L successive
    scales, each with n_{l+1} = floor(n_l/2) frequency resolution and
    c_{l+1} = 2*c_l channels. The same class is reused for both Encoder_total
    (in_channels = M) and Encoder_noise (in_channels = 1), since only the
    first Inception block depends on the number of input channels.
    """

    def __init__(
        self,
        in_channels: int,
        n_freq: int,
        n_layers: int = 3,
        base_channels: int = 64,
        kernel_sizes: Sequence[int] = (3, 5, 7),
    ):
        super().__init__()
        self.n_layers = n_layers
        self.scale_lengths = self._compute_lengths(n_freq, n_layers)
        self.scale_channels = [base_channels * (2 ** l) for l in range(n_layers)]

        # scale-1 Inception block: in_channels -> c_1
        self.stem = InceptionBlock1D(in_channels, self.scale_channels[0], kernel_sizes)

        # deeper scales: Inc (channel-preserving) + Down (pool + channel-doubling)
        self.inc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for l in range(n_layers - 1):
            c_l = self.scale_channels[l]
            c_next = self.scale_channels[l + 1]
            self.inc_blocks.append(InceptionBlock1D(c_l, c_l, kernel_sizes))
            self.down_blocks.append(DownsampleBlock1D(c_l, c_next))

    @staticmethod
    def _compute_lengths(n_freq: int, n_layers: int) -> List[int]:
        lengths = [n_freq]
        for _ in range(n_layers - 1):
            lengths.append(lengths[-1] // 2)
        return lengths

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        # x: (B, C_in, N)
        z = {}
        z[1] = self.stem(x)
        for l in range(1, self.n_layers):
            inc_out = self.inc_blocks[l - 1](z[l])
            z[l + 1] = self.down_blocks[l - 1](inc_out)
        return z  # {1: (B,c1,n1), ..., L: (B,cL,nL)}


# --------------------------------------------------------------------------- #
# Decoder (shared structure for Decoder_noise and Decoder_signal)
# --------------------------------------------------------------------------- #

class MultiScaleDecoder(nn.Module):
    """
    Decoder_noise / Decoder_signal (paper Eq. 8, 11 and Appendix A "Decoders").

    Traverses the latent hierarchy from the coarsest scale L down to the
    finest scale 1. At each step it upsamples the previous decoder state,
    merges it with the skip feature at that scale (MergeBlock), and finally
    projects the finest-scale features down to `out_channels` log-spectra, one
    per TDI channel (F3).

    `use_noise_conditioning=True` additionally concatenates a (resized)
    noise-estimate channel into every MergeBlock input. The paper mentions
    this as optional ("... and optionally the predicted noise n_hat");
    we use it for Decoder_signal only (it makes intuitive sense for the
    signal decoder to know what the estimated noise looks like), and leave
    it off for Decoder_noise. This is a design choice, not something the
    paper pins down explicitly — toggle freely.
    """

    def __init__(
        self,
        scale_channels: Sequence[int],
        scale_lengths: Sequence[int],
        use_noise_conditioning: bool = False,
        out_channels: int = 1,
        noise_channels: int = 1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.noise_channels = noise_channels
        self.n_layers = len(scale_channels)
        self.scale_channels = list(scale_channels)
        self.scale_lengths = list(scale_lengths)
        self.use_noise_conditioning = use_noise_conditioning

        self.up_blocks = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()
        for l in range(self.n_layers - 1, 0, -1):
            # going from scale l+1 -> l
            c_deep = self.scale_channels[l]        # index l corresponds to scale l+1 (0-indexed)
            c_shallow = self.scale_channels[l - 1]  # scale l
            self.up_blocks.append(UpsampleBlock1D(c_deep, c_shallow))

            merge_in = c_shallow + c_shallow  # skip (c_shallow) + upsampled (c_shallow)
            if use_noise_conditioning:
                merge_in += noise_channels  # the resized per-TDI-channel noise estimate
            self.merge_blocks.append(MergeBlock(merge_in, c_shallow))

        # projection head (Eq. A11): Conv1D_3 -> ReLU -> Conv1D_1  (channels -> 1)
        c1 = self.scale_channels[0]
        self.head_conv3 = nn.Conv1d(c1, c1, kernel_size=3, padding=1)
        self.head_act = nn.ReLU(inplace=True)
        self.head_conv1 = nn.Conv1d(c1, out_channels, kernel_size=1)

    def forward(
        self,
        seed_latents: Dict[int, torch.Tensor],
        skip_latents: Dict[int, torch.Tensor],
        noise_estimate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        seed_latents:  dict {1..L: tensor} — used to initialize z_dec^L (only
                       the coarsest entry, L, is actually used as the seed).
        skip_latents:  dict {1..L: tensor} — per-scale skip features merged
                       in at each decoding step (for Decoder_noise this is
                       z_total; for Decoder_signal this is the corrected
                       z_signal).
        noise_estimate: optional (B, n_ch, N) tensor, required if
                       use_noise_conditioning=True.
        """
        L = self.n_layers
        z_dec = seed_latents[L]  # (B, c_L, n_L)

        for idx, l in enumerate(range(L - 1, 0, -1)):
            target_len = self.scale_lengths[l - 1]
            z_up = self.up_blocks[idx](z_dec, target_len)  # (B, c_l, n_l)

            parts = [skip_latents[l], z_up]
            if self.use_noise_conditioning:
                assert noise_estimate is not None, "noise_estimate required when use_noise_conditioning=True"
                n_resized = F.interpolate(noise_estimate, size=target_len, mode="linear", align_corners=False)
                parts.append(n_resized)

            merged_in = torch.cat(parts, dim=1)
            z_dec = self.merge_blocks[idx](merged_in)  # (B, c_l, n_l)

        out = self.head_conv3(z_dec)
        out = self.head_act(out)
        return self.head_conv1(out)   # (B, out_channels, N)


# --------------------------------------------------------------------------- #
# Full MSMHAutoencoder
# --------------------------------------------------------------------------- #

class MSMHAutoencoder(nn.Module):
    """
    Full pipeline (paper Eqs. 7-11, Fig. 1):

        x --Encoder_total--> z_total
        z_total --Decoder_noise--> n_hat
        n_hat --Encoder_noise--> z_noise
        (z_total, z_noise) --Corrector (per scale)--> z_signal
        z_signal --Decoder_signal--> s_hat

    Parameters mirror Table I of the paper.
    """

    def __init__(
        self,
        M: int = 12,
        N: int = 1005,
        n_ch: int = 1,
        n_layers: int = 3,
        base_channels: int = 64,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        corrector_hidden: Optional[int] = None,
        noise_condition_signal_decoder: bool = True,
    ):
        super().__init__()
        self.M = M
        self.N = N
        self.n_ch = n_ch
        self.n_layers = n_layers

        # F3: x arrives as (B, M*n_ch, N) with the TDI channel varying slowest,
        # so Encoder_total sees M*n_ch input planes. Both decoder heads emit one
        # plane per TDI channel, and Encoder_noise consumes those n_ch planes.
        self.encoder_total = MultiScaleEncoder(
            in_channels=M * n_ch, n_freq=N, n_layers=n_layers,
            base_channels=base_channels, kernel_sizes=kernel_sizes,
        )
        self.encoder_noise = MultiScaleEncoder(
            in_channels=n_ch, n_freq=N, n_layers=n_layers,
            base_channels=base_channels, kernel_sizes=kernel_sizes,
        )

        scale_channels = self.encoder_total.scale_channels
        scale_lengths = self.encoder_total.scale_lengths

        self.decoder_noise = MultiScaleDecoder(
            scale_channels, scale_lengths, use_noise_conditioning=False,
            out_channels=n_ch, noise_channels=n_ch,
        )
        self.decoder_signal = MultiScaleDecoder(
            scale_channels, scale_lengths,
            use_noise_conditioning=noise_condition_signal_decoder,
            out_channels=n_ch, noise_channels=n_ch,
        )

        self.correctors = nn.ModuleList(
            [CorrectorBlock(c, hidden_channels=corrector_hidden) for c in scale_channels]
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, M*n_ch, N) — M consecutive log-spectra for each of n_ch TDI
        channels, N frequency bins, TDI channel varying SLOWEST on axis 1.

        Returns a dict with everything needed for the loss functions:
            n_hat, s_hat            : (B, n_ch, N) predicted noise / signal log-spectra
            z_total, z_noise, z_signal : dicts of per-scale latents (for latent /
                                          consistency losses)
        """
        z_total = self.encoder_total(x)                       # Eq. 7
        n_hat = self.decoder_noise(z_total, z_total)           # Eq. 8
        z_noise = self.encoder_noise(n_hat)                    # Eq. 9

        z_signal = {}
        for l in range(1, self.n_layers + 1):
            corr = self.correctors[l - 1](z_total[l], z_noise[l])
            z_signal[l] = (z_total[l] - z_noise[l]) + corr     # Eq. 10

        s_hat = self.decoder_signal(z_signal, z_signal, noise_estimate=n_hat)  # Eq. 11

        return {
            "n_hat": n_hat,
            "s_hat": s_hat,
            "z_total": z_total,
            "z_noise": z_noise,
            "z_signal": z_signal,
        }

    def encode_noise_latents(self, n: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Convenience: run Encoder_noise on an arbitrary (B, n_ch, N) noise
        spectrum (e.g. the *true* noise, needed for the latent-consistency
        loss)."""
        return self.encoder_noise(n)

    def encode_total_latents(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Convenience: run Encoder_total on an arbitrary (B, M*n_ch, N) input
        (e.g. the recombined x_hat, needed for the self-consistency loss)."""
        return self.encoder_total(x)


# Checkpoints and validation diagnostics


def save_checkpoint(path: str, model, **extra) -> None:
    """Write a model checkpoint stamped with the target representation (N1).

    A trained network is a function from normalised inputs to normalised
    outputs. Change (LOG_PSD_CENTER, LOG_PSD_SCALE) and the same weights now
    mean a different physical spectrum, with nothing in the state_dict to say
    so. Anything that saves weights should go through here.

    P5: the write is ATOMIC -- a temporary file in the same directory, then
    os.replace, which is atomic on POSIX within a filesystem. A job killed
    mid-write (a cluster walltime kill, an OOM) previously left a truncated
    .pt that torch.load cannot read, destroying the last good checkpoint as
    well as the one being written. Now the previous checkpoint survives
    untouched until the new one is complete on disk.
    """
    payload = dict(state_dict=model.state_dict(),
                   norm_stamp=normalisation_stamp(), **extra)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp",
                               dir=directory)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_checkpoint(path: str, model=None, allow_stamp_mismatch: bool = False):
    """Load a checkpoint, refusing one written under a different representation.

    Raises NormalisationMismatch on a stamp that differs from the constants in
    force, and on an UNSTAMPED file -- every unstamped checkpoint predates N1
    and is therefore known to be wrong. allow_stamp_mismatch exists only for
    forensics on an old artifact, never for producing a number.

    Returns the loaded dict; if `model` is given its weights are loaded too.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not allow_stamp_mismatch:
        check_normalisation_stamp(ckpt, path)
    if model is not None:
        model.load_state_dict(ckpt["state_dict"])
    return ckpt


def load_separators(checkpoints, mission):
    """Restore separator models for the paper's 512-bin A/E/T configuration."""
    models = []
    for path in checkpoints:
        checkpoint = load_checkpoint(path)
        geometry = (checkpoint["M"], checkpoint["N"], tuple(checkpoint["channels"]), checkpoint["n_avg"])
        if geometry != (1, 512, ("A", "E", "T"), mission.Nd):
            raise ValueError(f"{path} uses geometry {geometry}; expected the paper's A/E/T, 512-bin configuration.")
        model = MSMHAutoencoder(M=1, N=512, n_ch=3, n_layers=checkpoint["n_layers"],
                                base_channels=checkpoint["base_channels"])
        model.load_state_dict(checkpoint["state_dict"])
        models.append(model.eval())
    return models


def separate_spectra(models, spectra):
    """Separate spectra of shape (observations, 3, 512) in physical log10 PSD.

    Returns (signal, noise), with predictions averaged in log power.
    """

    inputs = torch.from_numpy(normalize_log_psd(spectra.astype(np.float64))).float()
    with torch.no_grad():
        predictions = [model(inputs) for model in models]
    signal, noise = (
        np.mean([denormalize_log_psd(pred[name].numpy().astype(np.float64))
                 for pred in predictions], axis=0)
        for name in ("s_hat", "n_hat")
    )
    return signal.astype(np.float32), noise.astype(np.float32)


class FrozenBinError(RuntimeError):
    """Raised when the network's signal head has stopped responding to the data
    over part of the band. See frozen_bin_report."""


def frozen_bin_report(s_hats: np.ndarray, rel_threshold: float = 1e-3) -> dict:
    """Diagnose bins where s_hat does not move when the data moves (8e).

    At n_layers = 3 one initialisation in three produced a signal head that
    emitted its output convolution's bias, bit-identically across every
    replicate, over 79 of 128 bins, because none of the eight channels of the
    ReLU feeding that convolution was active there. Held-out loss did not flag
    it -- that seed scored 0.400 against 0.372 and 0.390, and all three beat
    every n_layers = 2 run -- so nothing upstream of here will catch it.

    Two populations are counted separately, because they fail differently:

      exactly_zero: the replicate standard deviation is 0.0. Network outputs are
        float32, so bit-identical predictions give a variance that is exactly
        zero rather than merely small (verified over 4000 plausible frozen
        values and three array shapes). These bins reach the likelihood as
        sigma_log = 0 and are silently DROPPED by its validity mask -- the fit
        quietly loses them, and if every bin is frozen the likelihood becomes
        exactly flat and the sampler returns the prior.

      near_frozen: the standard deviation is positive but below rel_threshold
        times the median over live bins. These are worse, because they PASS a
        `var > 0` guard and arrive with a weight of 1/sigma^2. One replicate of
        two hundred differing by a single float32 ulp is enough to give a weight
        of order 1e15 and let that one bin carry the entire chi-squared.

    The default threshold is measured, not guessed. On trained checkpoints the
    smallest genuine per-bin sigma is 0.017 of the median at n_layers = 2 and
    0.085 at n_layers = 3, while a one-ulp near-frozen bin sits at 1.3e-5 of it
    -- a gap of more than three decades. 1e-3 sits inside that gap, seventeen
    times below the smallest real bin observed and seventy times above the
    degenerate case.

    s_hats: (n_replicates, N) -> dict with the two fractions, the two index
    arrays, and the median live sigma the threshold was taken against.
    """
    sd = np.asarray(s_hats).std(axis=0, ddof=1)
    exactly_zero = np.where(sd == 0.0)[0]
    live = sd[sd > 0]
    med = float(np.median(live)) if live.size else 0.0
    near = np.where((sd > 0.0) & (sd < rel_threshold * med))[0] if med > 0 else np.array([], int)
    n = sd.size
    return dict(
        n_bins=n,
        exactly_zero_idx=exactly_zero, near_frozen_idx=near,
        frac_exactly_zero=float(exactly_zero.size) / n,
        frac_near_frozen=float(near.size) / n,
        median_live_sigma=med, rel_threshold=rel_threshold,
        n_usable=int(n - exactly_zero.size - near.size),
    )


def frozen_bin_report_per_channel(s_hats: np.ndarray, n_ch: int,
                                   rel_threshold: float = 1e-3) -> dict:
    """frozen_bin_report run separately on each TDI channel (OF1, N1).

    A GLOBAL fraction averages the channels together and can hide a head that
    has gone dead in one of them. N1 gave the reason this is not a hypothetical:
    the T channel's signal target is about 40 per cent floor, and a floored bin
    contributes only through a one-sided hinge that is exactly zero while the
    prediction stays below the floor. So the signal head takes a third of its
    gradient from a channel where two fifths of the bins can supply no gradient
    at all -- which is precisely the condition under which a unit stops being
    updated, and precisely where 8e's frozen head would hide. A global average
    over three channels divides any such fraction by three before anyone sees it.

    s_hats is (n_replicates, n_ch * N) flattened CHANNEL-SLOWEST, the layout the
    rest of the bridge uses, or (n_replicates, n_ch, N). Returns a dict keyed by
    channel index plus "all" for the global figure, so callers that want the old
    number still have it.
    """
    a = np.asarray(s_hats)
    if a.ndim == 3:
        per = [a[:, i, :] for i in range(a.shape[1])]
        flat = a.reshape(a.shape[0], -1)
    else:
        if a.shape[1] % n_ch:
            raise ValueError(f"{a.shape[1]} columns is not divisible by n_ch={n_ch}")
        N = a.shape[1] // n_ch
        per = [a[:, i * N:(i + 1) * N] for i in range(n_ch)]
        flat = a
    out = {i: frozen_bin_report(p, rel_threshold) for i, p in enumerate(per)}
    out["all"] = frozen_bin_report(flat, rel_threshold)
    out["worst_channel"] = max(range(n_ch), key=lambda i: out[i]["frac_exactly_zero"]
                               + out[i]["frac_near_frozen"])
    return out


def check_no_frozen_bins(s_hats: np.ndarray, rel_threshold: float = 1e-3,
                          allow: bool = False, n_ch: int = 1) -> dict:
    """frozen_bin_report, raised as an error unless `allow` is set.

    Loud by default and quiet only on request: a frozen bin is not a noisy
    measurement, it is the absence of a measurement, and the pipeline should not
    be able to reach an MCMC without someone having decided what to do about it.
    """
    rep = frozen_bin_report(s_hats, rel_threshold)
    # OF1/N1: the per-channel breakdown is computed whenever there is more than
    # one channel and attached to the report, so the number that reaches a human
    # is never only the global average.
    if n_ch > 1:
        rep["per_channel"] = frozen_bin_report_per_channel(s_hats, n_ch, rel_threshold)
        worst = rep["per_channel"]["worst_channel"]
        w = rep["per_channel"][worst]
        rep["worst_channel"] = worst
        rep["worst_channel_frac"] = w["frac_exactly_zero"] + w["frac_near_frozen"]
    bad = rep["exactly_zero_idx"].size + rep["near_frozen_idx"].size
    if bad and not allow:
        raise FrozenBinError(
            "the signal head is not responding to the data in %d of %d bins "
            "(%.1f%% exactly frozen, %.1f%% near-frozen below %g x the median "
            "live sigma of %.3g). This is the 8e failure mode: the head emits "
            "its output-convolution bias where no channel of the preceding ReLU "
            "is active. Validation loss does not show it. Retrain or reseed; "
            "pass allow_frozen_bins=True only if you have decided these bins "
            "can be excluded from this training check."
            % (bad, rep["n_bins"], 100 * rep["frac_exactly_zero"],
               100 * rep["frac_near_frozen"], rel_threshold, rep["median_live_sigma"])
            + ("" if n_ch <= 1 else
               " Worst channel: index %d, %.1f%% of its bins."
               % (rep["worst_channel"], 100 * rep["worst_channel_frac"])))
    return rep


class UnresponsiveBinError(RuntimeError):
    """Raised when the signal head does not track the truth in bins where the
    truth is detectable. See responsiveness_report."""


RESPONSIVENESS_SLOPE_MIN = 0.1


RESPONSIVENESS_MIN_EVENTS = 32


def detectable_bins(n_trues: np.ndarray, s_trues: np.ndarray,
                    sigma_frac: np.ndarray, gate_threshold: float = 0.5,
                    frac_events: float = 0.5) -> np.ndarray:
    """Which bins carry a detectable signal in most of a set of events.

    Applies CAL2's criterion -- the injected signal's fraction of the total
    against that bin's own sigma_frac -- to each event, then keeps the bins that
    clear it in more than `frac_events` of them. That "in more than half the
    trials" aggregation is the same one CAL2 used to report its per-band bin
    counts, so the number this returns is comparable with that table.

    n_trues, s_trues: (n_events, n_ch, N) or (n_events, n_ch*N), NORMALIZED.
    sigma_frac: (n_ch, N) or (n_ch*N,) from multichannel_sigma_frac.
    Returns a boolean array shaped like one event's spectrum.
    """
    from lisa_gwb.losses import detectability_gate
    n = torch.as_tensor(np.asarray(n_trues), dtype=torch.float32)
    s = torch.as_tensor(np.asarray(s_trues), dtype=torch.float32)
    sf = torch.as_tensor(np.asarray(sigma_frac).reshape(s.shape[1:]),
                         dtype=torch.float32)
    with torch.no_grad():
        w = detectability_gate(n, s, sf).numpy()
    return (w > gate_threshold).mean(axis=0) > frac_events


def responsiveness_report(s_hats: np.ndarray, s_trues: np.ndarray,
                          detectable: Optional[np.ndarray] = None,
                          slope_threshold: float = RESPONSIVENESS_SLOPE_MIN,
                          min_events: int = RESPONSIVENESS_MIN_EVENTS) -> dict:
    """P1. Does s_hat MOVE when the truth moves, in the bins where it can?

    frozen_bin_report misses this by construction and CAL2 measured how badly.
    That guard flags a bin whose spread across REPLICATES OF ONE EVENT collapses
    -- the 8e signature, where the head emits its output-convolution bias
    bit-identically. In CAL2's failing band the replicate spread does not
    collapse at all: its minimum over 400 trials and 37 bins is 3.9e-3 of the
    median live sigma, four times above that guard's 1e-3 threshold, and its
    median is 0.873. Those bins respond to the noise draw perfectly well. What
    they do not respond to is the SIGNAL, and no statistic computed from
    repeated draws of a single event can see the difference, because the signal
    is the one thing held fixed across them.

    So the axis has to change: regress s_hat on s_true per bin across DISTINCT
    EVENTS, whose truths differ. The slope is then 1 for a bin that tracks and 0
    for a bin emitting a constant, and CAL2 measured exactly that separation --
    0.935 median in the core band against 0.001 to 0.002 at 0.28, 0.38 and
    0.48 Hz.

    ONLY WHERE THE SIGNAL IS DETECTABLE. A zero slope in a bin where nothing is
    recoverable is CORRECT behaviour, not a defect: CAL2's whole finding is that
    the network going flat above 30 mHz is what an oracle also does there.
    Flagging it would turn the guard into a generator of false alarms in half
    the band. `detectable` restricts the check to the bins where a zero slope
    means something has gone wrong; pass the mask from detectable_bins.

    s_hats, s_trues: (n_events, n_ch, N) or (n_events, n_ch*N). One prediction
        per DISTINCT event -- not replicates of one event, which is what
        frozen_bin_report takes and would make the slope undefined here.
    Returns a dict with the per-bin slope, its standard error, the r^2, the
    indices flagged, and the mask actually checked.
    """
    a = np.asarray(s_hats, dtype=float)
    b = np.asarray(s_trues, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"s_hats {a.shape} and s_trues {b.shape} must match")
    shape = a.shape[1:]
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    n_events = a.shape[0]
    if n_events < min_events:
        raise ValueError(
            f"responsiveness needs at least {min_events} distinct events to "
            f"estimate a slope, got {n_events}. See RESPONSIVENESS_MIN_EVENTS.")

    bc = b - b.mean(axis=0)
    ac = a - a.mean(axis=0)
    var_true = (bc ** 2).sum(axis=0)
    # A bin whose truth does not vary carries no information about the slope;
    # it is reported as nan and never flagged, rather than dividing by zero.
    ok = var_true > 0
    slope = np.full(a.shape[1], np.nan)
    slope[ok] = (ac[:, ok] * bc[:, ok]).sum(axis=0) / var_true[ok]
    resid = ac - slope[None, :] * bc
    dof = max(n_events - 2, 1)
    se = np.full(a.shape[1], np.nan)
    se[ok] = np.sqrt((resid[:, ok] ** 2).sum(axis=0) / dof / var_true[ok])
    var_hat = (ac ** 2).sum(axis=0)
    r2 = np.full(a.shape[1], np.nan)
    good = ok & (var_hat > 0)
    r2[good] = 1 - (resid[:, good] ** 2).sum(axis=0) / var_hat[good]

    if detectable is None:
        checked = np.ones(a.shape[1], bool)
    else:
        checked = np.asarray(detectable).reshape(-1).astype(bool)
        if checked.size != a.shape[1]:
            raise ValueError(
                f"detectable has {checked.size} bins, expected {a.shape[1]}")
    checked = checked & ok
    flagged = np.where(checked & (slope < slope_threshold))[0]
    return dict(
        slope=slope.reshape(shape), se=se.reshape(shape), r2=r2.reshape(shape),
        checked=checked.reshape(shape), flagged_idx=flagged,
        n_checked=int(checked.sum()), n_bins=int(a.shape[1]),
        n_events=n_events, slope_threshold=slope_threshold,
        frac_flagged=float(flagged.size) / max(int(checked.sum()), 1),
        median_slope_checked=float(np.median(slope[checked])) if checked.any() else np.nan,
    )


def check_responsive_bins(s_hats: np.ndarray, s_trues: np.ndarray,
                          detectable: Optional[np.ndarray] = None,
                          slope_threshold: float = RESPONSIVENESS_SLOPE_MIN,
                          min_events: int = RESPONSIVENESS_MIN_EVENTS,
                          allow: bool = False) -> dict:
    """responsiveness_report, raised as an error unless `allow` is set.

    Loud by default, for the reason the frozen-bin guard is: a bin that does not
    track the truth where the truth is recoverable is not a noisy measurement,
    it is the absence of one, and CAL1 shows what happens downstream when such a
    bin reaches Stage 2 carrying a small replicate sigma -- it gets the largest
    weight in the likelihood and rails a parameter against its prior.
    """
    rep = responsiveness_report(s_hats, s_trues, detectable=detectable,
                                slope_threshold=slope_threshold,
                                min_events=min_events)
    if rep["flagged_idx"].size and not allow:
        sl = rep["slope"].reshape(-1)[rep["flagged_idx"]]
        raise UnresponsiveBinError(
            "the signal head does not track the truth in %d of %d bins where "
            "the signal IS detectable (slope < %g; flagged slopes run %.4f to "
            "%.4f, against a median of %.3f over the checked bins, on %d "
            "events). This is the CAL2 failure mode, and frozen_bin_report "
            "cannot see it: those bins still respond to the noise draw, so "
            "their replicate spread is normal. Retrain or reseed; pass "
            "allow=True only if you have decided these bins should be masked "
            "out of the likelihood."
            % (rep["flagged_idx"].size, rep["n_checked"], slope_threshold,
               float(np.nanmin(sl)), float(np.nanmax(sl)),
               rep["median_slope_checked"], rep["n_events"]))
    return rep
