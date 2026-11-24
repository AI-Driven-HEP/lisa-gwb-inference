"""
Physics-informed loss functions (paper Section III.B, Eqs. 12-17).

    L_total = lambda_s * L_spectral + lambda_l * L_latent + lambda_c * L_consist

L_spectral : noise + signal reconstruction MSE, with a category-dependent
             weight on the signal term, plus a second-derivative smoothness
             penalty on the predicted signal spectrum.
L_latent   : consistency between Encoder_noise(n_hat) and Encoder_noise(n_true).
L_consist  : consistency between Encoder_total(x_hat) and Encoder_total(x),
             where x_hat is the recombined (noise+signal) prediction.

NOTE on Eq. 13: as printed, the smoothness term ||grad^2 s_hat_i||^2 sits
outside the batch-average sum, which would make it a single scalar computed
from whichever `i` a naive reading binds it to. That's almost certainly a
typesetting artifact of turning a per-sample sum into a paper equation.
Here it is applied per-sample and averaged over the batch along with the
other two terms, which is the only reading that is well-posed and consistent
with "a spectral smoothness penalty" being part of the *reconstruction* loss.
"""

from __future__ import annotations
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from lisa_gwb.simulation import (denormalize_log_psd, normalize_log_psd,
                                 SIGNAL_FLOOR_LOG_PSD)


# --------------------------------------------------------------------------- #
# D3: the floor below which "no signal" is asserted
# --------------------------------------------------------------------------- #
# Physical log10(PSD). For a noise-only sample the objective is a DETECTION
# statement -- do not claim signal -- not a regression target, so the loss
# penalises s_hat only where it rises above this level and is indifferent
# below it.
#
# Choosing the number needed two measurements, and both contradict the obvious
# reading of "one decade below the weakest injected signal in the prior":
#
# 1. Under the default thermodynamic prior the weakest injected signal is NOT A
#    FIXED NUMBER. h2Omega_b ~ K_tilde^2 / (beta/H)^2 with log10 K_tilde in
#    (-5, 0) and log10 beta/H in (0, 5), so the faint tail is unbounded by
#    construction: 20000 draws span 20 decades of amplitude, and the minimum
#    per-event peak keeps falling as the sample grows (-55.42 at 100 events,
#    -56.14 at 400). A minimum is therefore not a usable definition. The 5th
#    percentile is stable (-53.12 at 100 events, -53.48 at 400) and is what we
#    use: floor = round(p5) - 1 decade.
#
# 2. Taken literally from the sample minimum instead, the floor would be about
#    -55.7. Under the constants in force when D3 was written that normalized to
#    -1.33, OUTSIDE the documented window, while the percentile definition
#    landed at -1.125, inside it. N1 reset the constants from measurement, so
#    the numbers have moved -- the floor is now normalized -0.900 and the whole
#    target range is [-0.900, +0.861] -- but the ARGUMENT is unchanged and is
#    the one N1 acted on: a target outside the window the affine rescaling was
#    designed for becomes the largest term in the objective.
#
# For contrast, the sentinel this replaced was log10(eps) = -60, normalized
# -1.875 under the pre-N1 constants, five decades below anything physical.
#
# N1 MOVED THE CONSTANT ITSELF to simulation (imported above, value
# unchanged at -54.0) because the generator now floors the signal target at the
# same number the hinge switches on at, and two copies of that could drift. The
# reasoning stays here because this is where it was measured.


def signal_floor_normalized() -> float:
    """SIGNAL_FLOOR_LOG_PSD in the normalized units the network works in."""
    return float(normalize_log_psd(SIGNAL_FLOOR_LOG_PSD))


# Sample categories (paper Section III.B, Eq. 15)
K_NOISE_ONLY = 0
K_NOISE_COSMO = 1
K_NOISE_ASTRO = 2
K_NOISE_COSMO_ASTRO = 3


def detectability_gate(
    n_true: torch.Tensor,
    s_true: torch.Tensor,
    sigma_frac: torch.Tensor,
) -> torch.Tensor:
    """P1. Per-bin weight in [0, 1]: how much of this bin is worth regressing on.

    CAL2 established that above 30 mHz the injected signal is 1e-6 to 1e-9 of
    the total observed PSD against a best-possible per-bin precision of 2.5e-3,
    that the band carries a detection SNR of 0.27, and that an oracle handed the
    exact noise and twenty times the data does no better than a coin flip there.
    The loss was nevertheless asking the network to regress s_hat onto s_true in
    those bins. It cannot be done, so the request is not a hard target -- it is
    an impossible one, and the capacity spent failing at it is spent by a
    decoder shared with the band that can be measured.

    The criterion is CAL2's, unchanged: the injected signal's fraction of the
    total against that bin's own sigma_frac,

        r = [S/(S+N)] / sigma_frac ,      w = r^2 / (1 + r^2)

    so w = 1/2 exactly at CAL2's threshold r = 1, w -> 1 where the signal is
    comfortably detectable (0.999 at the core band's r = 40) and w -> 0 where it
    is not (2.5e-7 at the 5e-4 of the four named high-frequency bins).

    WHY A SMOOTH GATE RATHER THAN A MASK. A hard mask on r > 1 makes the set of
    retained bins jump discontinuously as an event's amplitude crosses the
    threshold, and makes the retained element COUNT event-dependent -- which is
    the quantity D5 went to some trouble to keep the loss balance independent
    of. The gate is within a per cent of unity everywhere CAL2 calls the signal
    detectable and within a part in a million where it does not, so it buys the
    mask's behaviour without the discontinuity, and the denominator it is
    normalised by moves smoothly rather than in steps.

    TRAINING, NOT INFERENCE. CAL2 called its det24/detex bin sets a diagnostic
    rather than a procedure, because they were selected using the injected truth
    and an analyst does not have it. That objection does not apply here and the
    distinction is the whole reason this is legitimate: at INFERENCE the truth
    is the unknown, so selecting bins with it assumes the answer, whereas at
    TRAINING the truth is precisely what is available -- s_true and n_true are
    the supervision signal, already passed to this function for the squared
    error. Using them to decide which bins carry information is the same class
    of act as using them as targets at all. Nothing here is available to, or
    needed by, Stage 2.

    NUMERICS. Every quantity is formed in log space. S/(S+N) = 1/(1 + 10^(n-s))
    and the bins this gate exists for have n - s of 6 to 9, so the linear form
    would compute 10^-44 -- subnormal in float32, and exactly zero at the -54
    signal floor -- before dividing. Writing the gate as a logistic in ln r
    keeps every intermediate O(1):

        ln r = ln10 * (log10 frac - log10 sigma_frac) ,   w = sigmoid(2 ln r)

    which is r^2/(1+r^2) identically, with no underflow anywhere. Same lesson as
    recombine_log_domain (A2): the literal expression is the wrong one.

    n_true, s_true: NORMALIZED (B, n_ch, N) or (B, N).
    sigma_frac: (n_ch, N), (N,) or scalar, from
        simulation.multichannel_sigma_frac -- a PHYSICAL fraction, not
        a normalized quantity, and not something to pass through
        normalize_log_psd.
    Returns a tensor broadcast to s_true's shape, in (0, 1).
    """
    ln10 = math.log(10.0)
    # log10(S/(S+N)) = -log10(1 + 10^(n-s)), stable for any separation.
    delta_ln = (denormalize_log_psd(n_true) - denormalize_log_psd(s_true)) * ln10
    log_frac_ln = -torch.logaddexp(torch.zeros_like(delta_ln), delta_ln)
    if not torch.is_tensor(sigma_frac):
        sigma_frac = torch.as_tensor(sigma_frac, dtype=s_true.dtype,
                                      device=s_true.device)
    sigma_frac = sigma_frac.to(dtype=s_true.dtype, device=s_true.device)
    # +inf sigma_frac (an empty bin) -> ln r = -inf -> w = 0, which is the safe
    # reading: a bin with no fine samples is never detectable.
    ln_r = log_frac_ln - torch.log(sigma_frac)
    return torch.sigmoid(2.0 * ln_r)


def signal_reconstruction_weight(
    categories: torch.Tensor,
    w_base: float = 1.0,
    w_high: float = 5.0,
    k_noise_only: int = K_NOISE_ONLY,
) -> torch.Tensor:
    """
    w(k_i) from Eq. 15: an elevated penalty w_high for noise-only samples
    (discourages the model from hallucinating signal where there is none),
    and w_base otherwise. `categories`: (B,) long tensor of k_i in {0,1,2,3}.
    Returns a (B,) tensor of per-sample weights.
    """
    return torch.where(
        categories == k_noise_only,
        torch.full_like(categories, w_high, dtype=torch.float32),
        torch.full_like(categories, w_base, dtype=torch.float32),
    )


def second_derivative_penalty(s_hat: torch.Tensor) -> torch.Tensor:
    """
    ||grad^2 s_hat_i||_2^2, Eq. 14: discrete second-derivative (curvature)
    along the frequency axis, penalizing jagged / unphysical spectra.

    Per-element mean over the interior points rather than a sum (D5), so that
    smoothness_weight means the same thing at any N.

    F3: the difference is taken along the LAST axis only, which is frequency,
    and every leading axis is averaged over. With s_hat shaped (B, n_ch, N) the
    curvature of each TDI channel is computed within that channel and the two
    are never differenced against each other. Under the old layout, where
    channels were concatenated along frequency, this term saw the step between
    the top of one channel's band and the bottom of the next as a genuine
    spectral feature and penalised it.

    s_hat: (B, N) or (B, n_ch, N) -> returns (B,) per-sample penalty.
    """
    # s[j+2] - 2 s[j+1] + s[j] for j = 1..N-2, along frequency
    d2 = s_hat[..., 2:] - 2 * s_hat[..., 1:-1] + s_hat[..., :-2]
    return (d2 ** 2).flatten(1).mean(dim=1)


def noise_only_hinge(s_hat: torch.Tensor, floor_norm: float) -> torch.Tensor:
    """
    One-sided penalty on claiming signal where there is none (D3).

    Returns mean_j relu(s_hat_j - floor)^2 per sample: zero wherever the
    predicted signal stays below the floor, growing quadratically above it. The
    network is free to put a noise-only prediction anywhere in the basement
    rather than being regressed onto one particular basement value.

    Per-element mean rather than sum, to match the rest of L_spectral (D5).

    s_hat: (B, N) or (B, n_ch, N) normalized -> returns (B,).
    """
    return (torch.relu(s_hat - floor_norm) ** 2).flatten(1).mean(dim=1)


def _weighted_batch_mean(per_sample: torch.Tensor,
                          sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
    """Batch reduction for a per-sample loss, with optional importance weights.

    P5. `sample_weight` is (B,) and is expected to already have mean 1 over the
    batch -- see `MSMHLoss.__call__` for why the normalisation is per batch and
    not per epoch. The weight multiplies the per-sample loss BEFORE the mean,
    rather than the mean being replaced by sum(w*L)/sum(w), so that at uniform
    weights `per_sample * 1.0` is bit-identical to `per_sample` and the
    reduction that follows is the same call on the same tensor. That is what
    makes `test_uniform_weights_reproduce_the_unweighted_loss_exactly` an
    equality rather than a tolerance.

    D5 is preserved: the reduction over (n_ch, N) inside `per_sample` is
    untouched, so every term is still a per-element mean and lambda stays
    transferable between architectures.
    """
    if sample_weight is None:
        return per_sample.mean()
    w = sample_weight.to(per_sample.dtype).reshape(-1)
    if w.shape != per_sample.shape:
        raise ValueError(f"sample_weight has shape {tuple(w.shape)}, expected "
                         f"{tuple(per_sample.shape)} (one weight per sample)")
    return (per_sample * w).mean()


def _weighted_latent_mean(z_hat, z_true, sample_weight) -> torch.Tensor:
    """Per-element mean of the squared latent difference, over all scales, with
    optional per-sample importance weights.

    The weight is broadcast onto the ELEMENTS of each sample before the same
    `.sum()` the unweighted path uses, for the bit-exactness reason in
    `_weighted_batch_mean`. `count` is untouched, so the result is a weighted
    per-element mean whose scale matches the unweighted one whenever the
    weights average to 1.
    """
    total = 0.0
    count = 0
    for l in z_hat:
        diff = z_hat[l] - z_true[l]
        sq = diff ** 2
        if sample_weight is not None:
            w = sample_weight.to(sq.dtype).reshape(-1, *([1] * (sq.dim() - 1)))
            sq = sq * w
        total = total + sq.sum()
        count += diff.numel()
    return total / count


def spectral_loss(
    n_hat: torch.Tensor,
    n_true: torch.Tensor,
    s_hat: torch.Tensor,
    s_true: torch.Tensor,
    categories: torch.Tensor,
    w_base: float = 1.0,
    w_high: float = 5.0,
    smoothness_weight: float = 1.0,
    signal_floor_norm: Optional[float] = None,
    detectability_weight: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L_spectral, Eq. 13. Spectral tensors are (B, n_ch, N) -- or (B, N) for the
    single-channel case -- and `categories` is (B,). Returns a scalar (already
    averaged over the batch). Every reduction is over frequency and channel
    independently; no operation mixes adjacent frequencies of DIFFERENT TDI
    channels (F3).

    detectability_weight (P1): per-bin weight from detectability_gate,
    broadcastable to s_true. None (the default) means no gating and reproduces
    the pre-P1 loss bit for bit, so every result measured before P1 stays
    reproducible from this code. The NOISE term is never gated under any
    setting -- the instrument noise is what the T channel is for and what the
    inputs are dominated by, and it is well measured in exactly the bins where
    the signal is not.

    D5: every term is a per-element MEAN, not a sum over the frequency axis.
    Under the sum convention the three terms of Eq. 12 carry different element
    counts, and the ratio between them is exactly n_layers * base_channels / 3
    -- the latent terms sum sum_l c_l n_l = n_layers * base_channels * N
    elements while this one sums about 3N. That ratio is a property of the
    ARCHITECTURE, so the lambdas required to balance the objective move when
    the architecture does: 5.3 at the tiny config against 64 at the paper
    config, a factor of twelve. A lambda tuned at the tiny config used
    for testing would be wrong by that factor at the config the paper uses.
    Means make the balance architecture-independent.

    `categories` is no longer read by the signal term after N1 -- the per-bin
    floor test decides the branch, and for k=0 every bin is floored, so the
    category is implied by the data rather than asserted alongside it. The
    argument is kept because the weighting API (eq15_weights) still takes it and
    because a caller passing mismatched categories should not silently work.

    D3: for noise-only samples (k=0) the signal term is a one-sided hinge above
    signal_floor_norm rather than a squared error against s_true. Eq. 15's
    w_high=5 encodes "do not hallucinate signal", which is a detection
    statement; the old form implemented it as a regression onto
    log10(eps) = -60, a sentinel five decades below any physical spectrum and
    outside the normalization window. Audit 01b measured that costing a 56x
    swing in L_spectral depending only on how many noise-only samples a batch
    happened to contain, since the largest single term in the objective was the
    network being pulled toward a placeholder. w_high still multiplies the
    hinge, so the strength of the "do not hallucinate" instruction is unchanged.
    """
    # Derived from the PHYSICAL floor through normalize_log_psd at every call,
    # never cached and never written out in normalised units. N1 changed the
    # normalisation constants, and a hardcoded -1.125 would have gone on looking
    # right while meaning a different physical level.
    floor = signal_floor_normalized() if signal_floor_norm is None else signal_floor_norm

    # flatten(1) averages over (n_ch, N) together; nothing is ever differenced
    # or summed ACROSS the channel axis, only reduced over it (F3).
    noise_term = ((n_hat - n_true) ** 2).flatten(1).mean(dim=1)         # (B,)

    # A9/N1: the hinge is applied PER BIN, not per sample. A bin whose target
    # sits at the floor says "no detectable signal at this frequency", which is
    # the same detection statement D3 made for a whole noise-only sample, and it
    # is now made at the resolution the problem actually has. After T2 the T
    # channel reaches the floor in about 16% of its signal-target bins on
    # signal-bearing events, so under the old per-sample rule those bins were
    # being regressed onto the floor value as though it were a measurement.
    #
    # This SUBSUMES the old behaviour rather than sitting alongside it: for a
    # noise-only event PS is identically zero, so every bin is floored, every
    # bin takes the hinge branch, and the per-element mean reduces exactly to
    # w_high * noise_only_hinge(...). test_hinge_reduces_to_the_per_sample_form
    # pins that.
    #
    # The tolerance is in normalised units and is generous by three orders of
    # magnitude: one float32 ulp near the floor is ~1e-7, while 1e-4 normalised
    # is about a millidecade of PSD, far below any real spectral structure.
    at_floor = s_true <= floor + 1e-4
    per_elem = torch.where(
        at_floor,
        w_high * torch.relu(s_hat - floor) ** 2,
        w_base * (s_hat - s_true) ** 2,
    )

    # P1: detectability-aware weighting. The gate multiplies the REGRESSION
    # branch only; floored bins keep weight 1.
    #
    # The hinge is deliberately never gated, and the reason is that the two
    # branches make different kinds of statement. The regression branch asserts
    # a value, and asserting a value the data cannot support is what CAL2 found
    # the loss doing above 30 mHz. The hinge asserts an upper limit -- "do not
    # claim signal here" -- which is not only still meaningful where nothing is
    # detectable, it is the ONLY meaningful statement there. Gating it would
    # switch off the do-not-hallucinate instruction exactly in the bins most
    # exposed to hallucination, and would also let sum(weight) reach zero for a
    # noise-only sample, where every bin is floored.
    #
    # D5 IS PRESERVED BY THE DENOMINATOR, not by the numerator. The term stays a
    # per-element MEAN by dividing through sum(weight) rather than the element
    # count: an unweighted mean over a gated numerator would shrink the signal
    # term by whatever fraction of bins happened to be detectable -- about a
    # half here, and different for every event and every channel -- so the
    # balance between the noise and signal terms, and hence every lambda, would
    # start depending on the element count again. That is precisely what D5
    # exists to prevent. With the weighted mean, an all-detectable batch
    # reproduces the ungated value exactly and lambda transfers unchanged.
    if detectability_weight is None:
        signal_term = per_elem.flatten(1).mean(dim=1)                   # (B,)
    else:
        w = torch.where(at_floor, torch.ones_like(detectability_weight),
                        detectability_weight)
        w = w.expand_as(per_elem).flatten(1)
        num = (w * per_elem.flatten(1)).sum(dim=1)
        # sum(w) >= the floored-bin count >= 0; clamp only guards the
        # pathological all-undetectable, none-floored event.
        signal_term = num / w.sum(dim=1).clamp_min(1e-8)                # (B,)
    smooth_term = smoothness_weight * second_derivative_penalty(s_hat)  # (B,)

    per_sample = noise_term + signal_term + smooth_term
    return _weighted_batch_mean(per_sample, sample_weight)


def latent_loss(z_noise_hat: Dict[int, torch.Tensor], z_noise_true: Dict[int, torch.Tensor],
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    L_latent, Eq. 16: squared Frobenius norm between Encoder_noise(n_hat) and
    Encoder_noise(n_true), across all scales l = 1..L.

    D5: averaged over every latent element rather than summed. A sum makes the
    term scale as n_layers * base_channels * N, so lambda_l would have to be
    retuned for every architecture change; see spectral_loss.

    P5: sample_weight is (B,) importance weights with mean 1 over the batch.
    """
    return _weighted_latent_mean(z_noise_hat, z_noise_true, sample_weight)


def consistency_loss(z_total_hat: Dict[int, torch.Tensor], z_total_true: Dict[int, torch.Tensor],
                     sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    L_consist, Eq. 17: squared Frobenius norm between Encoder_total(x_hat) and
    Encoder_total(x), across all scales l = 1..L.

    D5: averaged over every latent element rather than summed, for the same
    reason as latent_loss.

    P5: sample_weight is (B,) importance weights with mean 1 over the batch.
    """
    return _weighted_latent_mean(z_total_hat, z_total_true, sample_weight)


def recombine_log_domain(n_hat: torch.Tensor, s_hat: torch.Tensor, eps: float = 1e-100) -> torch.Tensor:
    """
    x_hat = log10(10^n_hat + 10^s_hat)  (paper footnote 2).

    UNITS: this works in PHYSICAL log10(PSD), i.e. values around -55 to -39.
    Callers holding normalized spectra must denormalize on the way in and
    re-normalize on the way out -- see MSMHLoss.__call__. Adding two spectra is
    only meaningful in linear power, and 10^(normalized value) is not a power.

    Two things were wrong with the previous version (audit finding A2):

    1. It was called with NORMALIZED inputs, so 10^n_hat was not a PSD at all.
       For a true physical pair (-44, -46) the round trip landed 1.55 decades
       off.
    2. eps defaulted to 1e-10, which is twenty-five orders of magnitude LARGER
       than the physical PSDs here, so it did not regularise the sum, it
       replaced it: recombine_log_domain(-44, -46) returned exactly -10.0.

    The sum is evaluated as max + log10(sum of 10^(x - max)) rather than
    literally. That is not a micro-optimisation: the pipeline runs in float32,
    whose smallest subnormal is ~1.4e-45, so 10^-46 underflows to zero and even
    10^-44 keeps barely one significant digit. Measured, the literal form gives
    -44.0084 in float32 against the true -43.9957, while this form gives
    -43.99568. Shifting by the max keeps every term in [0, 1] and is exact in
    both dtypes.

    eps is now far below any physical PSD (so it cannot dominate anything) and
    survives only as a floor for the degenerate case where both inputs are
    -inf, which would otherwise give NaN.

    n_hat, s_hat: (B, N) physical log10(PSD) -> returns (B, N), same units.
    """
    peak = torch.maximum(n_hat, s_hat)
    total = torch.pow(10.0, n_hat - peak) + torch.pow(10.0, s_hat - peak)
    out = peak + torch.log10(total)
    if eps > 0:
        out = torch.clamp(out, min=math.log10(eps))
    return out


class MSMHLoss:
    """
    Convenience wrapper bundling the three loss terms with the paper's
    default weights (lambda_l, lambda_c, lambda_s) = (0.1, 0.1, 0.5), Table I.
    """

    def __init__(
        self,
        lambda_s: float = 0.5,
        lambda_l: float = 0.1,
        lambda_c: float = 0.1,
        w_base: float = 1.0,
        w_high: float = 5.0,
        smoothness_weight: float = 1.0,
        signal_floor_norm: Optional[float] = None,
        sigma_frac=None,
    ):
        self.signal_floor_norm = signal_floor_norm
        self.lambda_s = lambda_s
        self.lambda_l = lambda_l
        self.lambda_c = lambda_c
        self.w_base = w_base
        self.w_high = w_high
        self.smoothness_weight = smoothness_weight
        # P1. sigma_frac is the per-bin fractional uncertainty of the averaged
        # periodogram, (n_ch, N) from multichannel_sigma_frac. None keeps the
        # pre-P1 behaviour exactly, which is what every measurement before P1
        # was made under.
        if sigma_frac is not None and not torch.is_tensor(sigma_frac):
            sigma_frac = torch.as_tensor(np.asarray(sigma_frac),
                                          dtype=torch.float32)
        self.sigma_frac = sigma_frac

    def __call__(
        self,
        model,
        outputs: Dict,
        n_true: torch.Tensor,
        s_true: torch.Tensor,
        categories: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        """
        model:  the MSMHAutoencoder instance (needed to re-run Encoder_noise /
                Encoder_total for the latent / consistency losses).
        outputs: dict returned by model(x).
        n_true, s_true: (B, N) ground-truth noise / signal log-spectra.
        categories: (B,) long tensor with values in {0,1,2,3} (Eq. 15).

        C4: this used to take the original input `x` as well and never read it.
        Eq. 17 compares Encoder_total(x_hat) against outputs["z_total"], which
        IS Encoder_total(x) already computed in the forward pass, so passing x
        again suggested a dependency that does not exist.

        sample_weight (P5): optional (B,) importance weights, ALREADY normalised
        to mean 1 over the batch. They multiply each sample's contribution to
        all three terms. None reproduces every pre-P5 number bit for bit, and
        uniform weights reproduce the unweighted value exactly rather than to a
        tolerance -- see `_weighted_batch_mean`. The normalisation is per batch
        rather than per epoch; P5 section 2 records why, and the short version is
        that the unit-norm gradient clip (a deliberate choice, measured) would
        otherwise cancel exactly the up-weighting an epoch-normalised weight is
        for.

        Returns (total_loss, dict_of_components) for logging.
        """
        n_hat, s_hat = outputs["n_hat"], outputs["s_hat"]

        # P1. The gate depends only on the TRUTH, never on s_hat, so it carries
        # no gradient and must not create one -- a weight that moved with the
        # prediction would let the network lower its own loss by declaring a bin
        # undetectable.
        det_w = None
        if self.sigma_frac is not None:
            with torch.no_grad():
                det_w = detectability_gate(n_true, s_true, self.sigma_frac)

        l_spec = spectral_loss(
            n_hat, n_true, s_hat, s_true, categories,
            w_base=self.w_base, w_high=self.w_high,
            smoothness_weight=self.smoothness_weight,
            signal_floor_norm=self.signal_floor_norm,
            detectability_weight=det_w,
            sample_weight=sample_weight,
        )

        # z_noise_true is the TARGET for the latent loss, not something we
        # want to optimize through — encode_noise_latents shares weights
        # (Encoder_noise) with the predicted branch (outputs["z_noise"]),
        # so without no_grad() here, gradients would flow through this
        # "true" branch too and pull Encoder_noise's weights toward moving
        # both ends of the same loss term at once, which can destabilize
        # training. Only the predicted branch should receive gradient here.
        with torch.no_grad():
            z_noise_true = model.encode_noise_latents(n_true)
        l_lat = latent_loss(outputs["z_noise"], z_noise_true, sample_weight)

        # A2: the network works in normalized units, but adding a noise and a
        # signal spectrum is only meaningful in linear power. Drop to physical
        # log10(PSD), recombine there, and come back -- the encoder below
        # expects normalized input, since outputs["z_total"] was produced from
        # normalized x.
        x_hat_log = normalize_log_psd(
            recombine_log_domain(denormalize_log_psd(n_hat), denormalize_log_psd(s_hat))
        )   # (B, n_ch, N)
        # x_hat must look like a (B, M*n_ch, N) stack to feed Encoder_total, with
        # the TDI channel varying SLOWEST -- the same layout make_training_batch
        # produces (F3). repeat_interleave along the channel axis gives exactly
        # [ch0 x M, ch1 x M, ...]; each channel's single recombined spectrum is
        # broadcast across its own M time segments. If your training pipeline
        # keeps per-timestamp noise/signal pairs instead of one (n_true, s_true)
        # pair per M-stack, replace this with the actual M-stacked
        # reconstruction.
        if x_hat_log.dim() == 2:                       # single-channel (B, N)
            x_hat_log = x_hat_log.unsqueeze(1)
        x_hat = x_hat_log.repeat_interleave(model.M, dim=1).contiguous()
        z_total_hat = model.encode_total_latents(x_hat)
        l_con = consistency_loss(z_total_hat, outputs["z_total"], sample_weight)

        total = self.lambda_s * l_spec + self.lambda_l * l_lat + self.lambda_c * l_con

        components = {
            "loss_total": total.item() if torch.is_tensor(total) else total,
            "loss_spectral": l_spec.item(),
            "loss_latent": l_lat.item() if torch.is_tensor(l_lat) else l_lat,
            "loss_consist": l_con.item() if torch.is_tensor(l_con) else l_con,
        }
        return total, components
