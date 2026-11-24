"""Simulate stochastic gravitational-wave backgrounds in LISA A, E and T.

The simulator combines instrument noise, Galactic and extragalactic
foregrounds, and a cosmological phase-transition signal. It generates
periodogram observations and their separate noise and signal targets.
Training spectra use channel-dependent coarse-graining weights.

The instrument-noise PSD uses the transfer expressions without the
sky-averaged response factor printed in arXiv:2403.03723, Eqs. (A.4)–(A.5).
See noise_psd for the unit conversion and detector_response for the signal
response.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

C_LIGHT = 299792458.0
MPC_IN_M = 3.085677581e22
#: The sidereal year, 365.25636 d. It is the period of the Galactic-foreground
#: modulation (arXiv:2406.04894 below Eq. 2.2: "t_yr, the sidereal year") and
#: the unit Tobs_yr is converted with, so that one "year" of mission is exactly
#: one modulation cycle.
SIDEREAL_YEAR_S = 365.25636 * 86400.0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class MissionConfig:
    L: float = 2.5e9          # LISA arm length [m]
    Tobs_yr: float = 4.0      # total observation time [yr]; M1: a real parameter now
    tau: float = 1.0e6        # segment duration [s]
    # Number of independent segments available per event. M1: DERIVED from
    # Tobs_yr and tau as floor(Tobs / tau) when left at None -- 126 at 4 yr,
    # 31 at 1 yr. An explicit value is accepted only as a smaller budget (a
    # test that wants a tight limit); asking for more segments than the mission
    # contains raises.
    Nd: Optional[int] = None
    f_min: float = 3e-5       # Hz
    f_max: float = 0.5        # Hz
    h: float = 0.674          # reduced Hubble constant

    # Instrument noise fiducial parameters (TM = test mass, OMS = optical metrology system)
    A_param: float = 3.0
    P_param: float = 15.0
    # Fractional 1-sigma jitter applied per event, to avoid the network
    # overfitting to one exact noise budget. Set to 0.0 to disable.
    noise_jitter_frac: float = 0.05

    # Effective relativistic degrees of freedom at the PT, used only for the
    # thermodynamic -> geometric PT parameter conversion (Sec. 2, Eqs. 2.2-2.3
    # of Caprini et al. 2403.03723). Fixed at the SM value, matching the
    # paper's own Sec. 5 benchmarks (their footnote 7: "we fix g*=106.75").
    g_star: float = 106.75

    def __post_init__(self):
        if self.Tobs_yr <= 0:
            raise ValueError(f"Tobs_yr must be positive, got {self.Tobs_yr}")
        available = int(np.floor(self.Tobs_s / self.tau + 1e-9))
        if available < 1:
            raise ValueError(f"Tobs_yr = {self.Tobs_yr} holds no complete segment "
                             f"of tau = {self.tau} s")
        if self.Nd is None:
            self.Nd = available
        elif not 1 <= self.Nd <= available:
            raise ValueError(f"Nd = {self.Nd} is outside [1, {available}], the "
                             f"number of tau = {self.tau:g} s segments in "
                             f"Tobs_yr = {self.Tobs_yr}")

    @property
    def Tobs_s(self) -> float:
        return self.Tobs_yr * SIDEREAL_YEAR_S

    @property
    def df(self) -> float:
        return 1.0 / self.tau

    @property
    def H0(self) -> float:
        return self.h * 100.0 * 1000.0 / MPC_IN_M

    @property
    def pref(self) -> float:
        return 3 * self.H0 ** 2 / (4 * np.pi ** 2)


# --------------------------------------------------------------------------- #
# Thermodynamic -> geometric PT parameter conversion
# (Caprini et al., "Gravitational waves from first-order phase transitions in
#  LISA: reconstruction pipeline and physics interpretation", arXiv:2403.03723,
#  Sec. 2.1 and Sec. 5 "Bubble collisions and highly relativistic fluid shells")
# --------------------------------------------------------------------------- #

def hubble_star_0(T_star_GeV: float, g_star: float) -> float:
    """H_{*,0} (Eq. 2.2): the Hubble rate at GW production, redshifted to
    today's frequency. Returns Hz."""
    return 1.65e-5 * (g_star / 100.0) ** (1.0 / 6.0) * (T_star_GeV / 100.0)


def redshift_h2FGW0(g_star: float) -> float:
    """h^2 F_{GW,0} (Eq. 2.3): the redshift factor for the fractional GW
    energy density."""
    return 1.64e-5 * (100.0 / g_star) ** (1.0 / 3.0)


def bpl_thermo_to_geometric(K_tilde: float, beta_over_H: float, T_star_GeV: float,
                             g_star: float = 106.75, A_str: float = 0.05):
    """
    Converts the thermodynamic PT parameters (K_tilde, beta/H*, T*) to the
    BPL template's free geometric parameters (h2Omega_b, fb), via Eq. 2.7.
    Valid for very strong transitions (alpha >> 1, K_tilde ~ alpha/(1+alpha)
    ~ 1) sourced by bubble collisions / highly relativistic fluid shells —
    the same regime Table 1's "strong PT" row (n1=-n2=2.4, a1=1.2) assumes,
    which is why f_b=f_p and Omega_b=Omega_p here (only true when n1=-n2).

    Verified against the paper's own Table 4 benchmark (K_tilde=1,
    beta/H*=500, T*=200 GeV): this reproduces log10(h2Omega_b) = -11.49,
    matching this file's long-standing fiducial of -11.5 almost exactly.

    A_str ~ 0.05 is the simulation-calibrated normalization from Ref. [32]
    of the paper (bubble-collision GW efficiency), not something to tune.
    """
    H_star_0 = hubble_star_0(T_star_GeV, g_star)
    h2_F_GW0 = redshift_h2FGW0(g_star)
    fb = 0.11 * H_star_0 * beta_over_H
    h2Omega_b = h2_F_GW0 * A_str * K_tilde ** 2 * (1.0 / beta_over_H) ** 2
    return h2Omega_b, fb


@dataclass
class SignalPriors:
    """Prior ranges for each randomizable signal component. Widen/narrow
    these to match whatever parameter space you actually care about.

    The cosmological PT signal can be parameterized TWO ways, selected by
    `pt_mode`:

    - "thermodynamic" (default): draws (K_tilde, beta/H*, T*) — Table 4's
      exact prior ranges (left column) from arXiv:2403.03723, the paper's
      Fig. 8 benchmark. Converted to (h2Omega_b, fb) via
      bpl_thermo_to_geometric.
    - "geometric": draws (log10_h2Omega_b, fb) directly — the paper's Fig. 3
      setup, and what an earlier version of this file (before the
      thermodynamic parameterization was added) used throughout.

    Either way, n1/n2/a1 are FIXED at Table 1's "strong PT" row (2.4, -2.4,
    1.2), not randomized — the paper fixes these by the source mechanism
    itself; only the amplitude and break frequency are free. This is a
    correction over an earlier version of this file, which incorrectly
    treated n1/n2/alpha as randomizable.

    IMPORTANT — a real issue found by testing, not a hypothetical concern:
    Table 4's literal prior range is built for a Bayesian RECONSTRUCTION
    context (deliberately wide, to explore parameter degeneracies given an
    assumed detection), not for generating ML training data. Sampled
    directly, only ~50% of draws land fb inside LISA's actual band
    (verified: 1005/2000 in a test run) — the other half produce a PT
    signal that's essentially invisible or irrelevant within the simulated
    frequency range, wasted as training examples. sample_event_params
    handles this via rejection sampling (redraw until fb lands in-band,
    capped at max_tries) rather than silently training on a half-useless
    population; see its docstring.

    On Table 3 (the paper's literal geometric BPL prior range): the HTML
    table extraction for this one came back structurally garbled — column
    labels and ranges didn't align cleanly, unlike Table 4 which extracted
    (and independently verified against the paper's own benchmark numbers)
    cleanly. My best inference from the surrounding DBPL column's pattern
    is roughly log10(h2Omega_b): (-30,-5), log10(fb/Hz): (-7,1) — but I
    can't verify this the way I verified Table 4, so I'm not shipping it as
    the default. It would have the same in-band problem as Table 4's literal
    range regardless (an 8-decade amplitude range and fb spanning nHz to
    10 Hz). The geometric default below instead reuses the same
    (well-tested, in-band, centered on the Fig. 3 fiducial) range this file
    used before the thermodynamic parameterization existed. If you want the
    literal Table 3 range for a reconstruction-style exercise rather than
    training-data generation, ask and I'll get a verified reading of it
    before shipping it as an option — I'd rather confirm than guess here.
    """

    pt_mode: str = "thermodynamic"  # "thermodynamic" or "geometric"

    # Thermodynamic PT parameters (Table 4, left column) -- used if pt_mode="thermodynamic"
    log10_K_tilde: Tuple[float, float] = (-5.0, 0.0)
    log10_beta_over_H: Tuple[float, float] = (0.0, 5.0)
    log10_T_star_GeV: Tuple[float, float] = (0.0, 8.0)
    pt_inband_max_tries: int = 50  # rejection-sampling cap; see sample_event_params

    # Geometric PT parameters (Fig. 3 setup) -- used if pt_mode="geometric"
    log10_h2Omega_b_geom: Tuple[float, float] = (-12.5, -10.5)
    fb_geom: Tuple[float, float] = (3e-4, 3e-3)

    # Fixed by source mechanism (Table 1, "strong PT" row) -- not randomized,
    # used by both pt_mode settings
    n1: float = 2.4
    n2: float = -2.4
    alpha: float = 1.2

    # Extragalactic foreground
    log10_h2Omega_ext: Tuple[float, float] = (-13.0, -11.5)

    # Galactic foreground
    log10_h2Omega_gal: Tuple[float, float] = (-8.5, -7.0)
    a1: float = -0.15
    b1: float = -2.72
    ak: float = -0.37
    bk: float = -2.49
    nu: float = 1.56
    f2: float = 6.7e-4

    # M1: annual modulation of the Galactic foreground, arXiv:2406.04894 Eq. 2.3,
    #     Omega_Gal(f, t) = Omega_Gal(f) [1 + A1 cos(w1 t + phi1) + A2 cos(w2 t + phi2)]
    # with w_n = 2 pi n / sidereal year. OFF by default so that every stream and
    # number measured before M1 is reproduced exactly; turning it on is a
    # decision for the Stage-C pool, not a side effect of adding it.
    gal_modulation: bool = False
    # Depth prior. Centred on the reference's fiducial values, A1 = 0.1 and
    # A2 = 0.5 (its Table 1 starting values, which its text says are the
    # injected ones; the second harmonic dominates because the response beam
    # crosses the Galactic plane twice a year). The reference's own MCMC prior,
    # U[0, 1] on each, cannot be used generatively: A1 + A2 > 1 on half its
    # volume, which makes the Galactic PSD negative for part of the year. These
    # widths (+-100% on A1, +-40% on A2) are our choice and keep
    # 1 - A1 - A2 >= 0.1 everywhere.
    gal_mod_A1: Tuple[float, float] = (0.0, 0.2)
    gal_mod_A2: Tuple[float, float] = (0.3, 0.7)
    # Phases: U[0, 2 pi], the reference's prior range.
    gal_mod_phase: Tuple[float, float] = (0.0, 2 * np.pi)
    # Mission start time t_0 relative to the modulation's phase origin, drawn
    # per event. Statistically redundant with uniform phases (it shifts phi_n by
    # w_n t_0), but it is what gives each plane a physical timestamp, and it
    # stops being redundant the moment the phases are fixed by an orbit model.
    gal_mod_t0_yr: Tuple[float, float] = (0.0, 1.0)


CATEGORY_NOISE_ONLY = 0
CATEGORY_COSMO = 1
CATEGORY_ASTRO = 2
CATEGORY_COSMO_ASTRO = 3
CATEGORY_NAMES = {0: "noise_only", 1: "cosmo", 2: "astro", 3: "cosmo_astro"}


# --------------------------------------------------------------------------- #
# Network I/O normalization
# --------------------------------------------------------------------------- #
# Raw log10(PSD) values for this LISA setup sit around -54 to -36 (real PSDs
# at these amplitudes are astronomically small). A freshly-initialized
# network outputs values near 0, so feeding raw log10(PSD) directly as the
# regression target starts training with an enormous, badly-scaled initial
# residual (~45 per bin, squared and summed over N bins -> loss in the
# hundreds of thousands from step 1) — consistent with a run that grows
# rather than converges under aggressive gradient clipping. This affine
# rescaling keeps the network operating in an O(1) regime; it's inverted
# when the separator predictions are converted back to physical log10 PSDs.
#
# N1 SET THE SCALE FROM MEASUREMENT. The old values were (-45.0, 8.0), chosen
# against a documented range of -55 to -39 that three corrections had made
# obsolete: T2's noise correction raised the top of the band, T4's reweighting
# of the T channel raised the bottom by nine decades (the old weight collapsed
# one coarse bin onto a response null, dragging the T noise target to -58.5),
# and A9 put a hard floor under the signal target at -54. Measured over 256
# events at N = 200 across all four categories and all three channels, the
# targets and inputs together span [-54.000, -36.391].
#
# centre -45.0, scale 10.0 maps that to [-0.900, +0.861]: inside [-1, +1] with
# margin at both ends, which is the property that matters. D3 established why
# -- a target that sits outside the window the rescaling was designed for
# becomes the single largest term in the objective, which is the whole reason
# the hinge exists, and the old constants put the floor at -1.125 with the top
# at +1.076, outside it at both ends.
#
# The centre is deliberately left at -45.0 even though the measured midpoint is
# -45.196: only one constant then changes, which is easier to see in a diff and
# to reason about, and 0.2 decades of asymmetry is nothing against 0.9 of
# half-range. A scale of exactly 10 also makes one normalised unit one DECADE
# of PSD, so a residual reads directly: 0.05 normalised is half a decade.
#
# T2 proposed (-47.5, 11.5) from the range as it stood then and declined to
# apply it. That proposal is superseded rather than adopted: against the range
# measured now it would give [-0.565, +0.966], lopsided and wasting the lower
# half of the window.
LOG_PSD_CENTER = -45.0
LOG_PSD_SCALE = 10.0


def normalize_log_psd(log_psd):
    return (log_psd - LOG_PSD_CENTER) / LOG_PSD_SCALE


def denormalize_log_psd(normalized):
    return normalized * LOG_PSD_SCALE + LOG_PSD_CENTER


# --------------------------------------------------------------------------- #
# The signal floor (D3, A9, N1)
# --------------------------------------------------------------------------- #
# Physical log10(PSD) below which the right statement about the SIGNAL is a
# detection statement ("no detectable signal here") rather than a number. It
# lives here rather than in losses.py because the generator and the loss must
# use one constant: the generator floors the signal target at it and the loss
# switches from squared error to a one-sided hinge at it, and if those two ever
# disagreed the hinge would fire on bins that carry a real target.
#
# The value is D3's, unchanged by N1 -- one decade below the 5th percentile of
# the prior's per-event peak signal. See losses.py for why a percentile and not
# a minimum: the thermodynamic prior's faint tail is unbounded, so the sample
# minimum keeps falling as the sample grows while the 5th percentile does not.
SIGNAL_FLOOR_LOG_PSD = -54.0
SIGNAL_FLOOR_PSD = 10.0 ** SIGNAL_FLOOR_LOG_PSD


# --------------------------------------------------------------------------- #
# Normalisation stamp
# --------------------------------------------------------------------------- #
def normalisation_stamp() -> str:
    """Identifies the target representation an artifact was produced under.

    Each saved checkpoint or dataset uses the
    constants that were in force when it was written: a network trained under
    one (centre, scale) emits numbers that mean something else under another,
    and nothing in the array itself reveals the difference. N1 changed both
    constants, so every artifact from before it is silently wrong rather than
    loudly wrong unless something checks. This is the frozen-bin principle --
    when the pipeline cannot produce a trustworthy number it must produce no
    number -- applied to the representation instead of to the network.

    The floor is included because it is part of the same contract: a target
    floored at -60 and one floored at -54 are different functions.
    """
    return (f"logpsd:c={LOG_PSD_CENTER:g},s={LOG_PSD_SCALE:g},"
            f"floor={SIGNAL_FLOOR_LOG_PSD:g}")


class NormalisationMismatch(RuntimeError):
    """An artifact uses a different PSD representation."""


def check_normalisation_stamp(data, path: str = "<arrays>") -> None:
    expected = normalisation_stamp()
    stamp = str(np.asarray(data["norm_stamp"]).item()) if "norm_stamp" in data else None
    if stamp != expected:
        raise NormalisationMismatch(
            f"{path} has normalisation {stamp!r}; expected {expected!r}. "
            "Regenerate this artifact with the current code."
        )


# --------------------------------------------------------------------------- #
# Frequency grid
# --------------------------------------------------------------------------- #

def build_frequency_grid(cfg: MissionConfig) -> np.ndarray:
    """Robust replacement for np.arange(f_min, f_max, df): computes an exact
    bin count first, so there's no floating-point drift at the endpoint."""
    n_bins = int(round((cfg.f_max - cfg.f_min) / cfg.df)) + 1
    return cfg.f_min + cfg.df * np.arange(n_bins)


# --------------------------------------------------------------------------- #
# Detector response and noise. Unchanged from the original script EXCEPT for
# T2, which removed the sky-averaged response factor from the two noise PSDs
# (see noise_psd's docstring); the response functions themselves are untouched.
# --------------------------------------------------------------------------- #

def detector_response(f: np.ndarray, cfg: MissionConfig):
    """Sky- and polarisation-averaged GW response of the A/E and T channels.

    T3 REPORTED TWO COEFFICIENT DISCREPANCIES HERE AND DID NOT FIX THEM, since
    fixing a fit coefficient blind is worse than recording it. Against
    arXiv:2511.00996 Eq. (4.6), which is R_A = (9/5)|W|^2[1 + (3x/4)^2]^-1 and
    R_T = (1/1008)|W|^2 x^6 [1 + (5/16128) x^8]^-1 (|W|^2 = 4 sin^2 x):

      - the A/E denominator is 1 + 0.7 x^2 here against 1 + 0.5625 x^2 there,
        and 1 + 0.6 x^2 in Robson, Cornish & Liu 2019. Leading coefficients
        agree exactly (9/5); the three fits differ by up to 24% asymptotically,
        worth 8.8% in Omega_sens at x = 1 and 23% at 0.1 Hz.
      - R_T/|W|^2 is (1/1000) x^8/(1 + 3.8889e-4 x^8) here against
        (1/1008) x^6/(1 + 3.1002e-4 x^8) there. The leading coefficients agree
        to 0.8%; the x^8 denominators differ by 25%.

    (The overall x^2 by which ours exceeds Eq. (4.6) is the fractional-frequency
    convention and is NOT a discrepancy -- noise_psd carries the same factor and
    it cancels in Omega_sens. See that docstring.)
    """
    x = 2 * np.pi * f * cfg.L / C_LIGHT
    Rtilde_A = (9 / 20) / (1 + 0.7 * x ** 2)
    Rtilde_T = (9 / 20) * x ** 6 / (1800 + 0.7 * x ** 8)
    R_A = 16 * np.sin(x) ** 2 * x ** 2 * Rtilde_A
    R_T = 16 * np.sin(x) ** 2 * x ** 2 * Rtilde_T
    R_E = R_A  # equal-arm approximation: A and E share the same response shape
    return x, R_A, R_E, R_T, Rtilde_A, Rtilde_T


def noise_psd(f: np.ndarray, x: np.ndarray, cfg: MissionConfig, rng: Optional[np.random.Generator] = None):
    """Instrument noise PSD for A/E/T. If `rng` is given and
    cfg.noise_jitter_frac > 0, the A/P noise-budget parameters are jittered
    per call (per event) rather than held perfectly fixed.

    SOURCE, corrected in T3. This is arXiv:2511.00996 Eqs. (4.7)-(4.10), which
    is where this repository's instrument model came from -- its band, tau and
    N0 match ours exactly, and its LISA budget is our A = 3 and P = 15. Written
    out, its Eqs. (4.9)-(4.10) are

        N_A = N_E = (8 sin^2 x / L^2)[(2 + cos x) S_OMS
                            + 4(1 + cos x + cos^2 x) S_acc/(2 pi f)^4]
        N_T       = (16 sin^2 x / L^2)[(1 - cos x) S_OMS
                            + 2(1 - cos x)^2 S_acc/(2 pi f)^4]

    and the four bracket coefficients below are exactly those. There is NO
    sky-averaged response factor in the published noise.

    T2 removed one from ours. It had come in from the OTHER source this
    simulator was assembled from: Caprini et al. arXiv:2403.03723, whose
    Eqs. (A.4)-(A.5) print ~R_AA / ~R_TT in front of the same brackets and
    supplied the broken-power-law signal template in Table 1. T2 reached the
    right answer by external measurement rather than from the source, which it
    did not have; T3 confirms it directly from the source.

    NORMALISATION. Ours equals the published expressions times x^2 =
    (2 pi f L/c)^2, in both terms of both channels, because we work in
    fractional frequency where the paper works in fractional optical path
    length -- STM and SOMS below fold in 1/(2 pi f c)^2 and (2 pi f/c)^2 where
    the paper carries 1/((2 pi f)^4 L^2) and 1/L^2. detector_response carries
    the same x^2, so it cancels in R/PN and Omega_sens is unaffected; the paper
    licenses exactly this below its Eq. (4.6). So the published 1/L^2 prefactor
    is already accounted for and must NOT be applied again -- doing so would put
    every PSD 18.8 decades low.
    """
    A_param, P_param = cfg.A_param, cfg.P_param
    if rng is not None and cfg.noise_jitter_frac > 0:
        A_param *= 1.0 + cfg.noise_jitter_frac * rng.standard_normal()
        P_param *= 1.0 + cfg.noise_jitter_frac * rng.standard_normal()

    STM = (
        (A_param * 1e-15) ** 2
        * (1 + (0.4e-3 / f) ** 2)
        * (1 + (f / 8e-3) ** 4)
        * (1 / (2 * np.pi * f * C_LIGHT)) ** 2
    )
    SOMS = (
        (P_param * 1e-12) ** 2
        * (1 + (2e-3 / f) ** 4)
        * (2 * np.pi * f / C_LIGHT) ** 2
    )

    PN_A = 8 * np.sin(x) ** 2 * (
        4 * (1 + np.cos(x) + np.cos(x) ** 2) * STM + (2 + np.cos(x)) * SOMS
    )
    PN_E = PN_A.copy()
    # The (1 - cos x) factors are the T null stream: they are what makes T a
    # (quasi-)null channel, and they are NOT touched by T2. They vanish at
    # x = 2*n*pi together with sin^2(x), so PN_T has an order-4 zero against
    # R_T's order-2 at f = 0.11992 Hz and its harmonics, and the Fisher weight
    # diverges there. This noise model uses the idealised equal-arm null.
    # See null_channel_valid for cuts based on an arm-length mismatch.
    PN_T = 16 * np.sin(x) ** 2 * (
        2 * (1 - np.cos(x)) ** 2 * STM + (1 - np.cos(x)) * SOMS
    )
    return PN_A, PN_E, PN_T


# --------------------------------------------------------------------------- #
# Signal components
# --------------------------------------------------------------------------- #

def omega_pt(f: np.ndarray, h2Omega_b: float, fb: float, n1: float, n2: float, alpha: float, h: float) -> np.ndarray:
    Omega_b = h2Omega_b / h ** 2
    xf = f / fb
    return Omega_b * xf ** n1 * (0.5 + 0.5 * xf ** alpha) ** ((n2 - n1) / alpha)


def omega_ext(f: np.ndarray, h2Omega_ext: float, h: float) -> np.ndarray:
    Omega_ext = h2Omega_ext / h ** 2
    return Omega_ext * (f / 1e-3) ** (2 / 3)


def omega_gal(f: np.ndarray, h2Omega_gal: float, priors: SignalPriors, cfg: MissionConfig) -> np.ndarray:
    Omega_Gal = h2Omega_gal / cfg.h ** 2
    f1 = 10 ** (priors.a1 * np.log10(cfg.Tobs_yr) + priors.b1)
    fknee = 10 ** (priors.ak * np.log10(cfg.Tobs_yr) + priors.bk)
    return (
        0.5 * f ** (2 / 3)
        * (1 + np.tanh((fknee - f) / priors.f2))
        * np.exp(-(f / f1) ** priors.nu)
        * Omega_Gal
    )


def sample_event_params(rng: np.random.Generator, priors: SignalPriors, category: int,
                         cfg: MissionConfig, amplitude_scale: float = 1.0) -> dict:
    """
    Draw randomized signal parameters for one event, zeroing out whichever
    components are absent for `category` (Eq. 15-style k in {0,1,2,3}).
    `amplitude_scale` multiplies the *linear* Omega of every active
    component — this is the hook for the MSMHAutoencoder curriculum
    scheduler (curriculum.py): pass its sampled A here each epoch/batch.

    The cosmological PT component uses `priors.pt_mode`:
    - "thermodynamic": draws (K_tilde, beta/H*, T*) and converts to
      (h2Omega_b, fb) via bpl_thermo_to_geometric. Table 4's literal prior
      range puts fb outside LISA's actual band about half the time (verified
      by testing, not assumed) — such a draw teaches the network nothing
      useful, so this rejection-samples: redraw fresh (K_tilde, beta/H*, T*)
      until fb lands in [cfg.f_min, cfg.f_max], up to
      priors.pt_inband_max_tries attempts (falls through with whatever the
      last draw was if that cap is hit, rather than looping forever — with
      the default range and cap this is only reached in pathological cases).
    - "geometric": draws (log10_h2Omega_b, fb) directly from
      priors.log10_h2Omega_b_geom / priors.fb_geom — no rejection sampling
      needed since that range is already centered in-band by construction.
    n1/n2/alpha are always fixed (Table 1's "strong PT" row), not drawn.
    """
    params = dict(
        h2Omega_b=0.0, fb=1e-3, n1=priors.n1, n2=priors.n2, alpha=priors.alpha,
        h2Omega_ext=0.0, h2Omega_gal=0.0,
        K_tilde=None, beta_over_H=None, T_star_GeV=None,  # filled in below if thermodynamic + cosmo
    )

    if category in (CATEGORY_COSMO, CATEGORY_COSMO_ASTRO):
        if priors.pt_mode == "thermodynamic":
            for _ in range(priors.pt_inband_max_tries):
                K_tilde = 10 ** rng.uniform(*priors.log10_K_tilde)
                beta_over_H = 10 ** rng.uniform(*priors.log10_beta_over_H)
                T_star_GeV = 10 ** rng.uniform(*priors.log10_T_star_GeV)
                h2Omega_b, fb = bpl_thermo_to_geometric(K_tilde, beta_over_H, T_star_GeV, cfg.g_star)
                if cfg.f_min <= fb <= cfg.f_max:
                    break  # in-band draw found, stop retrying

            params["h2Omega_b"] = amplitude_scale * h2Omega_b
            params["fb"] = fb
            params["K_tilde"] = K_tilde
            params["beta_over_H"] = beta_over_H
            params["T_star_GeV"] = T_star_GeV

        elif priors.pt_mode == "geometric":
            params["h2Omega_b"] = amplitude_scale * 10 ** rng.uniform(*priors.log10_h2Omega_b_geom)
            params["fb"] = rng.uniform(*priors.fb_geom)

        else:
            raise ValueError(f"priors.pt_mode must be 'thermodynamic' or 'geometric', got {priors.pt_mode!r}")

    if category in (CATEGORY_ASTRO, CATEGORY_COSMO_ASTRO):
        params["h2Omega_ext"] = amplitude_scale * 10 ** rng.uniform(*priors.log10_h2Omega_ext)
        params["h2Omega_gal"] = amplitude_scale * 10 ** rng.uniform(*priors.log10_h2Omega_gal)

    if priors.gal_modulation:
        # Drawn for every event, Galaxy present or not, so the stream consumed
        # per event does not depend on the category. Only the Galactic
        # component ever reads these.
        params.update(sample_gal_modulation(rng, priors))

    return params


def sample_gal_modulation(rng: np.random.Generator, priors: SignalPriors) -> dict:
    """One draw of (A1, A2, phi1, phi2, t0) from the M1 modulation prior."""
    return dict(
        mod_A1=rng.uniform(*priors.gal_mod_A1),
        mod_A2=rng.uniform(*priors.gal_mod_A2),
        mod_phi1=rng.uniform(*priors.gal_mod_phase),
        mod_phi2=rng.uniform(*priors.gal_mod_phase),
        t0=rng.uniform(*priors.gal_mod_t0_yr) * SIDEREAL_YEAR_S,
    )


def segment_times(M: int, n_avg: int, cfg: MissionConfig, t0: float = 0.0) -> np.ndarray:
    """Start times [s] of the M input planes: t_i = t0 + i * n_avg * tau.

    Plane i averages the n_avg consecutive segments starting at t_i, so it
    covers [t_i, t_i + n_avg * tau).
    """
    return t0 + np.arange(M) * n_avg * cfg.tau


def gal_modulation_factor(t_start, t_end, params: dict) -> np.ndarray:
    """Time average of m(t) = 1 + A1 cos(w1 t + phi1) + A2 cos(w2 t + phi2)
    over each window [t_start, t_end), w_n = 2 pi n / sidereal year.

    The average, not the value at the window start: a plane's expected
    periodogram is the mean PSD over the time it spans. For one tau = 1e6 s
    segment this differs from the midpoint value by under 1% (the semi-annual
    harmonic loses a factor sinc(w2 tau / 2) = 0.987), but for a plane that
    averages many segments it is the whole effect -- a plane spanning a full
    year sees m = 1 exactly.

    Events without modulation keys (e.g. fixed_params built before M1) get
    m = 1 identically.
    """
    t_start = np.asarray(t_start, dtype=float)
    t_end = np.asarray(t_end, dtype=float)
    m = np.ones(np.broadcast(t_start, t_end).shape)
    if "mod_A1" not in params:
        return m
    dt = t_end - t_start
    for n, A, phi in ((1, params["mod_A1"], params["mod_phi1"]),
                      (2, params["mod_A2"], params["mod_phi2"])):
        w = 2 * np.pi * n / SIDEREAL_YEAR_S
        m = m + A * (np.sin(w * t_end + phi) - np.sin(w * t_start + phi)) / (w * dt)
    return m


def build_pt_event_params(K_tilde: float, beta_over_H: float, T_star_GeV: float,
                           h2Omega_ext: float, h2Omega_gal: float,
                           cfg: MissionConfig, priors: SignalPriors) -> dict:
    """
    Build a fixed event for make_training_batch from thermodynamic
    phase-transition parameters and foreground amplitudes. For example:
        build_pt_event_params(K_tilde=1.0, beta_over_H=500.0, T_star_GeV=200.0,
                               h2Omega_ext=10**-12.38, h2Omega_gal=10**-7.84,
                               cfg=cfg, priors=priors)
    """
    h2Omega_b, fb = bpl_thermo_to_geometric(K_tilde, beta_over_H, T_star_GeV, cfg.g_star)
    return dict(
        h2Omega_b=h2Omega_b, fb=fb, n1=priors.n1, n2=priors.n2, alpha=priors.alpha,
        h2Omega_ext=h2Omega_ext, h2Omega_gal=h2Omega_gal,
        K_tilde=K_tilde, beta_over_H=beta_over_H, T_star_GeV=T_star_GeV,
    )


def omega_split(f: np.ndarray, params: dict, priors: SignalPriors, cfg: MissionConfig):
    """(Omega_stationary, Omega_Gal): the time-independent part (phase
    transition + extragalactic) and the unmodulated Galactic part.

    M1: the modulation multiplies the second only, so the generator needs them
    apart. total_omega is their sum, i.e. the Galaxy at m = 1.
    """
    om = np.zeros_like(f)
    if params["h2Omega_b"] > 0:
        om = om + omega_pt(f, params["h2Omega_b"], params["fb"], params["n1"], params["n2"], params["alpha"], cfg.h)
    if params["h2Omega_ext"] > 0:
        om = om + omega_ext(f, params["h2Omega_ext"], cfg.h)
    gal = np.zeros_like(f)
    if params["h2Omega_gal"] > 0:
        gal = omega_gal(f, params["h2Omega_gal"], priors, cfg)
    return om, gal


def total_omega(f: np.ndarray, params: dict, priors: SignalPriors, cfg: MissionConfig) -> np.ndarray:
    om, gal = omega_split(f, params, priors, cfg)
    return om + gal


# --------------------------------------------------------------------------- #
# Periodogram simulation
# --------------------------------------------------------------------------- #

def segment_power_sums(P_mean: np.ndarray, n_avg: int, P_mod: Optional[np.ndarray] = None,
                       m_row: Optional[np.ndarray] = None):
    """The two sums over the n_avg segments of ONE plane that determine the
    averaged periodogram's first two moments:

        S1 = sum_j P_j ,   S2 = sum_j P_j^2 ,   P_j = P_mean + m_j P_mod

    The averaged periodogram is (1/n_avg) sum_j Exp(P_j), so its mean is
    S1/n_avg and its variance is S2/n_avg^2 -- see `simulate_periodogram_stack`.
    The unmodulated case is P_mod = None, where every P_j = P_mean and the sums
    are n_avg P and n_avg P^2.

    Both are computed in closed form from sum(m) and sum(m^2), so the cost and
    the memory are independent of n_avg: no segment spectrum is ever
    materialised, let alone all n_avg of them. This is exact, not an
    approximation -- S2 expands as
    n_avg P^2 + 2 sum(m) P P_mod + sum(m^2) P_mod^2, and every term is positive,
    so there is no cancellation to worry about.

    G1: this function is also what the Stage-2 variance model should call if the
    modulation's variance inflation is ever folded in, since S2/n_avg^2 against
    (S1/n_avg)^2/n_avg IS that inflation. `modulation_variance_inflation` wraps
    exactly that ratio.
    """
    if P_mod is None:
        return n_avg * P_mean, n_avg * P_mean ** 2
    m_row = np.asarray(m_row, dtype=float)
    if m_row.shape != (n_avg,):
        raise ValueError(f"m_row has shape {m_row.shape}, expected {(n_avg,)}")
    s1m = float(m_row.sum())
    s2m = float((m_row ** 2).sum())
    S1 = n_avg * P_mean + s1m * P_mod
    S2 = n_avg * P_mean ** 2 + 2.0 * s1m * P_mean * P_mod + s2m * P_mod ** 2
    return S1, S2


def modulation_variance_inflation(P_mean: np.ndarray, n_avg: int, P_mod: np.ndarray,
                                  m_row: np.ndarray) -> np.ndarray:
    """Var(averaged periodogram) divided by what it would be if every segment
    shared the plane's mean power: (S2/n_avg^2) / (Pbar^2/n_avg), per frequency.

    Algebraically this is 1 + Var_j(P_j)/mean_j(P_j)^2 -- one plus the squared
    coefficient of variation of the per-segment expectations. With
    P_j = A + m_j B it is 1 + Var(m) (B/(A + <m> B))^2, i.e. the modulation's
    variance times the squared Galactic fraction of the total PSD. Over whole
    cycles <m> = 1, so the MEAN of the averaged plane is the unmodulated value
    and this ratio is the entire effect of the modulation at M = 1.
    """
    S1, S2 = segment_power_sums(P_mean, n_avg, P_mod, m_row)
    Pbar = S1 / n_avg
    out = np.ones_like(np.asarray(Pbar, dtype=float))
    good = Pbar > 0
    out[good] = (S2[good] / n_avg ** 2) / (Pbar[good] ** 2 / n_avg)
    return out


def simulate_periodogram_stack(P_mean: np.ndarray, n_realizations: int, tau: float, rng: np.random.Generator,
                               n_avg: int = 1, P_mod: Optional[np.ndarray] = None,
                               m: Optional[np.ndarray] = None,
                               draw: str = "normal") -> np.ndarray:
    """
    Draw `n_realizations` independent periodogram realizations whose
    expectation equals `P_mean` (shape (Nf,)). Returns (n_realizations, Nf).
    Same statistical model as the original script's simulate_periodogram,
    vectorized over realizations instead of a Python loop.

    n_avg (default 1, the original behaviour): each returned row is the mean of
    `n_avg` independent periodograms rather than a single one, i.e. the averaged
    periodogram over n_avg consecutive segments of duration tau. The expectation
    is unchanged; the variance is divided by n_avg. The mission has cfg.Nd
    segments available in total, so M * n_avg <= Nd is the physical budget.

    This exists so that the per-bin measurement noise can be held fixed while
    the number of coarse frequency bins N is varied at fixed band (experiment
    8d): coarse-graining into N bins averages ~Nf/N fine bins, so raising N
    makes every bin noisier in proportion, and n_avg buys that back exactly.

    M1, time-dependent spectra: pass P_mod (Nf,) and m (n_realizations, n_avg)
    and segment j of row r is drawn with expectation P_mean + m[r, j] * P_mod,
    so every segment carries its own modulation rather than the plane's mean
    -- which matters for the variance: averaging segments of unequal power
    gives var = sum_j P_j^2 / n_avg^2, not (mean P)^2 / n_avg. This path draws
    one row at a time (bounded memory at M = 126) and therefore consumes the
    stream in a different order; the P_mod = None path is untouched.

    G1, `draw`: which construction is used.

      "normal" (DEFAULT): the two Gaussian quadratures per segment, as above.
        Exact by construction and bit-for-bit reproducible against every draw
        taken before G1. Cost is 2 n_avg Nf normal variates per plane.

      "gamma": draw the AVERAGE directly. The sum of n_avg independent
        Exp(P_j) has mean S1 = sum_j P_j and variance S2 = sum_j P_j^2, so the
        Welch-Satterthwaite Gamma with
            k = S1^2 / S2 ,    theta = S2 / S1
        matches both exactly, and the average is Gamma(k, theta/n_avg).
        Unmodulated all P_j are equal, k = n_avg and theta = P, and this is the
        EXACT law of the mean of n_avg exponentials, not an approximation --
        Gamma(n_avg, P/n_avg). Modulated it is a two-moment match to a sum of
        exponentials with unequal scales. Cost is Nf Gamma variates per
        plane whatever n_avg is, which is what makes the 4 yr / n_avg = 126 pool
        affordable. It consumes a different stream, so it is opt-in.

    Both paths hold only O(Nf) per plane: the normal path accumulates each
    segment's squared modulus and discards it, and the gamma path never forms a
    segment spectrum at all (`segment_power_sums` is closed-form in n_avg).
    """
    if n_avg < 1:
        raise ValueError("n_avg must be >= 1")
    if (P_mod is None) != (m is None):
        raise ValueError("P_mod and m must be given together")
    if draw not in ("normal", "gamma"):
        raise ValueError(f"draw must be 'normal' or 'gamma', got {draw!r}")
    if draw == "gamma":
        if P_mod is not None:
            m = np.asarray(m, dtype=float)
            if m.shape != (n_realizations, n_avg):
                raise ValueError(f"m has shape {m.shape}, expected {(n_realizations, n_avg)}")
        out = np.empty((n_realizations, len(P_mean)))
        for r in range(n_realizations):
            if P_mod is None:
                # exact: the mean of n_avg iid Exp(P) is Gamma(n_avg, P/n_avg).
                # Written out rather than via S1, S2 so that k is exactly n_avg
                # and theta exactly P, with no round-off in the shape parameter.
                out[r] = rng.gamma(float(n_avg), P_mean / n_avg)
                continue
            S1, S2 = segment_power_sums(P_mean, n_avg, P_mod, m[r])
            k = np.zeros_like(S1)
            theta = np.zeros_like(S1)
            good = S2 > 0
            k[good] = S1[good] ** 2 / S2[good]
            theta[good] = S2[good] / S1[good]
            row = np.zeros_like(S1)
            row[good] = rng.gamma(k[good], theta[good] / n_avg)
            out[r] = row
        return out
    if P_mod is not None:
        m = np.asarray(m, dtype=float)
        if m.shape != (n_realizations, n_avg):
            raise ValueError(f"m has shape {m.shape}, expected {(n_realizations, n_avg)}")
        out = np.empty((n_realizations, len(P_mean)))
        for r in range(n_realizations):
            acc = np.zeros(len(P_mean))
            for j in range(n_avg):
                s = np.sqrt(tau * (P_mean + m[r, j] * P_mod) / 4.0)
                re = rng.normal(0.0, s)
                im = rng.normal(0.0, s)
                acc += re * re + im * im
            out[r] = 2.0 * acc / (n_avg * tau)
        return out
    sigma = np.sqrt(tau * P_mean / 4.0)
    acc = np.zeros((n_realizations, len(P_mean)))
    for _ in range(n_avg):                      # accumulate rather than allocate n_avg copies
        re = rng.normal(0.0, sigma, size=(n_realizations, len(P_mean)))
        # M1: the imaginary parts are drawn a few rows at a time. Consecutive
        # calls continue the same stream, so the values are identical to one
        # (n_realizations, Nf) call, but the peak memory at M = 126 falls from
        # ~3 GB (re, im, their complex sum and its modulus) to ~1 GB.
        for r0 in range(0, n_realizations, 8):
            r1 = min(r0 + 8, n_realizations)
            im = rng.normal(0.0, sigma, size=(r1 - r0, len(P_mean)))
            # np.abs on the complex value, not re**2 + im**2: it rounds differently
            # (hypot), and keeping the original expression makes n_avg = 1 bit-for-bit
            # identical to every draw taken before this argument existed.
            acc[r0:r1] += np.abs(re[r0:r1] + 1j * im) ** 2
        del re
    acc *= 2.0                                  # same operations, same order, as
    acc /= n_avg * tau                          # 2.0 * acc / (n_avg * tau), in place
    return acc


# --------------------------------------------------------------------------- #
# Coarse-graining (inverse-variance weighted — same scheme as
# bpl_likelihood_new.py's coarse_grain_channel, generalized to per-event P_th)
# --------------------------------------------------------------------------- #

def make_log_edges(f: np.ndarray, n_bins: int) -> np.ndarray:
    return np.logspace(np.log10(f[0]), np.log10(f[-1]), n_bins + 1)


def coarse_grain_reference(f: np.ndarray, cfg: MissionConfig, channel: str):
    """
    (P_ref, R) for one TDI channel: the spectrum whose variance sets the
    coarse-graining weights, and that channel's response.

    A-fix. The weight reference has to be the SAME at every call site, or the
    generator's targets and Stage-2's frequency grid land on different effective
    bin centres -- which is exactly audit finding A4. That rules out using each
    event's own periodogram, since then every event would be binned onto its own
    grid and a training batch could not be stacked at all. It equally rules out
    the injected total P_th the generator used to use, which is not available in
    a real analysis (finding A3).

    What is left, and what SGWBinner itself uses, is a fiducial instrument
    model: Flauger et al. arXiv:2009.11845 take the variance "to be well
    approximated by LISA's sensitivity" rather than from the data or the truth.
    So P_ref is the nominal instrument noise PSD at the configured budget with
    no jitter -- known before any data arrives, identical at every call site,
    and independent of the injected signal.

    Pass your own array as `P_th` to coarse_grain if you want a different
    reference (e.g. an observed Dbar); nothing here forbids it, but the grid
    then stops being shared and A4 reopens.
    """
    x, R_A, R_E, R_T, _, _ = detector_response(f, cfg)
    PN_A, PN_E, PN_T = noise_psd(f, x, cfg, rng=None)        # fiducial, unjittered
    P_ref = {"A": PN_A, "E": PN_E, "T": PN_T}[channel]
    R = {"A": R_A, "E": R_E, "T": R_T}[channel]
    return P_ref, R


def noise_budget_split(f: np.ndarray, cfg: MissionConfig):
    """Acceleration share of each channel's instrument noise, s_ch(f).

    N_ch = g_ch(f) [b_TM,ch(f) S_TM(f,A) + b_OMS,ch(f) S_OMS(f,P)], and both
    budget terms are quadratic in their parameter, so

        d ln N_ch / d ln A = 2 s_ch ,      d ln N_ch / d ln P = 2 (1 - s_ch)

    with s_ch the acceleration share. Returns (s_A, s_T); E shares A's.

    The two channels sit at opposite ends of this. At 3e-5 Hz the A/E noise is
    0.9991 acceleration and the T noise is 0.0007; at 1 mHz it is 0.875 against
    0.005. The T channel is very nearly pure optical metrology everywhere A and
    E are acceleration-dominated, which is why it is the lever on P -- and why
    it is worth keeping in the fit even with no GW signal fitted in it.
    """
    x, _, _, _, _, _ = detector_response(f, cfg)
    STM = (cfg.A_param * 1e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4) \
        * (1 / (2 * np.pi * f * C_LIGHT)) ** 2
    SOMS = (cfg.P_param * 1e-12) ** 2 * (1 + (2e-3 / f) ** 4) * (2 * np.pi * f / C_LIGHT) ** 2
    acc_A = 4 * (1 + np.cos(x) + np.cos(x) ** 2) * STM
    oms_A = (2 + np.cos(x)) * SOMS
    acc_T = 2 * (1 - np.cos(x)) ** 2 * STM
    oms_T = (1 - np.cos(x)) * SOMS
    return acc_A / (acc_A + oms_A), acc_T / (acc_T + oms_T)


#: Fractional arm-length mismatch Delta L / L assumed when deciding where the
#: idealised equal-arm T null is too deep to be believed. One per cent is the
#: scale of LISA's constellation breathing. It is a stated modelling input, not
#: a fitted number. See null_channel_valid for how it defines the cut.
ARM_MISMATCH = 0.01


def null_channel_valid(f: np.ndarray, cfg: MissionConfig,
                       delta: float = ARM_MISMATCH) -> np.ndarray:
    """Where is it safe to fit the T channel with N_T alone?

    T4. Dropping S_T from the T theory is an approximation, and its error is

        S_T/N_T = pref Omega R_T / (f^3 N_T) = Omega / Omega_sens_T

    exactly. That is tiny over almost the whole band -- 1.8e-15 at 3e-5 Hz on
    the Table-4 benchmark -- but at an EVEN multiple of c/2L the idealised
    equal-arm null sends N_T to zero as (f - f_even)^4 while R_T only goes as
    (f - f_even)^2, so S_T/N_T reaches 5.5e6 there. The model then claims the T
    channel has essentially no noise at a frequency where the data plainly has
    something, and the Gamma term -Nd Dbar/N_T explodes: measured, one bin at
    0.119917 Hz contributes -5.13e7 log-units and makes the exact likelihood
    prefer a demonstrably wrong parameter vector.

    The bins to exclude are exactly the ones where the idealised suppression is
    deeper than a real constellation could produce. With arms L_i = L(1+delta_i)
    the phases spread by delta*x and the cancellation cannot go below
    0.5 (delta x)^2, so the criterion is

        keep where   (1 - cos x)  >=  0.5 (delta x)^2

    which is T2's residual-floor derivation used as a statement about where the
    model is trustworthy rather than as a change to the noise PSD. At
    delta = 1% it removes 24027 of 499971 fine bins, 4.8% of the band, in
    windows of half-width n*delta*c/2L around the n-th harmonic, and bounds the
    neglected S_T to 1.1e-3 of N_T everywhere that is kept -- which is the same
    bound the floored model achieves without any mask at all (1.08e-3). The two
    agree because they are the same physical statement.
    """
    x = 2 * np.pi * f * cfg.L / C_LIGHT
    return (1 - np.cos(x)) >= 0.5 * (delta * x) ** 2


def coarse_grain_weight(f: np.ndarray, cfg: MissionConfig, channel: str,
                        t_null: bool = True) -> np.ndarray:
    """
    The compression weight for one channel: the single source of truth the
    binning contract requires, so make_training_batch, multichannel_f_bin and
    save_stage2_data cannot drift apart.

    A and E (and T, if t_null=False) get the Fisher information about a GW
    amplitude, q = (dP/dOmega)^2/Var(P) ~ (R/f^3/P_ref)^2 -- B2's choice,
    unchanged.

    T4: under the null-channel treatment (t_null, the default) the T channel
    carries no Omega dependence in the likelihood, so a weight built from
    dP_T/dOmega compresses a quantity T is no longer being used to measure.
    What T measures is the instrument noise, so its weight is the Fisher
    information about the noise parameters,

        q_T = (d ln N_T/d ln A)^2 + (d ln N_T/d ln P)^2
            = 4 s_T^2 + 4 (1 - s_T)^2

    which is the diagonal of the T contribution to arXiv:2511.00996 Eq. (5.2),
    summed over the two noise parameters. It is the same construction as the
    Omega weight -- (d ln P/d theta)^2, since Var(Dbar) = P^2/Nd makes the
    variance cancel into a constant -- applied to the parameter T actually
    constrains.

    The practical consequence is that q_T becomes almost flat: s_T is below
    0.03 across the band, so q_T sits between 3.807 and 4.000 everywhere,
    INCLUDING at the even response nulls where the Omega weight diverges as
    (f - f_even)^-4.
    """
    if channel not in ("A", "E", "T"):
        raise ValueError(f"channel must be A, E or T, got {channel!r}")
    if channel == "T" and t_null:
        _, s_T = noise_budget_split(f, cfg)
        return 4 * s_T ** 2 + 4 * (1 - s_T) ** 2
    P_ref, R = coarse_grain_reference(f, cfg, channel)
    return (R / f ** 3 / P_ref) ** 2


def detectability_sigma_frac(cfg: MissionConfig, N: int, channel: str,
                              M: int, n_avg: int) -> np.ndarray:
    """Fractional uncertainty on ONE coarse bin of the averaged periodogram.

    This is the number CAL2 measured the injected signal against, and it is the
    floor below which no estimator of any kind can recover a signal from this
    data -- not a property of the network, of the loss, or of the fit.

    A coarse bin is the q-weighted mean of its fine bins, and a periodogram fine
    bin at frequency f has variance P(f)^2, so the weighted mean has variance
    sum_i q_i^2 P_i^2 / (sum_i q_i)^2 against a mean of sum_i q_i P_i / sum_i
    q_i. The network sees M planes each averaging n_avg segments, hence

        sigma_frac = sqrt(sum_i q_i^2 P_i^2) / (sum_i q_i P_i) / sqrt(M n_avg)

    The naive figure 1/sqrt(M n_avg) is the special case of one fine bin per
    coarse bin. It is conservative by one to two orders of magnitude at the top
    of the band, where this grid puts 36568 fine bins into the last coarse bin
    against 3 in the first: CAL2 measured 0.118 at the bottom of the band,
    0.0136 in the core and 0.00248 above 30 mHz, against the flat 0.204.

    P AND q COME FROM THE FIDUCIAL INSTRUMENT MODEL, via the same
    coarse_grain_reference / coarse_grain_weight pair the binning contract
    already routes every call site through, so the criterion is identical for
    every event and every channel gets the weight T4 assigned it. That is the
    A-fix argument applied unchanged: a criterion built from each event's own
    total would put every event on its own threshold. The approximation costs
    almost nothing -- recomputed per event from its own injected total over
    CAL2's 400 trials, sigma_frac moves by at most 0.8 per cent anywhere in the
    band, because it is set by how many fine bins a coarse bin holds and by the
    shape of the weight, not by the signal sitting on top of the noise.

    Returns (N,) for the given channel; +inf in any bin with no fine samples,
    which reads as "never detectable" and is the safe direction.
    """
    f = build_frequency_grid(cfg)
    edges = make_log_edges(f, N)
    P_ref, _ = coarse_grain_reference(f, cfg, channel)
    q = coarse_grain_weight(f, cfg, channel)
    bin_idx = np.clip(np.digitize(f, edges) - 1, 0, N - 1)
    num = np.bincount(bin_idx, weights=(q * P_ref) ** 2, minlength=N)
    den = np.bincount(bin_idx, weights=q * P_ref, minlength=N)
    out = np.full(N, np.inf)
    ok = den > 0
    out[ok] = np.sqrt(num[ok]) / den[ok] / np.sqrt(M * n_avg)
    return out


def multichannel_sigma_frac(cfg: MissionConfig, N: int, channels: Sequence[str],
                             M: int, n_avg: int) -> np.ndarray:
    """detectability_sigma_frac per channel, stacked (n_ch, N).

    Channel-slowest like everything else crossing this boundary (F3), so it
    broadcasts directly against a (B, n_ch, N) target.
    """
    return np.stack([detectability_sigma_frac(cfg, N, ch, M, n_avg)
                     for ch in channels])


def coarse_grain(f: np.ndarray, values: np.ndarray, P_th: np.ndarray, edges: np.ndarray,
                  fill_empty: bool = True, floor: float = 1e-300,
                  weight_mode: str = "fisher", R: Optional[np.ndarray] = None,
                  q: Optional[np.ndarray] = None):
    """
    Inverse-variance-weighted coarse-graining — the same scheme as your
    Cobaya likelihood's `coarse_grain_channel`, vectorized with bincount
    instead of a per-bin Python loop, and generalized so P_th can be a fresh
    per-event theory curve rather than one fixed global truth.

    Weights come from 1/P_th^2 (a periodogram estimate at frequency f has
    variance P_th(f)^2, so this is the minimum-variance combination) —
    always computed from the *total* theoretical PN+PS, since that governs
    the statistics of the observed data regardless of which quantity
    (an observed draw, or a theoretical PN/PS curve on its own) you're
    binning. Using the same weights for x / n_true / s_true keeps them
    aligned to identical bins.

    `values` may be 1-D (Nf,) or 2-D (..., Nf) (e.g. M stacked realizations).

    weight_mode selects the weight q(f) used to combine fine bins:

      "inv_var" (the pre-A-fix behaviour, kept for the comparison figures):
        q = 1/P_th^2. This is the
        minimum-variance combination for estimating the MEAN of the spectrum,
        and it is what SGWBinner uses (Flauger et al. arXiv:2009.11845
        eq. 3.9). It has two problems for our purposes. It needs P_th, which
        on real data means it needs the answer. And it DIVERGES at a TDI
        response null: R(f) contains sin^2(2*pi*f*L/c), which is exactly zero
        at f = n*c/2L, so P_th -> 0 there and q -> infinity. The weight is
        maximal precisely where the instrument is blind.

      "fisher" (DEFAULT since the A-fix): q = (dP_th/dOmega)^2 / Var(P_th) = (pref * R / f^3 / P_th)^2,
        up to a constant. This is the Fisher information about a GW amplitude,
        i.e. the right weight for a compression whose output will be fed to a
        parameter fit rather than used as an estimate of the spectrum itself.
        At a null, the sensitivity dP/dOmega is proportional to R and vanishes
        at the same rate as the variance, so the ratio sends those frequencies
        to a FINITE weight rather than an infinite one. (Measured: R and P_th
        both vanish as sin^2(2 pi f L / c), so the ratio is constant through a
        null to five significant figures -- what is removed is the divergence,
        not the contribution.) Requires `R`, the channel's response on the same
        fine grid; see coarse_grain_reference for the intended P_th.

    Either way, pass an OBSERVED spectrum as P_th (rather than an injected
    truth) if you want a scheme reproducible on real data; the weights only
    need a variance estimate, not the true answer.

    fill_empty (default True): bins with zero raw samples (this grid starts
    getting them above ~370 bins — see module docstring) are filled by
    log-log linear interpolation from their nearest non-empty neighbors,
    rather than left as NaN. Safe here specifically because the empty bins
    for this f_min/f_max/df grid sit right at the low-frequency edge, where
    every quantity in this file varies smoothly (verified: no oscillatory
    structure down there — that only shows up above ~0.06 Hz, see the
    diagnostic plot). This is the SAME technique (and same justification)
    as an earlier version of this file's rebin_logf function, just now
    scoped to the handful of genuinely-empty bins rather than applied
    everywhere. Set False to get the original NaN-marking behavior (e.g. if
    you want to explicitly mask incomplete bins yourself, as the Bayesian
    likelihood files do with their own separate coarse-graining).
    """
    n_bins = len(edges) - 1
    bin_idx = np.clip(np.digitize(f, edges) - 1, 0, n_bins - 1)
    if q is not None:
        # T4: an explicit weight, supplied by coarse_grain_weight. Used for the
        # T channel under the null-channel treatment, where the weight is the
        # Fisher information about the NOISE parameters and cannot be written
        # as a function of (R, P_th).
        q = np.asarray(q, dtype=float)
        if q.shape != f.shape:
            raise ValueError(f"explicit q has shape {q.shape}, expected {f.shape}")
    elif weight_mode == "inv_var":
        q = 1.0 / P_th ** 2
    elif weight_mode == "fisher":
        if R is None:
            raise ValueError("weight_mode='fisher' needs the channel response R "
                             "on the same fine grid as f (dP/dOmega ~ R/f^3).")
        q = (R / f ** 3 / P_th) ** 2
    else:
        raise ValueError(f"weight_mode must be 'inv_var' or 'fisher', got {weight_mode!r}")
    sum_q = np.bincount(bin_idx, weights=q, minlength=n_bins)
    valid = sum_q > 0

    f_c = np.full(n_bins, np.nan)
    f_c[valid] = np.bincount(bin_idx, weights=q * f, minlength=n_bins)[valid] / sum_q[valid]

    P_c = np.zeros(n_bins)
    P_c[valid] = np.bincount(bin_idx, weights=q * P_th, minlength=n_bins)[valid] / sum_q[valid]
    w_c = np.zeros(n_bins)
    w_c[valid] = sum_q[valid] * P_c[valid] ** 2

    orig_shape = values.shape
    flat = values.reshape(-1, orig_shape[-1])
    out = np.full((flat.shape[0], n_bins), np.nan)
    for row in range(flat.shape[0]):
        num = np.bincount(bin_idx, weights=q * flat[row], minlength=n_bins)
        out[row, valid] = num[valid] / sum_q[valid]

    if fill_empty and (~valid).any() and valid.sum() >= 2:
        bin_centers = np.sqrt(edges[:-1] * edges[1:])  # geometric center, for empty bins only
        f_c[~valid] = bin_centers[~valid]
        log_f_all = np.log10(f_c)
        log_f_valid = log_f_all[valid]
        for row in range(flat.shape[0]):
            log_v_valid = np.log10(np.maximum(out[row, valid], floor))
            log_v_interp = np.interp(log_f_all, log_f_valid, log_v_valid)
            out[row, ~valid] = 10 ** log_v_interp[~valid]

    out = out.reshape(*orig_shape[:-1], n_bins)

    return f_c, out, w_c


# --------------------------------------------------------------------------- #
# Training batch generator
# --------------------------------------------------------------------------- #

def make_training_batch(
    batch_size: int,
    M: int,
    N: int,
    cfg: MissionConfig,
    priors: SignalPriors,
    rng: np.random.Generator,
    amplitude_scale=1.0,
    category_probs=(0.25, 0.25, 0.25, 0.25),
    channels: Sequence[str] = ("A",),
    eps: float = 1e-60,
    signal_floor: Optional[float] = None,
    fixed_params: Optional[dict] = None,
    normalize: bool = True,
    n_avg: int = 1,
    return_meta: bool = False,
    draw: str = "normal",
):
    """
    Returns numpy arrays (x, n_true, s_true, category), and with
    return_meta=True a fifth element, a dict with
        t_plane  : (B, M) start time [s] of each input plane, t0 + i*n_avg*tau
        m_plane  : (B, M) mean Galactic modulation factor over each plane
                   (all ones when the event carries no modulation)
        params   : list of B event parameter dicts

        x        : (B, M * n_ch, N)
        n_true   : (B, n_ch, N)
        s_true   : (B, n_ch, N)
        category : (B,) int in {0,1,2,3}

    LAYOUT (F3, audit finding A1). TDI channels live on the CHANNEL axis, never
    along frequency. On x's channel axis the channel index varies SLOWEST:

        x[:, ci * M + m, :]  is channel `channels[ci]`, time segment m
        n_true[:, ci, :]     is channel `channels[ci]`
        s_true[:, ci, :]     is channel `channels[ci]`

    so the axis reads [ch0 seg0 .. ch0 seg(M-1), ch1 seg0 .. ch1 seg(M-1), ...].
    N is the per-channel frequency resolution and is the FULL width of the
    frequency axis; it is no longer multiplied by the channel count anywhere.

    This replaces concatenating channels along frequency, which put the top of
    one channel's band numerically adjacent to the bottom of the next. Under
    that layout the Inception kernels (k = 3, 5, 7) slid straight across the
    junction, the Eq. 14 curvature penalty differenced across it, and max-pooling
    paired a bin of one channel with a bin of another whenever N was odd. None
    of those are meaningful operations: two TDI channels at the same frequency
    are different measurements, and the same channel at adjacent frequencies is
    a spectrum -- the old layout confused the two.

    A and E are statistically independent measurements under the equal-arm
    approximation (same PN/R shape, but genuinely separate noise
    realizations — this file draws them independently, not by copying one
    channel's draw), so adding a channel is a second, complementary way to give
    the network more independent information about the same underlying event,
    alongside M. T is the null/insensitive channel and is usually kept separate
    as an instrument-noise calibration check rather than fed into the same
    signal-separation network — you can still pass it in `channels` if you have
    a specific reason to.

    amplitude_scale: scalar, or an array of length batch_size giving each
    event its own curriculum amplitude. The array form is what the Stage-C
    pool uses (P4): its amplitudes come from the epoch-averaged schedule and
    therefore vary within a chunk, and calling this per event instead would
    discard the G1 hoist that makes the pool affordable.

    fixed_params: if given, every sample in the batch uses these exact
    signal parameters (only the noise realization is redrawn per sample).
    This is the "same event, many noise draws" mode needed later to
    empirically estimate sigma_log(f) for the Stage-2 MCMC likelihood —
    generate a batch with fixed_params set, run it through the frozen
    trained model, and take the per-frequency std of the resulting s_hat.
    P8: it may instead be a sequence of batch_size dicts, one per sample --
    the simulation-based-inference training set, where every sample has its
    own parameters drawn from the inference prior. The stream is consumed in
    the same order as batch_size separate batch-1 calls, so a list reproduces
    those calls exactly while keeping the G1 hoist.

    n_avg: how many segments of duration cfg.tau are averaged into each of the
    M spectra on the input axis (default 1 — one segment per input plane, the
    original behaviour). M * n_avg segments are consumed in total and the
    mission has cfg.Nd of them, which this checks. Averaging n_avg segments
    divides the per-fine-bin variance by n_avg without touching the mean, so
    this is the knob that holds the per-coarse-bin noise fixed when N is
    changed at fixed band: a coarse bin averages ~Nf/N fine bins, so its
    fractional variance goes as N/(Nf * n_avg) and n_avg proportional to N
    keeps it constant. Used by experiment 8d; production uses the default.

    M1, TIME. Plane i starts at t_i = t0 + i * n_avg * tau, with t0 taken from
    the event's parameters (drawn per event when priors.gal_modulation is on;
    0 otherwise). When the event carries modulation parameters, every one of
    the M * n_avg segments is drawn from its own expectation
    PN + S_stationary + m(t) S_Gal, with m averaged over that segment, and the
    SIGNAL TARGET is the time average over all of them,
    S_stationary + mean(m) S_Gal -- a (B, n_ch, N) spectrum as before, so
    Stage 2 is unchanged in shape. Instrument noise, the extragalactic
    background and the phase transition are never modulated.

    draw: "normal" (default) or "gamma", passed to simulate_periodogram_stack.
    "gamma" draws each averaged plane directly as a Gamma variate instead of
    building it from 2 * n_avg * Nf normal variates. At n_avg = 1 unmodulated it
    is the exact same law (Exp(P)) and at n_avg > 1 unmodulated it is again
    exact; modulated it is a two-moment (Welch-Satterthwaite) match. It changes
    the random stream, so it is opt-in; G1 validated it and it is what makes the
    Stage-C pool at n_avg = 126 affordable.

    normalize: if True (default), x/n_true/s_true are affinely rescaled via
    normalize_log_psd so they're O(1) instead of O(-45) — see that
    function's docstring for why this matters for training stability. Pass
    False to get raw physical log10(PSD) values instead (e.g. for
    diagnostic plotting).
    """
    if M * n_avg > cfg.Nd:
        raise ValueError(f"M * n_avg = {M * n_avg} exceeds the cfg.Nd = {cfg.Nd} "
                         "segments the mission provides")
    if (fixed_params is not None and not isinstance(fixed_params, dict)
            and len(fixed_params) != batch_size):
        raise ValueError(f"fixed_params has {len(fixed_params)} dicts for "
                         f"batch_size = {batch_size}")
    # P4: amplitude_scale may be a scalar (every event of the batch at the same
    # curriculum amplitude -- the training-loop usage) or a length-batch_size
    # array (one amplitude per event -- the pooled usage, where the pool's
    # amplitudes come from the epoch-AVERAGED schedule and so differ within a
    # chunk). Without the array form a pool build would have to call this with
    # batch_size = 1, losing the G1 hoist of detector_response and the
    # coarse-graining reference, which is ~1.4 s/event at n_avg = 126 against a
    # total of ~1.06 s/event -- it would more than double the pool's cost.
    amp = np.broadcast_to(np.asarray(amplitude_scale, dtype=float), (batch_size,))
    s_floor = SIGNAL_FLOOR_PSD if signal_floor is None else signal_floor
    f = build_frequency_grid(cfg)
    edges = make_log_edges(f, N)
    n_ch = len(channels)
    x_out = np.zeros((batch_size, M * n_ch, N), dtype=np.float32)
    n_out = np.zeros((batch_size, n_ch, N), dtype=np.float32)
    s_out = np.zeros((batch_size, n_ch, N), dtype=np.float32)
    cat_out = np.zeros(batch_size, dtype=np.int64)
    t_out = np.zeros((batch_size, M))
    m_out = np.ones((batch_size, M))
    params_out = []

    # G1: hoisted out of the per-event loop. All three depend only on (f, cfg),
    # never on the event, so recomputing them per event was pure cost -- about
    # 1.4 s of the 13 s/event at n_avg = 126, and dominant once the Gamma draw
    # removes the 2 * n_avg * Nf normal variates. Deterministic, so the random
    # stream and every previously measured draw are unchanged.
    xg, R_A, R_E, R_T, _, _ = detector_response(f, cfg)
    ref_cache = {ch: (coarse_grain_reference(f, cfg, ch), coarse_grain_weight(f, cfg, ch))
                 for ch in channels}

    for b in range(batch_size):
        category = rng.choice([0, 1, 2, 3], p=category_probs)
        cat_out[b] = category

        PN_A, PN_E, PN_T = noise_psd(f, xg, cfg, rng=rng)  # one noise-budget jitter per event, shared across its channels (same instrument)

        if fixed_params is None:
            params = sample_event_params(rng, priors, category, cfg,
                                         amplitude_scale=float(amp[b]))
        elif isinstance(fixed_params, dict):
            params = fixed_params
        else:
            params = fixed_params[b]          # P8: one parameter dict per sample
        # channel-independent (only R below varies by channel)
        Omega_stat, Omega_G = omega_split(f, params, priors, cfg)
        Omega = Omega_stat + Omega_G         # same sum, same order, as total_omega

        t_plane = segment_times(M, n_avg, cfg, params.get("t0", 0.0))
        t_out[b] = t_plane
        params_out.append(params)
        modulated = "mod_A1" in params and bool(Omega_G.any())
        if modulated:
            t_seg = t_plane[:, None] + cfg.tau * np.arange(n_avg)[None, :]   # (M, n_avg)
            m_seg = gal_modulation_factor(t_seg, t_seg + cfg.tau, params)
            m_out[b] = m_seg.mean(axis=1)

        for ci, ch in enumerate(channels):
            PN = {"A": PN_A, "E": PN_E, "T": PN_T}[ch]
            R = {"A": R_A, "E": R_E, "T": R_T}[ch]
            if modulated:
                PS_stat = cfg.pref * Omega_stat / f ** 3 * R
                PS_gal = cfg.pref * Omega_G / f ** 3 * R
                PS = PS_stat + m_seg.mean() * PS_gal        # time-averaged target
                realizations = simulate_periodogram_stack(PN + PS_stat, M, cfg.tau, rng,
                                                          n_avg=n_avg, P_mod=PS_gal, m=m_seg,
                                                          draw=draw)
            else:
                PS = cfg.pref * Omega / f ** 3 * R
                P_total = PN + PS
                realizations = simulate_periodogram_stack(P_total, M, cfg.tau, rng, n_avg=n_avg, draw=draw)  # (M, Nf), fresh independent draw per channel

            # A-fix: Fisher weights from the FIDUCIAL instrument model, not from
            # this event's injected P_total. Two reasons, both from audit 01.
            # The old weights needed the answer, so they could not be
            # reproduced in a real analysis (A3); and being event-dependent
            # they put every event on its own effective bin centres, which no
            # downstream consumer could reconstruct (A4). The reference here is
            # identical at every call site -- see coarse_grain_reference.
            # T4: the weight now comes from coarse_grain_weight, which gives A
            # and E the unchanged Fisher-about-Omega weight and gives T the
            # Fisher-about-the-noise-parameters weight. The SIGNAL is still
            # simulated in T -- PS above is the physical T-channel signal and is
            # binned and emitted as a target exactly as before. Only the
            # weighting changes; see the note in T4 on why the simulator and the
            # likelihood are deliberately asymmetric here.
            (P_ref, R_ref), q_ref = ref_cache[ch]
            _, PN_bin, _ = coarse_grain(f, PN, P_ref, edges, R=R_ref, q=q_ref)
            _, PS_bin, _ = coarse_grain(f, PS, P_ref, edges, R=R_ref, q=q_ref)
            _, x_bin, _ = coarse_grain(f, realizations, P_ref, edges, R=R_ref, q=q_ref)  # (M, N)

            # F3: channel index varies SLOWEST on the channel axis, so rows
            # ci*M .. (ci+1)*M-1 are the M time segments of channel `ci`.
            # N1/A9: the SIGNAL target is floored at a physical level that means
            # "no detectable signal here" -- SIGNAL_FLOOR_PSD, log10 = -54, the
            # same floor D3's hinge uses -- while the noise target and the
            # inputs keep the bare numerical guard `eps`. The two must not share
            # a constant. After T2 the T-channel NOISE target is a genuine
            # 22-decade ramp reaching log10 PSD = -58.5, so flooring it at -54
            # would clip four and a half decades of real instrument noise; it is
            # only the signal that has a level below which the right statement
            # is a detection statement rather than a number.
            x_out[b, ci * M:(ci + 1) * M, :] = np.log10(np.maximum(x_bin, eps))
            n_out[b, ci, :] = np.log10(np.maximum(PN_bin, eps))
            s_out[b, ci, :] = np.log10(np.maximum(PS_bin, s_floor))

    if normalize:
        x_out = normalize_log_psd(x_out)
        n_out = normalize_log_psd(n_out)
        s_out = normalize_log_psd(s_out)

    if return_meta:
        return x_out, n_out, s_out, cat_out, dict(t_plane=t_out, m_plane=m_out,
                                                  params=params_out)
    return x_out, n_out, s_out, cat_out


def multichannel_f_bin(cfg: MissionConfig, N: int, channels: Sequence[str]) -> np.ndarray:
    """
    The flattened frequency axis matching make_training_batch's output, channel
    slowest: the bin centres each channel's spectra were actually binned onto,
    concatenated once per channel.

    A-fix. This used to bin with uniform weights (P_th = ones) while the
    generator binned with inverse-variance weights from the injected truth, so
    the axis disagreed with the data living on it by up to 2.35 per cent in
    frequency -- audit finding A4. Both now use the same Fisher weights against
    the same fiducial reference, so the centres agree by construction.

    The centres are per-CHANNEL and no longer a tile of one array: the weights
    contain the channel's own response, so A and T bin onto slightly different
    effective frequencies. Flauger et al. note the same thing below their
    eq. (3.9).
    """
    f = build_frequency_grid(cfg)
    edges = make_log_edges(f, N)
    parts = []
    for ch in channels:
        P_ref, R_ref = coarse_grain_reference(f, cfg, ch)
        f_c, _, _ = coarse_grain(f, np.zeros_like(f), P_ref, edges, R=R_ref,
                                 q=coarse_grain_weight(f, cfg, ch))
        parts.append(f_c)
    return np.concatenate(parts)


# Signal prior used for neural posterior estimation
SBI_PARAMETER_NAMES = ("log10_h2Omega_Gal", "log10_h2Omega_Ext", "log10_h2Omega_b", "log10_fb")
SBI_PRIOR_BOUNDS = {
    "log10_h2Omega_Gal": (-8.5, -7.0),
    "log10_h2Omega_Ext": (-13.0, -11.5),
    "log10_h2Omega_b":   (-12.5, -10.5),
    "log10_fb":          (-4.0, -2.0),
}


def draw_sbi_parameters(rng) -> np.ndarray:
    return np.array([rng.uniform(*SBI_PRIOR_BOUNDS[n]) for n in SBI_PARAMETER_NAMES])


def sbi_parameters_to_event(theta: np.ndarray, priors: SignalPriors) -> dict:
    """Map the sampled log parameters to the simulator's event format."""
    return dict(
        h2Omega_gal=10 ** theta[0], h2Omega_ext=10 ** theta[1],
        h2Omega_b=10 ** theta[2], fb=10 ** theta[3],
        n1=priors.n1, n2=priors.n2, alpha=priors.alpha,
    )
