"""Build and load the separator training pool.

The training pool is sampled from the mixture of amplitude distributions
across the full curriculum. Training reweights the same pool each epoch.
The separate validation split is generated once at physical signal strength.

Usage:
    python -m lisa_gwb.separator_data --out results/stage_c_pool
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from lisa_gwb.curriculum import CurriculumAmplitudeSampler, CurriculumConfig
from lisa_gwb.simulation import (
    MissionConfig,
    SignalPriors,
    build_frequency_grid,
    check_normalisation_stamp,
    make_training_batch,
    multichannel_f_bin,
    normalisation_stamp,
)

#: Epochs the curriculum's epoch-averaged proposal averages over. 200 is P1's
#: production run length; `CurriculumConfig.E_curr = 100` is only the annealing
#: phase, after which `a_min` stops moving and epochs 100-200 share one law.
N_EPOCHS = 200

#: The normalised window. N1 measured the physical range onto [-0.900, +0.861];
#: D3 established that a target outside [-1, +1] becomes the single largest
#: term in the objective. Not a tunable -- it is the representation's domain.
WINDOW = 1.0


# --------------------------------------------------------------------------- #
# The epoch-averaged amplitude proposal, and the per-epoch reweighting
# --------------------------------------------------------------------------- #
def epoch_averaged_amplitudes(n: int, rng: np.random.Generator,
                              sampler: CurriculumAmplitudeSampler | None = None,
                              n_epochs: int = N_EPOCHS):
    """Draw `n` amplitudes from p_bar = (1/E) sum_e p_e, P2a section 4.

    Returns (A, epoch): the amplitude and the epoch index it was drawn under.
    The epoch is stored with the pool because it is the label that makes the
    mixture a *defensive* one -- knowing which component produced each event is
    not needed for the weights (they are closed form) but it is what lets the
    realised proposal be checked against the intended one after the fact.

    Drawn epoch-by-epoch rather than one at a time so the whole pool costs
    `n_epochs` vectorised normal draws, not `n`.
    """
    sampler = sampler or CurriculumAmplitudeSampler(CurriculumConfig(), rng=rng)
    epoch = rng.integers(1, n_epochs + 1, size=n)
    A = np.empty(n, dtype=float)
    for e in np.unique(epoch):
        sel = epoch == e
        A[sel] = sampler.sample(int(e), size=int(sel.sum()))
    return A, epoch.astype(np.int64)


def _clipped_lognormal_density(A, mu, sigma, a_min, a_max, smear_dex):
    """Density of `clip(LogNormal(mu, sigma), a_min, a_max)` with respect to the
    log10-amplitude measure, with the two clip ATOMS smeared over `smear_dex`.

    The curriculum's law is not absolutely continuous: `sample` clips, so mass
    piles onto exactly `a_min(e)` and exactly `A_max`. P2a's fix is to smear
    each atom over +-`smear_dex` (default 0.02 dex, half the schedule's own
    0.040 dex per-epoch step), so the substitution an epoch makes is smaller
    than the difference between consecutive epochs of the curriculum itself.
    """
    x = np.log10(np.asarray(A, dtype=float))
    ln10 = np.log(10.0)
    # continuous part, as a density in log10 A
    z = (x * ln10 - mu) / sigma
    cont = np.exp(-0.5 * z ** 2) / (sigma * np.sqrt(2 * np.pi)) * ln10
    inside = (A > a_min) & (A < a_max)
    dens = np.where(inside, cont, 0.0)
    # atoms: total mass below a_min and above a_max, spread over a 2*smear box
    from math import erf, sqrt, log
    def _Phi(v):
        return 0.5 * (1.0 + erf(v / sqrt(2.0)))
    p_lo = _Phi((log(a_min) - mu) / sigma)
    p_hi = 1.0 - _Phi((log(a_max) - mu) / sigma)
    lo_x, hi_x = np.log10(a_min), np.log10(a_max)
    if p_lo > 0:
        box = np.abs(x - lo_x) <= smear_dex
        dens = dens + np.where(box, p_lo / (2 * smear_dex), 0.0)
    if p_hi > 0:
        box = np.abs(x - hi_x) <= smear_dex
        dens = dens + np.where(box, p_hi / (2 * smear_dex), 0.0)
    return dens


def epoch_density(A, epoch: int, sampler: CurriculumAmplitudeSampler | None = None,
                  smear_dex: float = 0.02):
    """p_e(A), the curriculum's epoch-`epoch` amplitude density in log10 A."""
    sampler = sampler or CurriculumAmplitudeSampler(CurriculumConfig())
    cfg = sampler.cfg
    return _clipped_lognormal_density(A, sampler.mu_ln(epoch), sampler.sigma_ln(epoch),
                                      sampler.a_min(epoch), cfg.A_max, smear_dex)


def epoch_weights(A, epoch: int, sampler: CurriculumAmplitudeSampler | None = None,
                  n_epochs: int = N_EPOCHS, smear_dex: float = 0.02,
                  normalise: bool = True):
    """P2a's per-epoch importance weights `w_i = p_e(A_i) / p_bar(A_i)`.

    This is what turns one pool into a per-epoch training distribution: draw
    each epoch's minibatches with probability proportional to `w`, and the
    epoch sees the curriculum's own law rather than the pool's. `p_bar` is the
    same mixture the pool was drawn from, so the weights are bounded by
    `n_epochs` and no event can dominate.

    Returns weights normalised to mean 1 (so `1/sum(w_norm^2)` scaled by n is
    the effective sample size).
    """
    sampler = sampler or CurriculumAmplitudeSampler(CurriculumConfig())
    A = np.asarray(A, dtype=float)
    num = epoch_density(A, epoch, sampler, smear_dex)
    den = np.zeros_like(A)
    for e in range(1, n_epochs + 1):
        den += epoch_density(A, e, sampler, smear_dex)
    den /= n_epochs
    w = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
    if normalise and w.sum() > 0:
        w = w * (len(w) / w.sum())
    return w


def effective_sample_size(w) -> float:
    """Kish ESS, sum(w)^2 / sum(w^2) -- P2a's ESS column."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    return float(s * s / np.sum(w ** 2)) if s > 0 else 0.0


# --------------------------------------------------------------------------- #
# The normalised-window guard
# --------------------------------------------------------------------------- #
def window_excursion(n_true, s_true):
    """Per-event max |normalised target|, over both targets and all channels.

    The quantity the window guard is about: an event is admissible iff this is
    <= WINDOW. Returns shape (B,).
    """
    a = np.abs(np.asarray(n_true, dtype=np.float64)).reshape(len(n_true), -1)
    b = np.abs(np.asarray(s_true, dtype=np.float64)).reshape(len(s_true), -1)
    return np.maximum(a.max(axis=1), b.max(axis=1))


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024.0 ** 2 if sys.platform == "darwin" else 1024.0)


def _mem_available_mb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return float("nan")


class MemMonitor:
    """G1's machine-level memory probe: the fall in MemAvailable while the
    workers run, which counts fork-shared pages once and includes the page
    cache the shard writes consume."""

    def __init__(self, interval=0.25):
        import threading
        self.interval, self._stop = interval, threading.Event()
        self.base = _mem_available_mb()
        self.low = self.base
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            self.low = min(self.low, _mem_available_mb())

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=1.0)
        self.low = min(self.low, _mem_available_mb())

    @property
    def used_mb(self) -> float:
        return self.base - self.low


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _write_shard(path, arrays, meta, compress: bool):
    """Write one shard and return (seconds, bytes). Timed separately from
    generation because G1's pilot never measured it -- a single 40-event
    `savez_compressed` in memory is not the cost of 20 000 events to disk."""
    t0 = time.time()
    fn = np.savez_compressed if compress else np.savez
    fn(path, **arrays, **meta)
    dt = time.time() - t0
    return dt, os.path.getsize(path)


def _worker(args):
    (wid, lo, hi, seed_state, opt) = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg = MissionConfig(Tobs_yr=opt["tobs"])
    n_avg = cfg.Nd if opt["n_avg"] is None else opt["n_avg"]
    priors = SignalPriors(gal_modulation=opt["modulation"])
    rng = np.random.default_rng(seed_state)
    sampler = CurriculumAmplitudeSampler(CurriculumConfig(), rng=rng)

    n_want = hi - lo
    made = 0
    n_rejected = 0
    shard_idx = 0
    t_gen = 0.0
    t_write = 0.0
    bytes_written = 0
    raw_bytes = 0
    excursions = []          # realised max |normalised target| of every ADMITTED event
    rejected_excursions = []
    rejected_amps = []
    buf = {k: [] for k in ("x", "n_true", "s_true", "category", "amplitude", "epoch")}
    buf_n = 0
    shard_records = []

    def flush():
        nonlocal shard_idx, buf, buf_n, t_write, bytes_written, raw_bytes
        if buf_n == 0:
            return
        arrays = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
        path = os.path.join(opt["shard_dir"], f"shard_{wid:02d}_{shard_idx:04d}.npz")
        raw = int(sum(a.nbytes for a in arrays.values()))
        dt, nb = _write_shard(path, arrays, opt["stamp_meta"], opt["compress"])
        t_write += dt
        bytes_written += nb
        raw_bytes += raw
        shard_records.append(dict(path=os.path.basename(path), n=int(buf_n),
                                  seconds=dt, bytes=int(nb), raw_bytes=raw))
        shard_idx += 1
        buf = {k: [] for k in buf}
        buf_n = 0

    while made < n_want:
        b = min(opt["chunk"], n_want - made)
        if opt["validation"]:
            amps = np.ones(b)
            eps_ = np.zeros(b, dtype=np.int64)     # 0 = "not from the curriculum"
        else:
            amps, eps_ = epoch_averaged_amplitudes(b, rng, sampler, opt["n_epochs"])
        t0 = time.time()
        x, n_true, s_true, cat = make_training_batch(
            b, opt["M"], opt["N"], cfg, priors, rng, amplitude_scale=amps,
            channels=tuple(opt["channels"]), n_avg=n_avg, draw=opt["draw"])
        t_gen += time.time() - t0

        exc = window_excursion(n_true, s_true)
        keep = exc <= WINDOW
        if opt["window_policy"] == "flag":
            keep = np.ones(b, dtype=bool)
        elif opt["window_policy"] == "raise" and not keep.all():
            bad = np.where(~keep)[0]
            raise RuntimeError(
                f"worker {wid}: {bad.size} of {b} events exceed the normalised "
                f"window |t| <= {WINDOW}; worst {exc.max():.4f} at "
                f"A = {amps[bad[np.argmax(exc[bad])]]:.3g}. Nothing written for "
                "this shard. (--window-policy raise is the literal reading of "
                "P2a; see this module's docstring.)")
        if not keep.all():
            n_rejected += int((~keep).sum())
            rejected_excursions.extend(exc[~keep].tolist())
            rejected_amps.extend(amps[~keep].tolist())
            frac = n_rejected / max(made + int(keep.sum()), 1)
            if frac > opt["max_reject_frac"] and n_rejected > 5:
                raise RuntimeError(
                    f"worker {wid}: window rejection rate {frac:.3g} exceeds "
                    f"--max-reject-frac {opt['max_reject_frac']:.3g} after "
                    f"{n_rejected} rejections. This is not the ~2e-4 tail P2a "
                    "measured; the prior or the normalisation has moved and the "
                    "pool would be rejection-sampled from a distorted "
                    "amplitude distribution. Build stopped.")

        k = int(keep.sum())
        if k:
            buf["x"].append(x[keep])
            buf["n_true"].append(n_true[keep])
            buf["s_true"].append(s_true[keep])
            buf["category"].append(cat[keep])
            buf["amplitude"].append(amps[keep].astype(np.float32))
            buf["epoch"].append(eps_[keep])
            excursions.extend(exc[keep].tolist())
            buf_n += k
            made += k
        if buf_n >= opt["shard_events"]:
            flush()
    flush()

    return dict(wid=wid, lo=lo, hi=hi, n=made, n_rejected=n_rejected,
                t_gen=t_gen, t_write=t_write, bytes=bytes_written,
                raw_bytes=raw_bytes, rss=peak_rss_mb(), shards=shard_records,
                max_excursion=float(max(excursions)) if excursions else 0.0,
                rejected_excursions=rejected_excursions,
                rejected_amps=rejected_amps)


def _run(jobs, workers):
    if len(jobs) == 1:
        return [_worker(jobs[0])]
    ctx = mp.get_context("fork")
    with ctx.Pool(len(jobs)) as pool:
        return pool.map(_worker, jobs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory for shards + manifest")
    p.add_argument("--n-events", type=int, default=20_000)
    p.add_argument("--n-val", type=int, default=1024,
                   help="validation events, drawn at A = 1 OUTSIDE the pool")
    p.add_argument("--workers", type=int, default=4,
                   help="4 is G1's recommendation (1.5 h, 546 MB); 2 is the "
                        "documented fallback (1.9 h, 264 MB)")
    p.add_argument("--M", type=int, default=1)
    p.add_argument("--n-avg", type=int, default=None, help="default cfg.Nd (126 at 4 yr)")
    p.add_argument("--N", type=int, default=512)
    p.add_argument("--tobs", type=float, default=4.0)
    p.add_argument("--channels", default="AET")
    p.add_argument("--draw", choices=("normal", "gamma"), default="gamma",
                   help="passed explicitly at the call site; the library default "
                        "is 'normal' because the Gamma consumes a different stream")
    p.add_argument("--no-modulation", action="store_true",
                   help="the pool is built WITH modulation; this is for the P4 "
                        "matched-pair variance measurement only")
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--shard-events", type=int, default=500,
                   help="events per shard file. 500 x 18.4 kB = 9.2 MB per shard.")
    p.add_argument("--no-compress", action="store_true",
                   help="np.savez instead of savez_compressed; the manifest "
                        "reports both cost and ratio either way")
    p.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    p.add_argument("--window-policy", choices=("reject", "raise", "flag"),
                   default="reject")
    p.add_argument("--max-reject-frac", type=float, default=2e-2,
                   help="abort if the window rejection rate exceeds this; "
                        "default is 100x the 2e-4 P2a measured at A = 100")
    p.add_argument("--seed", type=int, default=20260920)
    a = p.parse_args(argv)

    cfg = MissionConfig(Tobs_yr=a.tobs)
    n_avg = cfg.Nd if a.n_avg is None else a.n_avg
    Nf = len(build_frequency_grid(cfg))
    os.makedirs(a.out, exist_ok=True)
    shard_dir = os.path.join(a.out, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    stamp_meta = dict(norm_stamp=normalisation_stamp(), M=a.M, N=a.N, n_avg=n_avg,
                      Tobs_yr=a.tobs, channels=np.array(list(a.channels)),
                      draw=a.draw, gal_modulation=not a.no_modulation,
                      window=WINDOW, n_epochs=a.n_epochs)

    print(f"Stage-C pool build -> {a.out}")
    print(f"  {a.n_events} pool events + {a.n_val} validation events at A = 1")
    print(f"  Tobs = {a.tobs} yr, M = {a.M}, n_avg = {n_avg} (Nd = {cfg.Nd}), "
          f"N = {a.N}, channels = {a.channels}, Nf = {Nf}")
    print(f"  draw = {a.draw}, modulation = {not a.no_modulation}, "
          f"workers = {a.workers}, shard = {a.shard_events} events, "
          f"compress = {not a.no_compress}")
    print(f"  stamp = {normalisation_stamp()}, window = |t| <= {WINDOW}, "
          f"policy = {a.window_policy}")
    sys.stdout.flush()

    base = dict(M=a.M, N=a.N, n_avg=a.n_avg, tobs=a.tobs, channels=a.channels,
                draw=a.draw, chunk=a.chunk, modulation=not a.no_modulation,
                shard_dir=shard_dir, shard_events=a.shard_events,
                compress=not a.no_compress, stamp_meta=stamp_meta,
                window_policy=a.window_policy, max_reject_frac=a.max_reject_frac,
                n_epochs=a.n_epochs)

    manifest = dict(config=dict(n_events=a.n_events, n_val=a.n_val, M=a.M, N=a.N,
                                n_avg=n_avg, Tobs_yr=a.tobs, channels=a.channels,
                                draw=a.draw, gal_modulation=not a.no_modulation,
                                workers=a.workers, seed=a.seed,
                                shard_events=a.shard_events,
                                compress=not a.no_compress,
                                n_epochs=a.n_epochs, window=WINDOW,
                                window_policy=a.window_policy,
                                norm_stamp=normalisation_stamp(),
                                curriculum=asdict(CurriculumConfig())),
                    stages={})

    t_start = time.time()
    for stage, n_ev, seed_off, val in (("pool", a.n_events, 0, False),
                                       ("val", a.n_val, 987_654, True)):
        if n_ev <= 0:
            continue
        sd = os.path.join(shard_dir, stage)
        os.makedirs(sd, exist_ok=True)
        opt = dict(base, shard_dir=sd, validation=val)
        if val:
            opt["window_policy"] = "flag"   # the val set is at A = 1; P2a
            # measured nothing within 0.19 normalised units of the ceiling
            # there, and it must not be rejection-sampled under any
            # circumstances -- it is the fixed yardstick.
        nw = max(1, min(a.workers, n_ev))
        bounds = np.linspace(0, n_ev, nw + 1).astype(int)
        states = np.random.SeedSequence(a.seed + seed_off).spawn(nw)
        jobs = [(w, int(bounds[w]), int(bounds[w + 1]), states[w], opt)
                for w in range(nw) if bounds[w + 1] > bounds[w]]
        t0 = time.time()
        with MemMonitor() as mon:
            res = _run(jobs, nw)
        wall = time.time() - t0
        res.sort(key=lambda r: r["wid"])

        n_made = sum(r["n"] for r in res)
        n_rej = sum(r["n_rejected"] for r in res)
        t_gen = sum(r["t_gen"] for r in res)
        t_wr = sum(r["t_write"] for r in res)
        nb = sum(r["bytes"] for r in res)
        raw = sum(r["raw_bytes"] for r in res)
        rss = np.array([r["rss"] for r in res])
        max_exc = max(r["max_excursion"] for r in res)

        print(f"\n[{stage}] {n_made} events in {wall:.1f} s "
              f"({wall/max(n_made,1):.3f} s/event aggregate)")
        print(f"  generate      {t_gen:.1f} CPU-s  ({t_gen/max(n_made,1):.3f} s/event/worker)")
        print(f"  write         {t_wr:.1f} CPU-s  ({100*t_wr/max(t_gen+t_wr,1e-9):.2f}% of "
              f"generate+write), {nb/1e6:.1f} MB on disk")
        print(f"  compression   {raw/1e6:.1f} MB raw -> {nb/1e6:.1f} MB "
              f"({raw/max(nb,1):.3f}x), {raw/1e6/max(t_wr,1e-9):.1f} MB/s raw through the writer")
        print(f"  peak RSS      max {rss.max():.1f} MB/worker, median {np.median(rss):.1f} MB")
        if np.isfinite(mon.base):
            print(f"  MemAvailable  fell {mon.used_mb:.0f} MB "
                  f"({mon.base:.0f} -> {mon.low:.0f} MB)")
        print(f"  window        max |normalised target| = {max_exc:.4f} "
              f"(limit {WINDOW}); {n_rej} events rejected and redrawn "
              f"({n_rej/max(n_made+n_rej,1):.2e})")
        sys.stdout.flush()

        manifest["stages"][stage] = dict(
            n_events=n_made, n_rejected=n_rej, wall_s=wall, gen_cpu_s=t_gen,
            write_cpu_s=t_wr, bytes=int(nb), raw_bytes=int(raw),
            compression_ratio=raw / max(nb, 1),
            write_frac_of_cpu=t_wr / max(t_gen + t_wr, 1e-9),
            peak_rss_mb_max=float(rss.max()), peak_rss_mb_median=float(np.median(rss)),
            mem_available_fall_mb=float(mon.used_mb),
            max_excursion=float(max_exc),
            rejected_amplitudes=[float(v) for r in res for v in r["rejected_amps"]],
            rejected_excursions=[float(v) for r in res for v in r["rejected_excursions"]],
            shards=[s for r in res for s in r["shards"]])

    # the frequency axis the pool's spectra live on -- the Stage-2 bridge's
    # grid, saved once rather than per shard (P2a section 6: identical for
    # every event by construction, and now pinned by test_p2a_pool_grid).
    f_bin = multichannel_f_bin(cfg, a.N, tuple(a.channels))
    np.savez_compressed(os.path.join(a.out, "grid.npz"), f_bin=f_bin,
                        **stamp_meta)
    manifest["total_wall_s"] = time.time() - t_start
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\ntotal {manifest['total_wall_s']/3600:.2f} h; "
          f"manifest written to {os.path.join(a.out, 'manifest.json')}")
    return 0


# Training pool loader


SHARD_ARRAYS = ("x", "n_true", "s_true", "category", "amplitude", "epoch")

#: The validation set's curriculum-epoch sentinel. Not a real epoch: it marks
#: events drawn at A = 1 outside the curriculum, which must never be reweighted.
VALIDATION_EPOCH = 0


class PoolError(RuntimeError):
    """A pool that cannot be used, with the specific reason in the message.

    Separate from NormalisationMismatch because the two call for different
    responses: a stamp mismatch means the pool was built under different
    constants and must be rebuilt, while a PoolError usually means the path is
    wrong or the build did not finish.
    """


@dataclass
class StageCPool:
    """One split (pool or validation) of a Stage-C build, resident in memory."""

    x: np.ndarray            # (n, M*n_ch, N) normalised log10 PSD
    n_true: np.ndarray       # (n, n_ch, N)
    s_true: np.ndarray       # (n, n_ch, N)
    category: np.ndarray     # (n,)
    amplitude: np.ndarray    # (n,) the curriculum amplitude each event was drawn at
    epoch: np.ndarray        # (n,) the schedule component it came from; 0 = validation
    meta: dict               # M, N, n_avg, Tobs_yr, channels, draw, stamp, ...
    path: str

    def __len__(self) -> int:
        return len(self.category)

    @property
    def resident_bytes(self) -> int:
        """Bytes the split actually occupies. The three spectral arrays
        dominate; `category`/`amplitude`/`epoch` are 16 B/event together."""
        return int(sum(getattr(self, k).nbytes for k in SHARD_ARRAYS))

    @property
    def is_validation(self) -> bool:
        return bool(np.all(self.epoch == VALIDATION_EPOCH))

    def weights_for_epoch(self, epoch: int, n_epochs: int = N_EPOCHS,
                          sampler: CurriculumAmplitudeSampler | None = None):
        """P2a's importance weights `w_i = p_e(A_i)/p_bar(A_i)`, mean 1.

        Refuses on the validation split. That is not defensiveness about an
        unlikely mistake: reweighting the validation set is the one thing that
        would make the validation loss incomparable between epochs, and it
        would do so silently, since the loss would still go down.
        """
        if self.is_validation:
            raise PoolError(
                f"{self.path} is the validation split (every event carries "
                f"curriculum epoch {VALIDATION_EPOCH}) and must never be "
                "reweighted -- it is drawn once at A = 1 so that the validation "
                "loss compares the same problem from epoch to epoch.")
        return epoch_weights(self.amplitude.astype(float), epoch,
                             sampler=sampler, n_epochs=n_epochs)


def _shard_paths(split_dir: str) -> list:
    if not os.path.isdir(split_dir):
        raise PoolError(
            f"no such directory: {split_dir}. A finished build has "
            f"shards/pool/ and shards/val/ under the --out directory given to "
            f"lisa_gwb.separator_data.")
    paths = sorted(os.path.join(split_dir, p) for p in os.listdir(split_dir)
                   if p.endswith(".npz"))
    if not paths:
        raise PoolError(f"{split_dir} contains no .npz shards. The build "
                        "either did not run or did not finish.")
    return paths


def load_split(split_dir: str, expect_validation: bool | None = None) -> StageCPool:
    """Load every shard of one split into one resident array per field.

    Concatenation happens once, here. Each shard is opened, copied into the
    output arrays and closed, so peak memory is the final pool plus one shard
    rather than twice the pool -- at 369 MB in a 6 GB machine that margin is
    not academic.
    """
    paths = _shard_paths(split_dir)

    # Pass 1: stamps, configuration agreement, and the event count per shard.
    # Only `category` is decompressed here -- npz members are separate zip
    # entries, so this costs 8 bytes per event rather than a whole shard.
    meta, counts = None, []
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            check_normalisation_stamp(d, p)
            missing = [k for k in SHARD_ARRAYS if k not in d]
            if missing:
                raise PoolError(
                    f"{p} is missing {missing}. A shard written by "
                    f"lisa_gwb.separator_data carries {list(SHARD_ARRAYS)}; this "
                    "file was written by something else or by an older build.")
            counts.append(len(d["category"]))
            this = {k: d[k].tolist() if d[k].ndim else d[k].item()
                    for k in ("M", "N", "n_avg", "Tobs_yr", "channels", "draw",
                              "gal_modulation", "norm_stamp") if k in d}
            if meta is None:
                meta = this
            elif this != meta:
                raise PoolError(
                    f"{p} disagrees with {paths[0]} on the build configuration: "
                    f"{this} against {meta}. Shards from two different builds "
                    "are in the same directory.")

    # Pass 2: allocate the final arrays once and fill them in place, so peak
    # memory is the pool plus one shard rather than two copies of the pool.
    total = int(sum(counts))
    with np.load(paths[0], allow_pickle=False) as d0:
        out = {k: np.empty((total, *d0[k].shape[1:]), dtype=d0[k].dtype)
               for k in SHARD_ARRAYS}
    at = 0
    for p, n in zip(paths, counts):
        with np.load(p, allow_pickle=False) as d:
            for k in SHARD_ARRAYS:
                out[k][at:at + n] = d[k]
        at += n
    pool = StageCPool(meta=meta or {}, path=split_dir, **out)

    if expect_validation is True and not pool.is_validation:
        raise PoolError(
            f"{split_dir} was loaded as the validation split but not every "
            f"event carries curriculum epoch {VALIDATION_EPOCH} "
            f"({int(np.sum(pool.epoch != VALIDATION_EPOCH))} of {len(pool)} do "
            "not). The validation set must be drawn at A = 1 outside the pool.")
    if expect_validation is False and pool.is_validation:
        raise PoolError(
            f"{split_dir} was loaded as the training pool but every event "
            f"carries the validation sentinel epoch {VALIDATION_EPOCH}. The "
            "pool and validation directories are probably swapped.")
    return pool


def load_pool(root: str):
    """Load both splits of a build. Returns (train_pool, val_pool, manifest).

    `root` is the --out directory given to lisa_gwb.separator_data: it holds
    shards/pool/, shards/val/, grid.npz and manifest.json.
    """
    if not os.path.isdir(root):
        raise PoolError(f"no such pool directory: {root}")
    man_path = os.path.join(root, "manifest.json")
    manifest = None
    if os.path.exists(man_path):
        with open(man_path) as fh:
            manifest = json.load(fh)
    else:
        raise PoolError(
            f"{man_path} is missing. The builder writes it last, so its "
            "absence means the build did not finish and the shards on disk are "
            "an incomplete pool.")
    check_normalisation_stamp(manifest.get("config", {}), man_path)
    train = load_split(os.path.join(root, "shards", "pool"), expect_validation=False)
    val = load_split(os.path.join(root, "shards", "val"), expect_validation=True)
    return train, val, manifest


def batch_ess(weights) -> float:
    """Kish effective sample size of one batch's weights.

    The number P5 section 3 gates on. For a minibatch it is the count of
    equally-weighted samples that would carry the same information as these
    weighted ones, so an ESS of 1.7 at batch 64 means the gradient is an
    average over not much more than a single event however large the batch is.
    """
    return effective_sample_size(np.asarray(weights, dtype=float))


def normalise_batch_weights(w):
    """Scale a batch's weights to mean exactly 1 -- the P5 section 2 reading.

    Returns the input unchanged in the degenerate all-zero case, which cannot
    occur for weights from `epoch_weights` (the epoch-averaged proposal covers
    every epoch's support) but would otherwise divide by zero.
    """
    w = np.asarray(w, dtype=np.float64)
    m = w.mean()
    return w if m <= 0 else w / m


if __name__ == "__main__":
    sys.exit(main())
