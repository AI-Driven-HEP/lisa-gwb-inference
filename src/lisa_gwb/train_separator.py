"""Train and validate the spectral separator.

The production path loads a fixed simulation pool and applies the curriculum
weights each epoch. Validation uses the same physical-amplitude simulations
throughout training. Checkpoints store both model weights and the state
needed to resume an interrupted run.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import torch

from lisa_gwb.curriculum import CurriculumAmplitudeSampler, CurriculumConfig
from lisa_gwb.losses import MSMHLoss
from lisa_gwb.separator import (
    MSMHAutoencoder,
    check_no_frozen_bins,
    check_responsive_bins,
    detectable_bins,
    frozen_bin_report,
    responsiveness_report,
    save_checkpoint,
)
from lisa_gwb.separator_data import N_EPOCHS as POOL_N_EPOCHS
from lisa_gwb.separator_data import PoolError, batch_ess, load_pool, normalise_batch_weights
from lisa_gwb.simulation import (
    MissionConfig,
    SignalPriors,
    make_training_batch,
    multichannel_sigma_frac,
    normalisation_stamp,
)


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
_GEN = {}


def _init_gen(M, N, channels, n_avg):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _GEN.update(M=M, N=N, channels=channels, n_avg=n_avg,
                cfg=MissionConfig(), priors=SignalPriors())


def _gen_chunk(args):
    """One worker's slice. `amps` is None for the validation set (A = 1) or an
    array of per-event curriculum amplitudes for a training refresh."""
    seed, amps, n = args
    rng = np.random.default_rng(seed)
    g = _GEN
    xs, ns, ss, cs = [], [], [], []
    for i in range(n):
        A = 1.0 if amps is None else float(amps[i])
        x, nt, st, c = make_training_batch(
            1, g["M"], g["N"], g["cfg"], g["priors"], rng,
            channels=g["channels"], n_avg=g["n_avg"], amplitude_scale=A)
        xs.append(x[0]); ns.append(nt[0]); ss.append(st[0]); cs.append(c[0])
    return np.stack(xs), np.stack(ns), np.stack(ss), np.array(cs)


def generate(pool, n, seed, n_proc, amps=None):
    """n events across the pool. Returns torch tensors (x, n_true, s_true, cat)."""
    counts = [n // n_proc + (1 if i < n % n_proc else 0) for i in range(n_proc)]
    offs = np.cumsum([0] + counts)
    args = [(seed + i, None if amps is None else amps[offs[i]:offs[i + 1]], counts[i])
            for i in range(n_proc)]
    parts = pool.map(_gen_chunk, args)
    cat = np.concatenate([p[3] for p in parts])
    out = [torch.from_numpy(np.concatenate([p[k] for p in parts])) for k in range(3)]
    return out[0], out[1], out[2], torch.from_numpy(cat).long()


# --------------------------------------------------------------------------- #
# P5: CPU thread control
# --------------------------------------------------------------------------- #
#: Conservative on purpose. Three seeds are expected to run concurrently on a
#: shared node, and PyTorch's default is to take every visible core, so three
#: unconfigured jobs on a 64-core node ask for 192 threads and spend their time
#: in the scheduler. 2 is what this machine's torch already defaults to and what
#: every measurement in the notebook was taken at.
DEFAULT_THREADS = 2

#: Curriculum epochs the pool's epoch-averaged amplitude proposal spans. Must
#: equal separator_data.N_EPOCHS: the weights are p_e/p_bar and p_bar is
#: the average over exactly these epochs, so a mismatch reweights toward a
#: schedule the pool was never drawn from.
POOL_SCHEDULE_EPOCHS = POOL_N_EPOCHS

_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")


def apply_denormal_mode(flush: bool) -> bool:
    """Flush subnormal floats to zero, and say whether it took.

    P6 section 5: x86 handles subnormals -- magnitudes below 2^-126 =
    1.18e-38 -- in microcode, at tens to hundreds of cycles per operation
    instead of one. Two of the five rates in the P6 learning-rate scan drove
    part of their weights into that range partway through training and then
    paid for every operation touching them: measured on the final weights at a
    FIXED FLOP count, the forward+backward pass cost 1.05x, 1.22x, 3.73x, 4.63x
    and 4.69x more without flushing, and exactly the same (317-346 ms) with it.
    Resuming a degraded checkpoint with this on cut two epochs from 462.3 s to
    171.3 s, 2.93x, and left the validation loss BIT-IDENTICAL -- the values
    being flushed are below 1.18e-38 in a network whose normalised targets live
    in [-0.900, +0.861], so they are cost without information.

    WHERE THIS HAS TO BE CALLED, and why it is a function rather than a line.
    FTZ/DAZ is a bit in MXCSR and MXCSR is per THREAD. torch's intra-op pool
    inherits it only if the pool is created AFTER the bit is set, so calling
    this once the pool exists sets the main thread alone and buys 1.05x rather
    than 4.7x. It is therefore called at the top of `train`, before the thread
    limit is applied and before any tensor is touched.

    Returns False if the platform refuses the mode, which is not fatal: the run
    is then numerically identical and merely slower, and the caller says so.
    """
    if not flush:
        return False
    return bool(torch.set_flush_denormal(True))


def apply_thread_limit(threads: int) -> dict:
    """Set the torch and OpenMP/BLAS thread counts together, and report them.

    CAVEAT, stated because it decides where the flag has to be set. Exporting
    OMP_NUM_THREADS from Python is only reliably honoured by runtimes that have
    not started yet -- OpenMP reads it at first use, which for torch is at
    import. `torch.set_num_threads` does take effect at runtime and is what
    actually controls the intra-op pool in this process; the environment
    variables here are for the generation workers (which are forked after this
    runs) and for any BLAS that initialises later. `run_training.sh` exports
    them BEFORE starting Python, which is the path that is reliable end to end,
    and is why the handoff goes through the shell script rather than calling
    this module directly.
    """
    if threads < 1:
        raise ValueError(f"--threads must be >= 1, got {threads}")
    for v in _THREAD_ENV:
        os.environ[v] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # already fixed by earlier parallel work in this process; not fatal
        pass
    return dict(requested=threads, torch_num_threads=torch.get_num_threads(),
                torch_interop=torch.get_num_interop_threads(),
                omp=os.environ.get("OMP_NUM_THREADS"))


# --------------------------------------------------------------------------- #
# P5: RNG capture and restore, for a resume that is a genuine continuation
# --------------------------------------------------------------------------- #
def capture_rng() -> dict:
    """Every stream a training step can consume, in one dict.

    Torch's global generator is included even though nothing in this loop draws
    from it today (no dropout, and BatchNorm keeps running statistics rather
    than sampling). That is exactly why it belongs here: the cost is a few
    hundred bytes and the failure it prevents -- a future stochastic layer
    silently restarting its stream on every resume -- is one that no loss curve
    would reveal.
    """
    return dict(python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state())


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _adopt_pool_config(a, pool) -> None:
    """In --data pool the geometry is a property of the DATA, not a choice.

    M, N, the channel list and n_avg are baked into every shard, so a
    command-line value that disagrees is not an alternative configuration, it
    is a mistake -- and one that would otherwise surface as a shape error deep
    inside the model. They are taken from the pool and the substitution is
    printed. Everything else (width, depth, lr, batch) stays under the
    command line's control, because those are genuinely free.
    """
    meta = pool.meta
    ch = ",".join(meta["channels"]) if isinstance(meta.get("channels"), list) \
        else str(meta.get("channels"))
    adopted = dict(M=int(meta["M"]), N=int(meta["N"]), channels=ch,
                   n_avg=int(meta["n_avg"]))
    changed = {k: (getattr(a, k), v) for k, v in adopted.items()
               if getattr(a, k) != v}
    for k, v in adopted.items():
        setattr(a, k, v)
    if changed:
        print("pool geometry adopted from the shards (command-line values for "
              "these are ignored in --data pool):")
        for k, (was, now) in changed.items():
            print(f"    {k}: {was} -> {now}")


def _as_tensors(split):
    """Views onto the resident pool -- torch.from_numpy shares the buffer, so
    this does not double the 369 MB."""
    return (torch.from_numpy(split.x), torch.from_numpy(split.n_true),
            torch.from_numpy(split.s_true),
            torch.from_numpy(split.category).long())


def batch_is_empty(bw) -> bool:
    """True if this batch's importance weights carry no mass at all.

    P5 section 3 measured how often this happens: 22.1% of batches in epoch 1
    at batch 32, 3.9% at batch 64. The epoch-1 curriculum is a point mass at
    A = 100 and only 3.8% of the pool sits there, so a whole minibatch can
    easily contain no event the epoch's law gives any weight to.
    """
    return bw is not None and float(np.sum(np.asarray(bw, dtype=float))) <= 0.0


def run_batch(model, opt, loss_fn, xt, nt, st, ct, idx, bw, clip=1.0) -> bool:
    """Take one optimiser step for this batch -- or refuse to take one.

    Returns True if a step was taken and False if the batch was skipped
    because its importance weights carry no mass (P6 item 1).

    The guard lives HERE, inside the function that owns the step, rather than
    in the training loop, so that the test asserting bit-identity exercises the
    same code the run does instead of a restatement of it.

    Why skipping is the correct action and not merely an optimisation: with
    every weight zero the loss is exactly zero and so is the gradient, but
    `opt.step()` is not the identity map at zero gradient. Adam applies its
    `weight_decay = 5e-6` term and, more importantly, its accumulated first and
    second moments, so the parameters keep moving in the direction of the LAST
    batch that did have data. Those steps shrink the network on no evidence;
    they are worse than a skipped step, and P5 section 3 measured 22.1% of
    epoch-1 batches taking them at batch 32.
    """
    if batch_is_empty(bw):
        return False
    sw = (None if bw is None
          else torch.as_tensor(normalise_batch_weights(bw), dtype=torch.float32))
    opt.zero_grad()
    l, _ = loss_fn(model, model(xt[idx]), nt[idx], st[idx], ct[idx],
                   sample_weight=sw)
    l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return True


def train(a):
    # Before apply_thread_limit and before any tensor work -- see the docstring.
    flush_denormal = apply_denormal_mode(getattr(a, "flush_denormal", False))
    if getattr(a, "flush_denormal", False) and not flush_denormal:
        print("WARNING: --flush-denormal requested but this platform refused "
              "it. The run is numerically unchanged and may be up to 4.7x "
              "slower if the weights drift subnormal (P6 section 5).",
              flush=True)
    thread_info = apply_thread_limit(a.threads)
    channels = tuple(a.channels.split(","))

    train_pool = val_pool = manifest = None
    if a.data == "pool":
        if not a.pool_dir:
            raise PoolError("--data pool needs --pool_dir, the --out directory "
                            "that separator_data.py wrote.")
        t_load = time.time()
        train_pool, val_pool, manifest = load_pool(a.pool_dir)
        _adopt_pool_config(a, train_pool)
        channels = tuple(a.channels.split(","))
        resident = train_pool.resident_bytes + val_pool.resident_bytes
        print(f"pool: {len(train_pool)} train + {len(val_pool)} validation events "
              f"from {a.pool_dir} in {time.time()-t_load:.1f}s, "
              f"{resident/1e6:.1f} MB resident "
              f"({train_pool.resident_bytes/1e6:.1f} + "
              f"{val_pool.resident_bytes/1e6:.1f})")
        print(f"      stamp {train_pool.meta['norm_stamp']}, draw="
              f"{train_pool.meta['draw']}, modulation="
              f"{train_pool.meta['gal_modulation']}")

    n_ch = len(channels)
    cfg = MissionConfig()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed % (2 ** 32))

    sigma_frac = multichannel_sigma_frac(cfg, a.N, channels, a.M, a.n_avg)
    model = MSMHAutoencoder(M=a.M, N=a.N, n_ch=n_ch, n_layers=a.n_layers,
                            base_channels=a.base_channels)
    n_param = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=5e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = MSMHLoss(sigma_frac=None if a.no_detectability else sigma_frac)
    curric = CurriculumAmplitudeSampler(
        CurriculumConfig(E_curr=a.curriculum_epochs),
        rng=np.random.default_rng(a.seed + 555))

    # Dedicated streams, so that each can be saved and restored independently
    # and so that adding one does not shift another (which is what would make a
    # resume look like a continuation while quietly training on different data).
    g = torch.Generator().manual_seed(a.seed)                 # batch order
    sub_rng = np.random.default_rng(a.seed + 4242)            # pool subsample

    print(f"data={a.data} channels={channels} M={a.M} N={a.N} n_avg={a.n_avg} "
          f"layers={a.n_layers} width={a.base_channels} lr={a.lr} "
          f"batch={a.batch} params={n_param}")
    print(f"curriculum={'OFF' if a.no_curriculum else f'ON (E_curr={a.curriculum_epochs})'}  "
          f"detectability weighting={'OFF' if a.no_detectability else 'ON'}  "
          f"threads={thread_info['torch_num_threads']}  "
          f"denormals={'FLUSHED' if flush_denormal else 'kept'}")
    if a.data == "pool":
        print(f"importance weights={'OFF (uniform)' if a.pool_unweighted else 'ON'}"
              f" normalised per batch to mean 1")

    gen_pool = (contextlib.nullcontext(None) if a.data == "pool"
                else Pool(a.n_proc, initializer=_init_gen,
                          initargs=(a.M, a.N, channels, a.n_avg)))
    with gen_pool as pool:
        t0 = time.time()
        if a.data == "pool":
            xv, nv, sv, cv = _as_tensors(val_pool)
        else:
            xv, nv, sv, cv = generate(pool, a.n_val, a.seed + 900000, a.n_proc)
            print(f"validation set: {a.n_val} events at A=1.0 in "
                  f"{time.time()-t0:.0f}s", flush=True)

        def evaluate():
            model.eval()
            tot = spec = 0.0
            with torch.no_grad():
                for i in range(0, len(xv), 32):
                    sl = slice(i, i + 32)
                    l, comp = loss_fn(model, model(xv[sl]), nv[sl], sv[sl], cv[sl])
                    tot += float(l) * len(xv[sl])
                    spec += comp["loss_spectral"] * len(xv[sl])
            return tot / len(xv), spec / len(xv)

        best, best_ep, bad, hist = np.inf, -1, 0, []
        xt = nt = st = ct = None
        refresh_ep, refresh_amps = None, None
        start_ep = 1

        # ---- resume -----------------------------------------------------
        last_path = os.path.join(a.ckpt_dir, "last.pt") if a.ckpt_dir else None
        if last_path and os.path.exists(last_path):
            ck = torch.load(last_path, map_location="cpu", weights_only=False)
            _check_resume_compatible(ck, a, n_param)
            model.load_state_dict(ck["state_dict"])
            opt.load_state_dict(ck["opt_state"])
            sched.load_state_dict(ck["sched_state"])
            restore_rng(ck["rng_state"])
            g.set_state(ck["batch_gen_state"])
            sub_rng.bit_generator.state = ck["sub_rng_state"]
            curric.rng.bit_generator.state = ck["curric_rng_state"]
            best, best_ep, bad = ck["best"], ck["best_ep"], ck["bad"]
            # Pre-P6 checkpoints carry (epoch, val_total, val_spectral); the
            # skipped-batch count is appended as 0 rather than left ragged, so
            # a run started before the guard existed still resumes.
            hist = [tuple(h) + (0,) * (4 - len(tuple(h))) for h in ck["hist"]]
            start_ep = int(ck["epoch"]) + 1
            refresh_ep = ck["refresh_ep"]
            refresh_amps = (None if ck["refresh_amps"] is None
                            else np.asarray(ck["refresh_amps"]))
            print(f"RESUMED from {last_path} at epoch {ck['epoch']}; "
                  f"continuing at {start_ep} (best {best:.4f}@{best_ep}, "
                  f"bad {bad})", flush=True)
            if start_ep > a.epochs:
                print(f"nothing to do: checkpoint is at epoch {ck['epoch']} of "
                      f"{a.epochs}.", flush=True)

        for ep in range(start_ep, a.epochs + 1):
            if a.data == "pool":
                # Uniform subsample of the pool, reweighted in the loss. The
                # subsample is uniform on purpose: it is the weights that carry
                # the curriculum, and drawing the subsample by weight instead
                # would hide the per-batch ESS that P5 section 3 measures.
                sel = sub_rng.choice(len(train_pool), size=a.n_train,
                                     replace=len(train_pool) < a.n_train)
                if a.pool_unweighted:
                    w_ep = np.ones(len(sel))
                else:
                    w_ep = train_pool.weights_for_epoch(
                        ep, n_epochs=a.pool_schedule_epochs)[sel]
                xt, nt, st, ct = _as_tensors(train_pool)
                order = torch.randperm(len(sel), generator=g)
                batches = [sel[order[i:i + a.batch].numpy()]
                           for i in range(0, len(sel), a.batch)]
                wts = [w_ep[order[i:i + a.batch].numpy()]
                       for i in range(0, len(sel), a.batch)]
            else:
                need = (xt is None) or ((ep - 1) % a.refresh_every == 0)
                if need:
                    # On a resume mid-refresh-window the data must be the data
                    # the interrupted run had, not a fresh draw for THIS epoch,
                    # or the resumed run trains on a different set. The epoch
                    # the set belongs to and its amplitudes are both in the
                    # checkpoint, and generation is deterministic given them.
                    if refresh_amps is not None and refresh_ep is not None \
                            and (ep - 1) % a.refresh_every != 0:
                        amps, src_ep = refresh_amps, refresh_ep
                    else:
                        amps = (np.ones(a.n_train) if a.no_curriculum
                                else curric.sample(ep, size=a.n_train))
                        src_ep = ep
                    tg = time.time()
                    xt, nt, st, ct = generate(pool, a.n_train,
                                              a.seed + 1000 * src_ep, a.n_proc,
                                              amps=amps)
                    refresh_ep, refresh_amps = src_ep, amps
                    print(f"  [ep {ep}] refreshed {a.n_train} events (for epoch "
                          f"{src_ep}), A median {np.median(amps):.4g} range "
                          f"[{amps.min():.3g}, {amps.max():.3g}] in "
                          f"{time.time()-tg:.0f}s", flush=True)
                order = torch.randperm(len(xt), generator=g)
                batches = [order[i:i + a.batch]
                           for i in range(0, len(order), a.batch)]
                wts = [None] * len(batches)

            model.train()
            ess_sum, ess_n, n_empty = 0.0, 0, 0
            for bidx, bw in zip(batches, wts):
                if bw is not None and a.pool_unweighted:
                    bw = None                  # the ablation: no weights at all
                if bw is not None:
                    ess_sum += batch_ess(bw)   # zero-weight batches count as 0
                    ess_n += 1
                idx = torch.as_tensor(bidx, dtype=torch.long)
                if not run_batch(model, opt, loss_fn, xt, nt, st, ct, idx, bw):
                    n_empty += 1
            if n_empty == len(batches) and n_empty:
                # Nothing in this epoch moved the network. The cosine schedule
                # still advances -- see the note on sched.step() below -- but
                # this is worth saying out loud rather than inferring from a
                # flat loss curve.
                print(f"  [ep {ep}] WARNING: every one of {n_empty} batches had "
                      "zero total importance weight; no step was taken this "
                      "epoch.", flush=True)
            # The scheduler is stepped once per EPOCH, not once per batch, and
            # it is deliberately left outside the guard: T_max is keyed to
            # --epochs, and P5 section 4 rests on the learning rate being a
            # function of the epoch index alone, so that a resumed job sees the
            # rate it would have seen had it never stopped. Making it depend on
            # how many batches happened to carry weight would make the schedule
            # data-dependent and the resume no longer reproducible from the
            # epoch number.
            sched.step()

            vt, vs = evaluate()
            hist.append((ep, vt, vs, n_empty))
            if vs < best - 1e-5:
                best, best_ep, bad = vs, ep, 0
                save_checkpoint(a.out, model, M=a.M, N=a.N, n_ch=n_ch,
                                n_layers=a.n_layers, base_channels=a.base_channels,
                                channels=list(channels), n_avg=a.n_avg, epoch=ep,
                                val_spectral=vs, seed=a.seed,
                                detectability_weighting=not a.no_detectability,
                                curriculum=not a.no_curriculum)
            else:
                bad += 1

            if last_path:
                save_checkpoint(
                    last_path, model,
                    epoch=ep, opt_state=opt.state_dict(),
                    sched_state=sched.state_dict(), rng_state=capture_rng(),
                    batch_gen_state=g.get_state(),
                    sub_rng_state=sub_rng.bit_generator.state,
                    curric_rng_state=curric.rng.bit_generator.state,
                    best=best, best_ep=best_ep, bad=bad, hist=hist,
                    refresh_ep=refresh_ep,
                    refresh_amps=(None if refresh_amps is None
                                  else np.asarray(refresh_amps)),
                    config=vars(a), n_param=n_param)

            if ep % 10 == 0 or bad == 0:
                extra = (f" ess/batch {ess_sum/max(ess_n,1):.1f}"
                         f" skipped {n_empty}/{len(batches)}" if ess_n else "")
                print(f"ep {ep:4d} val_total {vt:.4f} val_spectral {vs:.4f} "
                      f"best {best:.4f}@{best_ep} ({time.time()-t0:.0f}s)"
                      f"{extra}", flush=True)
            if bad >= a.patience:
                print(f"stopped at {ep}: no improvement for {a.patience}", flush=True)
                break
            if a.stop_after and ep < a.epochs \
                    and (ep - start_ep + 1) >= a.stop_after:
                # A run-length cap for this PROCESS, not a change to the run.
                # It is deliberately not part of the schedule: T_max stays at
                # --epochs, so a job stopped by a walltime and resubmitted sees
                # exactly the learning rate it would have seen had it never
                # stopped. The guards are skipped because they judge a finished
                # network and this one is not finished.
                #
                # `ep < a.epochs` is the whole of P7: the cap says where to
                # PAUSE, and there is nothing to pause before once the run has
                # reached its horizon. Without it a cap that divides --epochs
                # exactly -- 100 into 200, which is the runbook's own block (b)
                # -- lands on the final epoch, and the run exits by the resume
                # route telling the reader to re-run a command that has nothing
                # left to do, with no BEST line and the guards never run. The
                # checkpoints were fine; the only record that the signal head
                # is alive everywhere was missing, and that record is the one
                # thing no loss threshold can replace (8e).
                print(f"--stop_after {a.stop_after}: stopping this process at "
                      f"epoch {ep} of {a.epochs}. Re-run the same command to "
                      f"resume from {last_path}.", flush=True)
                return

        if best_ep < 0:
            print("no epoch ran; nothing to check.", flush=True)
            return

        # ---- guards, on the best checkpoint ---------------------------------
        print(f"\nBEST val_spectral {best:.4f} at epoch {best_ep}")
        ck = torch.load(a.out, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            s_hat_val = torch.cat([model(xv[i:i + 32])["s_hat"]
                                   for i in range(0, len(xv), 32)]).numpy()

        det = detectable_bins(nv.numpy(), sv.numpy(), sigma_frac)
        rep = responsiveness_report(s_hat_val.reshape(len(xv), -1),
                                    sv.numpy().reshape(len(xv), -1),
                                    detectable=det.reshape(-1))
        print(f"responsiveness: {rep['n_checked']}/{rep['n_bins']} bins checked "
              f"(detectable), median slope {rep['median_slope_checked']:.3f}, "
              f"{rep['flagged_idx'].size} flagged below {rep['slope_threshold']}")

        # 8e probe: replicates of ONE event, which is a different axis entirely
        rng = np.random.default_rng(a.seed + 77)
        priors = SignalPriors(gal_modulation=(a.data == "pool"))
        from lisa_gwb.simulation import sample_event_params
        ev = sample_event_params(rng, priors, 3, cfg)
        reps = []
        for _ in range(a.n_replicates):
            xr, _, _, _ = make_training_batch(1, a.M, a.N, cfg, priors, rng,
                                              channels=channels, n_avg=a.n_avg,
                                              fixed_params=ev,
                                              draw="gamma" if a.data == "pool" else "normal")
            with torch.no_grad():
                reps.append(model(torch.from_numpy(xr))["s_hat"].numpy().reshape(-1))
        fr = frozen_bin_report(np.stack(reps))
        print(f"frozen bins: {fr['frac_exactly_zero']*100:.2f}% exactly zero, "
              f"{fr['frac_near_frozen']*100:.2f}% near-frozen")

        np.savetxt(a.out.replace(".pt", "_hist.txt"), np.array(hist),
                   header="epoch val_total val_spectral skipped_zero_weight_batches")
        check_no_frozen_bins(np.stack(reps), allow=a.allow_bad_bins, n_ch=n_ch)
        check_responsive_bins(s_hat_val.reshape(len(xv), -1),
                              sv.numpy().reshape(len(xv), -1),
                              detectable=det.reshape(-1), allow=a.allow_bad_bins)
        print("guards clean.")


#: Fields of the saved config that must match for a resume to be a
#: continuation rather than a different run wearing the same output directory.
RESUME_CRITICAL = ("M", "N", "channels", "n_avg", "n_layers", "base_channels",
                   "batch", "lr", "epochs", "seed", "data", "n_train",
                   "no_curriculum", "no_detectability", "pool_unweighted",
                   "refresh_every", "curriculum_epochs", "pool_schedule_epochs",
                   # P6 section 6. FTZ/DAZ changes the arithmetic, so a run
                   # resumed under a different setting is half one thing and
                   # half another, and P5's bit-exact resume guarantee no
                   # longer holds across the boundary. Checkpoints written
                   # before the flag existed have no such key and are skipped
                   # by the `k in old` test, so the scan's runs still resume.
                   "flush_denormal")


def _check_resume_compatible(ck, a, n_param) -> None:
    """Refuse a resume whose configuration has moved, naming every field.

    The failure this prevents is the quiet one: a checkpoint from a different
    seed or learning rate loads without complaint, training continues, and the
    run is neither the old one nor a clean new one. It is also the failure most
    likely on a cluster, where the same output directory gets reused.
    """
    if ck.get("norm_stamp") != normalisation_stamp():
        raise RuntimeError(
            f"checkpoint was written under normalisation "
            f"{ck.get('norm_stamp')!r} but the constants in force are "
            f"{normalisation_stamp()!r}. Its weights mean a different physical "
            "spectrum; start a new run in a new --out directory.")
    old = ck.get("config") or {}
    diff = [(k, old.get(k), getattr(a, k)) for k in RESUME_CRITICAL
            if k in old and old.get(k) != getattr(a, k, None)]
    if diff:
        lines = "\n".join(f"    {k}: checkpoint {w!r} -> now {n!r}"
                           for k, w, n in diff)
        raise RuntimeError(
            "refusing to resume: the configuration has changed since the "
            f"checkpoint was written.\n{lines}\n"
            "Resuming would produce a run that is neither the old one nor a "
            "clean new one. Use a different --ckpt_dir for a new "
            "configuration, or restore the arguments above.")
    if "n_param" in ck and ck["n_param"] != n_param:
        raise RuntimeError(
            f"checkpoint holds {ck['n_param']} parameters but this model has "
            f"{n_param}. The architecture has changed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=12)
    p.add_argument("--N", type=int, default=1005)
    p.add_argument("--channels", type=str, default="A,E,T")
    p.add_argument("--n_avg", type=int, default=1)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--base_channels", type=int, default=64)
    # P6 item 2: 64, raised from 32. Free in wall clock -- P5 section 7
    # measured 58.6 s/epoch at batch 32 against 58.9 s at batch 64, twice the
    # step cost against half the steps -- and it roughly doubles the per-batch
    # effective sample size, taking the fraction of epoch-1 batches whose
    # importance weights are ALL zero from 22.1% to 3.9% (P5 section 3).
    p.add_argument("--batch", type=int, default=64,
                   help="minibatch size (default 64). Raised from 32 in P6: at "
                        "the pool's importance weights a larger batch roughly "
                        "doubles the per-batch effective sample size and "
                        "almost removes all-zero-weight batches, and it costs "
                        "nothing per epoch because the step count halves.")
    # NOT the paper's 5e-5 and NOT 4d's 1e-3. Both were measured wrong for this
    # width: 4d found 1e-3 already too large at base_channels=64 and OF1 found
    # 1e-2 frankly diverging, while CAL1's scan settled 3e-4 at width 32. The
    # rate must be re-scanned at the production width before a long run -- this
    # default is CAL1's, carried over as a starting point, not a measurement.
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--n_train", type=int, default=1024)
    p.add_argument("--n_val", type=int, default=128)
    p.add_argument("--refresh_every", type=int, default=5)
    p.add_argument("--curriculum_epochs", type=int, default=100)
    p.add_argument("--n_replicates", type=int, default=20)
    p.add_argument("--n_proc", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--out", type=str, default="stage1_model.pt",
                   help="path of the BEST-validation checkpoint")
    # ---- P5: data source -------------------------------------------------
    p.add_argument("--data", choices=("regen", "pool"), default="regen",
                   help="regen (default) regenerates the training set every "
                        "--refresh_every epochs, which is what every number "
                        "measured before P5 was taken under. pool reads the "
                        "Stage-C pool built by separator_data.py and "
                        "reweights it per epoch; it needs --pool_dir, and it "
                        "takes M/N/channels/n_avg from the shards.")
    p.add_argument("--pool_dir", type=str, default=None,
                   help="the --out directory separator_data.py wrote "
                        "(holds shards/, grid.npz, manifest.json)")
    p.add_argument("--pool_unweighted", action="store_true",
                   help="--data pool with every importance weight set to 1: "
                        "the ablation that isolates what the reweighting does")
    p.add_argument("--pool_schedule_epochs", type=int, default=POOL_SCHEDULE_EPOCHS,
                   help="number of curriculum epochs the pool's epoch-averaged "
                        "amplitude proposal was built from. MUST match the "
                        "builder; changing it silently changes every weight.")
    # ---- P5: checkpoint / resume ----------------------------------------
    p.add_argument("--stop_after", type=int, default=0,
                   help="stop THIS process after N epochs and exit, leaving a "
                        "resumable checkpoint (0 = run to --epochs). For a "
                        "walltime-limited queue: it does not change the "
                        "cosine schedule, which stays keyed to --epochs, so "
                        "a resubmitted job continues rather than restarts.")
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="directory for the every-epoch resumable checkpoint "
                        "(last.pt). If it already holds one, training RESUMES "
                        "from it automatically. Omit for the pre-P5 behaviour "
                        "of only writing the best-validation checkpoint.")
    # ---- P5: CPU threads -------------------------------------------------
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                   help=f"CPU threads for torch and OMP/MKL/OpenBLAS "
                        f"(default {DEFAULT_THREADS}). Deliberately NOT all "
                        "visible cores: several seeds are expected to share a "
                        "node, and PyTorch's default of one thread per core "
                        "makes them contend rather than run. Raise it only if "
                        "this job has the node to itself.")
    p.add_argument("--no_curriculum", action="store_true",
                   help="train at the physical amplitude throughout (the CAL1 setting)")
    p.add_argument("--no_detectability", action="store_true",
                   help="pre-P1 loss: regress on every bin, detectable or not")
    p.add_argument("--allow_bad_bins", action="store_true",
                   help="report the guards but do not raise on them")
    p.add_argument("--flush-denormal", "--flush_denormal", dest="flush_denormal",
                   action="store_true",
                   help="flush subnormal floats to zero (FTZ/DAZ). Adopted for "
                        "the production run after P6 section 5: up to 4.7x "
                        "faster on a run whose weights have drifted subnormal, "
                        "bit-identical validation loss. The P6 scan itself ran "
                        "WITHOUT it.")
    p.add_argument("--smoke_test", action="store_true")
    a = p.parse_args()
    if a.smoke_test:
        # The tiny test config, plus the smallest val set the responsiveness
        # guard will accept (RESPONSIVENESS_MIN_EVENTS).
        a.M, a.N, a.channels, a.n_layers, a.base_channels = 4, 64, "A", 2, 8
        a.batch, a.epochs, a.n_train, a.n_val = 2, 2, 16, 32
        a.refresh_every, a.n_replicates, a.allow_bad_bins = 1, 4, True
        a.out = os.path.join(tempfile.gettempdir(), "stage1_smoke.pt")
    train(a)
