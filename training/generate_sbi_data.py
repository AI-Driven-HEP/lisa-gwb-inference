"""Generate the simulations used to train and develop the SBI models."""

import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from lisa_gwb.simulation import (
    MissionConfig,
    SignalPriors,
    draw_sbi_parameters,
    make_training_batch,
    sample_gal_modulation,
    sbi_parameters_to_event,
)

SEPARATOR_SEEDS = (20260922, 20260923, 20260924)
TRAINING_SEED = 20260925
CHUNK_SIZE = 50
DATA_SPLITS = {"paper": (360, 400), "smoke": (1, 2)}
DATA_FOLDER = "sbi_training"
_worker = {}


def init_worker(checkpoints, out):
    _worker.update(checkpoints=checkpoints, out=out)


def generate_chunk(chunk_index):
    path = _worker["out"] / f"chunk_{chunk_index:04d}.npz"
    if path.exists():
        return chunk_index

    rng = np.random.default_rng(np.random.SeedSequence(TRAINING_SEED, spawn_key=(chunk_index,)))
    if "models" not in _worker:
        mission = MissionConfig()
        models = []
        if _worker["checkpoints"]:
            import torch

            from lisa_gwb.separator import load_separators

            torch.set_num_threads(1)
            models = load_separators(_worker["checkpoints"], mission)
        _worker.update(mission=mission, priors=SignalPriors(gal_modulation=True), models=models)
    mission, priors = _worker["mission"], _worker["priors"]
    theta, events = [], []
    for _ in range(CHUNK_SIZE):
        params = draw_sbi_parameters(rng)
        event = sbi_parameters_to_event(params, priors)
        event.update(sample_gal_modulation(rng, priors))
        theta.append(params)
        events.append(event)

    spectra = make_training_batch(
        CHUNK_SIZE, 1, 512, mission, priors, rng,
        channels=("A", "E", "T"), fixed_params=events,
        normalize=False, n_avg=mission.Nd, draw="gamma",
    )[0]
    arrays = dict(theta=np.asarray(theta), x=spectra.astype(np.float32))
    if _worker["models"]:
        from lisa_gwb.separator import separate_spectra

        signal, noise = separate_spectra(_worker["models"], spectra)
        arrays.update(s_hat=signal, n_hat=noise)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return chunk_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, help="exclusive chunk index")
    parser.add_argument("--mode", choices=DATA_SPLITS, default="paper",
                        help="paper uses 20,000 examples; smoke uses 100")
    parser.add_argument("--spectra", choices=("both", "raw"), default="both",
                        help="raw generates SBI-R inputs without separator checkpoints")
    args = parser.parse_args()
    train_chunks, total_chunks = DATA_SPLITS[args.mode]
    stop = total_chunks if args.stop is None else args.stop
    if args.workers < 1 or not 0 <= args.start < stop <= total_chunks:
        parser.error(f"Use at least one worker and 0 <= start < stop <= {total_chunks}.")

    checkpoints = []
    if args.spectra == "both":
        checkpoints = [args.results / "prod" / f"seed{s}" / "best.pt" for s in SEPARATOR_SEEDS]
    for path in checkpoints:
        if not path.is_file():
            parser.error(f"Missing separator checkpoint {path}. Train all three separator seeds first.")
    out = args.results / DATA_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        separator_sha256={str(seed): hashlib.sha256(path.read_bytes()).hexdigest()
                          for seed, path in zip(SEPARATOR_SEEDS, checkpoints)},
        simulation_seed=TRAINING_SEED, chunk_size=CHUNK_SIZE,
        train_chunks=train_chunks, total_chunks=total_chunks,
    )
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error(f"{out} was generated with different settings or checkpoints. Use a fresh results directory.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    with Pool(args.workers, initializer=init_worker, initargs=(checkpoints, out)) as pool:
        for chunk in pool.imap_unordered(generate_chunk, range(args.start, stop)):
            print(f"Ready: {out / f'chunk_{chunk:04d}.npz'}", flush=True)


if __name__ == "__main__":
    main()
