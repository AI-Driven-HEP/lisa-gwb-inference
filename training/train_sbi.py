"""Train the SBI neural spline flow on a fixed set of simulated observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from generate_sbi_data import DATA_FOLDER, DATA_SPLITS
from lisa_gwb.sbi import build_and_train, summary


def load_split(data_dir, chunks, kind):
    keys = ("theta", "s_hat", "n_hat") if kind == "primary" else ("theta", "x")
    parts = {key: [] for key in keys}
    for chunk in chunks:
        with np.load(data_dir / f"chunk_{chunk:04d}.npz") as data:
            for key in keys:
                parts[key].append(data[key])
    return {key: np.concatenate(values) for key, values in parts.items()}


def development_loss(posterior, data, kind):
    estimator = posterior.posterior_estimator.eval()
    theta = torch.as_tensor(data["theta"], dtype=torch.float32)
    spectra = torch.as_tensor(summary(kind, data), dtype=torch.float32)
    with torch.inference_mode():
        losses = [estimator.loss(y, x) for y, x in zip(theta.split(256), spectra.split(256))]
    return torch.cat(losses).mean().item()


def train(results, variant, config, threads, mode="paper"):
    torch.set_num_threads(threads)
    kind = {"sbi-s": "primary", "sbi-r": "control"}[variant]
    if kind == "control":
        config = dict(config, patience=60)
    if mode == "smoke":
        config = dict(config, batch=16, max_epochs=1)
    train_chunks, total_chunks = DATA_SPLITS[mode]
    data_dir = results / DATA_FOLDER
    manifest_path = data_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["train_chunks"], manifest["total_chunks"]) != (train_chunks, total_chunks):
            raise SystemExit(f"{data_dir} uses a different data split. Match the generator's --mode.")
        if kind == "primary" and not manifest["separator_sha256"]:
            raise SystemExit("SBI-S needs separated spectra. Generate data with --spectra both in a fresh results directory.")
    for chunk in range(total_chunks):
        if not (data_dir / f"chunk_{chunk:04d}.npz").is_file():
            raise SystemExit(f"Missing SBI chunk {chunk} in {data_dir}. Run generate_sbi_data.py first.")

    train_data = load_split(data_dir, range(train_chunks), kind)
    posterior, info = build_and_train(kind, config, train_data)
    del train_data
    development = load_split(data_dir, range(train_chunks, total_chunks), kind)
    info["development_nll"] = development_loss(posterior, development, kind)
    out = results / variant
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "posterior.pt"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(dict(posterior=posterior, kind=kind, config=config, info=info), temporary)
    temporary.replace(checkpoint)
    (out / "training.json").write_text(json.dumps(dict(info, config=config), indent=2) + "\n")
    print(f"Saved {checkpoint} after {info['epochs']} epochs. "
          f"Development NLL: {info['development_nll']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("sbi-s", "sbi-r"), default="sbi-s")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--mode", choices=DATA_SPLITS, default="paper")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    config = json.loads(Path(__file__).with_name("sbi_paper.json").read_text())
    train(args.results, args.variant, config, args.threads, args.mode)


if __name__ == "__main__":
    main()
