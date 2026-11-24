"""Neural posterior estimation for the separated or measured LISA spectra."""

import time

import numpy as np
import torch
from torch import nn

from lisa_gwb.simulation import SBI_PARAMETER_NAMES, SBI_PRIOR_BOUNDS

LO = np.array([SBI_PRIOR_BOUNDS[name][0] for name in SBI_PARAMETER_NAMES])
HI = np.array([SBI_PRIOR_BOUNDS[name][1] for name in SBI_PARAMETER_NAMES])


def summary(kind, d):
    """(n, C, N) float32 summary of the given kind from a set dict."""
    if kind == "primary":
        return np.concatenate([d["s_hat"][:, :2], d["n_hat"]], axis=1)
    if kind == "control":
        return d["x"]
    raise ValueError(kind)


class Standardize(nn.Module):
    def __init__(self, mean, sd):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))

    def forward(self, x):
        return (x.reshape(x.shape[0], *self.mean.shape) - self.mean) / self.sd


class PCAEmbed(nn.Module):
    """Standardise, project onto the training set's top-k principal components
    (fixed, not learned), scale each to unit variance, then a small MLP. The
    projection caps the capacity the flow's conditioner can spend on a
    2560-number summary with 18 000 simulations to learn from."""

    def __init__(self, mean, sd, X_std, k=128, hidden=(128, 128), out=64):
        super().__init__()
        self.std = Standardize(mean, sd)
        Z = X_std.reshape(len(X_std), -1)
        mu = Z.mean(0)
        _, sv, Vt = np.linalg.svd(Z - mu, full_matrices=False)
        comp_sd = sv[:k] / np.sqrt(len(Z) - 1)
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("V", torch.as_tensor(Vt[:k].T / comp_sd, dtype=torch.float32))
        self.explained = float((sv[:k] ** 2).sum() / (sv ** 2).sum())
        dims = [k, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], out))

    def forward(self, x):
        z = self.std(x).flatten(1)
        return self.net((z - self.mu) @ self.V)


def build_and_train(kind, cfg, tr):
    """Fit the PCA projection, embedding and conditional spline flow."""
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform
    torch.manual_seed(cfg.get("seed", 0))
    X = summary(kind, tr).astype(np.float32)
    mean, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)                # a bin the separator floors everywhere
    X_std = ((X - mean) / sd).astype(np.float32)
    embedding = PCAEmbed(mean, sd, X_std, k=cfg["pca_k"],
                         hidden=tuple(cfg["embed_hidden"]), out=cfg["embed_out"])
    del X_std
    prior = BoxUniform(torch.tensor(LO, dtype=torch.float32), torch.tensor(HI, dtype=torch.float32))
    estimator = posterior_nn(
        model=cfg["flow"], z_score_x="none",
        z_score_theta="transform_to_unconstrained", x_dist=prior,
        hidden_features=cfg["hidden"], num_transforms=cfg["transforms"],
        embedding_net=embedding,
    )
    inf = NPE(prior=prior, density_estimator=estimator, show_progress_bars=False)
    inf.append_simulations(torch.tensor(tr["theta"], dtype=torch.float32), torch.from_numpy(X))
    t0 = time.time()
    est = inf.train(training_batch_size=cfg["batch"],
                    learning_rate=cfg["lr"],
                    stop_after_epochs=cfg["patience"],
                    max_num_epochs=cfg["max_epochs"],
                    validation_fraction=0.1, show_train_summary=False)
    s = inf._summary
    info = dict(train_seconds=time.time() - t0, epochs=len(s["training_loss"]),
                best_val_loss=float(s["best_validation_loss"][-1])
                if "best_validation_loss" in s else float(min(s["validation_loss"])),
                train_loss_curve=[float(v) for v in s["training_loss"]],
                val_loss_curve=[float(v) for v in s["validation_loss"]],
                pca_explained=embedding.explained)
    return inf.build_posterior(est), info


def sample(posterior, spectra, n=4000, seed=20260927, block=10):
    """Draw parameters with shape (n, observations, 4) from log10 PSD inputs."""
    torch.manual_seed(seed)
    blocks = []
    for start in range(0, len(spectra), block):
        x = torch.as_tensor(spectra[start:start + block], dtype=torch.float32)
        draws = posterior.sample_batched((n,), x=x, show_progress_bars=False,
                                         reject_outside_prior=False)
        blocks.append(draws.detach().numpy())
    return np.concatenate(blocks, axis=1)
