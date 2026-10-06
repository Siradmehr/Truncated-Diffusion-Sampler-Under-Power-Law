"""Experiment 3: Gaussian-mixture models of CIFAR patches and spectral truncation.

The paper models image patches as a zero-mean Gaussian mixture
q = sum_m pi_m N(0, C_m), after removing the mean patch. This script fits that
model to CIFAR-10/100 patches and then uses the fitted mixture as a known,
non-Gaussian data law for the truncation results:

  1. Fit. Zero-mean mixtures for several numbers of components M, fitted by EM;
     held-out log-likelihood against a single Gaussian (M = 1); M chosen on a
     validation split.
  2. Assumption 1 per component. Power-law exponent and envelope ratio of every
     component spectrum and of the pooled spectrum.
  3. Shared basis. Truncate in the eigenbasis of the pooled covariance. TV of the
     ideal truncated sampler for every K, compared with the Gaussian prediction
     of Theorem 1, and rules for choosing K (noise level, explained variance,
     Pinsker tail budget, Theorem 2(a) with fitted constants) compared with the
     smallest sufficient K.
  4. Component bases (Proposition 4). Truncate each component in its own basis
     and compare the mixture TV with the bound sum_m pi_m eps_m.

TV is estimated from the exact mixture densities,
TV(P, Q) = E_{(P+Q)/2} |tanh(log(dP/dQ) / 2)|, with half of the draws from each
law and common random numbers across K.

Example:
    python 03_gmm_patches.py --profile smoke
    python 03_gmm_patches.py --profile paper --datasets cifar10 cifar100
    python 03_gmm_patches.py --profile smoke --data synthetic   # no download; code test only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

EXPERIMENT = "03_gmm_patches"
LOG2PI = math.log(2 * math.pi)

PROFILES = {
    "smoke": dict(datasets=["cifar10"], patch=8, patches_per_image=1, n_val_images=1000,
                  max_train_images=6000, max_test_images=2000, M_grid=[1, 2, 4], em_max_iter=40,
                  knees=[4, 16], eps=[0.2, 0.1, 0.05], mc=1500, budget_points=8),
    "paper": dict(datasets=["cifar10", "cifar100"], patch=8, patches_per_image=2,
                  n_val_images=5000, max_train_images=45000, max_test_images=10000,
                  M_grid=[1, 2, 4, 8, 16, 32], em_max_iter=100, knees=[2, 4, 8, 16, 32],
                  eps=[0.3, 0.2, 0.1, 0.05, 0.02, 0.01], mc=6000, budget_points=24),
}

BASE_CFG = dict(seed=2027, dequantize=True, em_tol=1e-5, reg_rel=1e-5, init_iters=8,
                window_low=8, window_dim_fraction=0.25, threads=None)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
class Output:
    """Writes tables and figures to one run directory."""

    def __init__(self, base: Path, cfg: dict):
        run_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
        self.dir = base / EXPERIMENT / f"{cfg['profile']}_{cfg['data']}_{run_hash}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_hash = run_hash

    def table(self, name, df):
        df.to_csv(self.dir / name, index=False)
        return df

    def figure(self, name):
        plt.tight_layout()
        plt.savefig(self.dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Data: CIFAR images, random patches, image-disjoint splits
# ---------------------------------------------------------------------------
def cifar_arrays(name, data_dir, cfg):
    """uint8 CIFAR train and test images (N, 32, 32, 3)."""
    if cfg["data"] == "synthetic":
        return dead_leaves(name, cfg)
    import torchvision.datasets as datasets

    cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[name]
    tr = cls(str(data_dir), train=True, download=True)
    te = cls(str(data_dir), train=False, download=True)
    return np.asarray(tr.data, np.uint8), np.asarray(te.data, np.uint8)


def dead_leaves(name, cfg, size=32, disks=150):
    """Occluding random disks with power-law radii and random colours (code test only)."""
    n_train = cfg["max_train_images"] + cfg["n_val_images"]
    n_test = cfg["max_test_images"]
    rng = np.random.default_rng({"cifar10": 1, "cifar100": 2}.get(name, 3))
    yy, xx = np.mgrid[:size, :size]

    def one():
        rad = (1 - rng.uniform(size=disks) * (1 - (1 / 24) ** 2)) ** -0.5   # p(r) ~ r^-3 on [1, 24]
        cy, cx = rng.uniform(-4, size + 4, (2, disks))
        col = rng.uniform(0, 1, (disks + 1, 3))
        cover = (yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2 \
            <= rad[:, None, None] ** 2
        top = np.where(cover.any(0), cover.argmax(0), disks)                  # first disk on top
        return np.clip(np.rint(255 * col[top]), 0, 255).astype(np.uint8)

    return np.stack([one() for _ in range(n_train)]), np.stack([one() for _ in range(n_test)])


def patches(images, p, per_image, rng, dequantize):
    """Random p x p patches, flattened and scaled to [0, 1] as (u + U) / 256."""
    n, h, w, _ = images.shape
    ys = rng.integers(0, h - p + 1, (n, per_image))
    xs = rng.integers(0, w - p + 1, (n, per_image))
    out = np.stack([images[i, y:y + p, x:x + p].reshape(-1)
                    for i in range(n) for y, x in zip(ys[i], xs[i])]).astype(np.float64)
    return (out + (rng.uniform(size=out.shape) if dequantize else 0.5)) / 256


def load_patches(name, data_dir, cfg, out):
    """Mean-centred train/val/test patches; train and val come from disjoint training images."""
    tr, te = cifar_arrays(name, data_dir, cfg)
    rng = np.random.default_rng(cfg["seed"])
    order = rng.permutation(len(tr))
    val_ids = order[:cfg["n_val_images"]]
    train_ids = order[cfg["n_val_images"]:][:cfg["max_train_images"]]
    test_ids = rng.permutation(len(te))[:cfg["max_test_images"]]
    splits = [("train", tr, train_ids), ("val", tr, val_ids), ("test", te, test_ids)]
    P = {s: patches(arr[ids], cfg["patch"], cfg["patches_per_image"],
                    np.random.default_rng(cfg["seed"] + j), cfg["dequantize"])
         for j, (s, arr, ids) in enumerate(splits)}
    mean = P["train"].mean(0)
    info = {s: dict(images=int(len(ids)), patches=int(len(P[s]))) for s, _, ids in splits}
    (out.dir / f"{name}_splits.json").write_text(json.dumps(info, indent=2))
    return {s: x - mean for s, x in P.items()}, mean


# ---------------------------------------------------------------------------
# Zero-mean Gaussian mixture fitted by EM
# ---------------------------------------------------------------------------
def comp_logpdf(X, C=None, chol=None):
    """log N(x; 0, C) for every row of X, from C or its lower Cholesky factor."""
    L = chol if chol is not None else sla.cholesky(C, lower=True, check_finite=False)
    Z = sla.solve_triangular(L, X.T, lower=True, check_finite=False)
    return -0.5 * (Z * Z).sum(0) - np.log(np.diag(L)).sum() - 0.5 * X.shape[1] * LOG2PI


class ZeroMeanGMM:
    """q = sum_m w_m N(0, C_m)."""

    def __init__(self, weights, covs):
        self.w = np.asarray(weights, float)
        self.C = np.asarray(covs, float)

    @property
    def M(self):
        return len(self.w)

    def log_joint(self, X):
        """(n, M) array of log w_m + log N(x; 0, C_m)."""
        return np.stack([math.log(w) + comp_logpdf(X, C) for w, C in zip(self.w, self.C)], 1)

    def score(self, X, batch=20000):
        """Mean log-likelihood (nats per patch)."""
        ll = [logsumexp(self.log_joint(X[i:i + batch]), 1) for i in range(0, len(X), batch)]
        return float(np.mean(np.concatenate(ll)))

    def pooled_cov(self):
        return np.tensordot(self.w, self.C, 1)


def init_lines(X, M, rng, iters):
    """Hard clustering of patch directions up to sign (k-lines).

    A zero-mean component is symmetric under x -> -x, so patches are clustered
    by direction rather than by position.
    """
    U = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    cent = U[rng.choice(len(U), M, replace=False)]
    for _ in range(iters):
        lab = np.abs(U @ cent.T).argmax(1)
        for m in range(M):
            S = U[lab == m]
            cent[m] = (sla.eigh(S.T @ S, check_finite=False)[1][:, -1] if len(S) > 1
                       else U[rng.integers(len(U))])
    return np.abs(U @ cent.T).argmax(1)


def fit_gmm(X, M, seed, max_iter, tol, reg_rel, init_iters, log=None):
    """EM for a zero-mean mixture.

    A ridge reg_rel * tr(Sigma) / d keeps covariances positive definite.
    Components with fewer than 2d effective points are re-seeded from the pooled
    covariance. EM stops after three consecutive gains below tol nats per
    dimension, or after max_iter iterations.
    """
    n, d = X.shape
    rng = np.random.default_rng(seed)
    reg = reg_rel * np.trace(X.T @ X / n) / d
    eye = np.eye(d)
    pooled = X.T @ X / n + reg * eye
    if M == 1:
        return ZeroMeanGMM([1.0], [pooled]), dict(iters=0, reseeded=0, converged=True)

    lab = init_lines(X, M, rng, init_iters)
    w = np.array([(lab == m).mean() for m in range(M)])
    C = np.empty((M, d, d))
    for m in range(M):
        S = X[lab == m]
        C[m] = S.T @ S / len(S) + reg * eye if len(S) > 2 * d else pooled * rng.uniform(0.5, 2)
    w = np.maximum(w, 1 / n)
    w /= w.sum()

    prev, it, converged, small, reseeded = -np.inf, 0, False, 0, 0
    for it in range(1, max_iter + 1):
        lj = ZeroMeanGMM(w, C).log_joint(X)
        ll = logsumexp(lj, 1)
        R = np.exp(lj - ll[:, None])
        cur = float(ll.mean())
        Nm = R.sum(0)
        for m in range(M):
            if Nm[m] < 2 * d:                       # lost support: re-seed
                C[m] = pooled * rng.uniform(0.5, 2)
                Nm[m] = 2 * d
                reseeded += 1
                continue
            C[m] = (X * R[:, m:m + 1]).T @ X / Nm[m] + reg * eye
        w = Nm / Nm.sum()
        if log is not None:
            log.append(dict(M=M, iter=it, train_ll=cur))
        small = small + 1 if cur - prev < tol * d else 0
        if small >= 3 and it > 5:
            converged = True
            break
        prev = cur
    return ZeroMeanGMM(w, C), dict(iters=it, reseeded=reseeded, converged=converged)


def eig_desc(C):
    """Eigenvalues (descending, floored at 1e-300) and eigenvectors of a symmetric matrix."""
    v, V = sla.eigh(C, check_finite=False)
    return np.maximum(v[::-1], 1e-300), V[:, ::-1].copy()


# ---------------------------------------------------------------------------
# TV estimators
# ---------------------------------------------------------------------------
def mean_se(vals):
    v = np.asarray(vals)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def diag_tv(a, b, n, seed, batch=500):
    """Monte Carlo TV(N(0, diag a), N(0, diag b)) (the Gaussian reference)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = a != b
    a, b = a[keep], b[keep]
    if not len(a):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    vals = []
    for s in range(0, n, batch):
        m = min(batch, n - s)
        coin = rng.integers(0, 2, (m, 1))
        z = rng.standard_normal((m, len(a)))
        x2 = z * z * np.where(coin, a, b)
        ell = -0.5 * np.log(a / b).sum() - 0.5 * (x2 * (1 / a - 1 / b)).sum(1)
        vals.append(np.abs(np.tanh(ell / 2)))
    return mean_se(np.concatenate(vals))


def pinsker_suffix(lam, eta2):
    """Explicit bound of Theorem 1(a), sqrt(1/8 sum_{i>K} x_i^2 / (1 + x_i)), for every K."""
    x = lam / eta2
    t = x * x / (1 + x)
    return np.sqrt(np.r_[np.cumsum(t[::-1])[::-1], 0.0] / 8)


def wiener_suffix(lam, eta2):
    """Wiener-gain tail ||g_{>K}||_2 for every K."""
    g = lam / (lam + eta2)
    return np.sqrt(np.r_[np.cumsum((g * g)[::-1])[::-1], 0.0])


def tv_from_logs(lp_p, lq_p, lp_q, lq_q):
    """Stratified TV estimate: half of the draws from P, half from Q."""
    v = np.r_[np.abs(np.tanh((lp_p - lq_p) / 2)), np.abs(np.tanh((lp_q - lq_q) / 2))]
    return mean_se(v)


class SharedBasis:
    """Target q^eta = sum_m pi_m N(0, C_m + eta^2 I) and ideal truncated samplers.

    Works in the eigenbasis of the pooled covariance. The truncated sampler keeps
    coordinates < K of a draw from q^eta and replaces the rest by fresh
    N(0, eta^2) noise, which gives exact draws of q_K^eta with common random
    numbers across K.
    """

    def __init__(self, gmm, eta2, n, seed):
        self.lam, self.V = eig_desc(gmm.pooled_cov())
        self.w = gmm.w
        self.eta2 = eta2
        self.d = d = len(self.lam)
        self.Ct = np.stack([self.V.T @ C @ self.V for C in gmm.C])   # components in PC coordinates
        self.Lq = [sla.cholesky(C + eta2 * np.eye(d), lower=True) for C in self.Ct]
        rng = np.random.default_rng(seed)
        h = n // 2
        self.yq = self._sample_q(h, rng)
        self.yp_full = self._sample_q(h, rng)
        self.xi = math.sqrt(eta2) * rng.standard_normal((h, d))
        self.lq_q = self.logq(self.yq)

    def _sample_q(self, n, rng):
        c = rng.choice(len(self.w), n, p=self.w)
        y = np.empty((n, self.d))
        for m in np.unique(c):
            idx = c == m
            y[idx] = rng.standard_normal((idx.sum(), self.d)) @ self.Lq[m].T
        return y

    def logq(self, Y):
        terms = [math.log(w) + comp_logpdf(Y, chol=L) for w, L in zip(self.w, self.Lq)]
        return logsumexp(np.stack(terms, 1), 1)

    def logp(self, Y, K):
        tail = (-0.5 * (Y[:, K:] ** 2).sum(1) / self.eta2
                - 0.5 * (self.d - K) * (math.log(self.eta2) + LOG2PI))
        if K == 0:
            return tail
        head = [math.log(w) + comp_logpdf(Y[:, :K], self.Ct[m][:K, :K] + self.eta2 * np.eye(K))
                for m, w in enumerate(self.w)]
        return logsumexp(np.stack(head, 1), 1) + tail

    def tv(self, K):
        if K >= self.d:
            return 0.0, 0.0
        yp = np.c_[self.yp_full[:, :K], self.xi[:, K:]]
        return tv_from_logs(self.logp(yp, K), self.logq(yp), self.logp(self.yq, K), self.lq_q)

    def tv_vs_gaussian(self):
        """TV(q^eta, N(0, Sigma_pooled + eta^2 I)): how non-Gaussian the mixture is at eta."""
        var = self.lam + self.eta2

        def lg(Y):
            return -0.5 * (Y * Y / var).sum(1) - 0.5 * np.log(var).sum() - 0.5 * self.d * LOG2PI

        yg = np.random.default_rng(5).standard_normal(self.yq.shape) * np.sqrt(var)
        return tv_from_logs(lg(yg), self.logq(yg), lg(self.yq), self.lq_q)


class ComponentBases:
    """Proposition 4: component m keeps K_m directions of its own eigenbasis.

    Discarded directions are replaced by N(0, eta^2) noise.
    """

    def __init__(self, gmm, eta2, n, seed):
        self.w = gmm.w
        self.eta2 = eta2
        eig = [eig_desc(C) for C in gmm.C]
        self.lam = np.stack([e[0] for e in eig])
        self.V = np.stack([e[1] for e in eig])
        self.M, self.d = self.lam.shape
        rng = np.random.default_rng(seed)
        h = n // 2
        self.cq = rng.choice(self.M, h, p=self.w)
        self.zq = rng.standard_normal((h, self.d))
        self.cp = rng.choice(self.M, h, p=self.w)
        self.zp = rng.standard_normal((h, self.d))
        self.xq = self._draw(self.cq, self.zq, self.lam + eta2)
        self.Pq = self._proj(self.xq)

    def _draw(self, c, z, var):
        x = np.empty_like(z)
        for m in np.unique(c):
            idx = c == m
            x[idx] = (z[idx] * np.sqrt(var[m])) @ self.V[m].T
        return x

    def _proj(self, X):
        """Squared coordinates of X in every component basis."""
        return [(X @ self.V[m]) ** 2 for m in range(self.M)]

    def _logmix(self, P2, var):
        terms = [math.log(self.w[m]) - 0.5 * (P2[m] / var[m]).sum(1) - 0.5 * np.log(var[m]).sum()
                 for m in range(self.M)]
        return logsumexp(np.stack(terms, 1), 1) - 0.5 * self.d * LOG2PI

    def variances(self, Ks):
        mask = np.arange(self.d)[None, :] < np.asarray(Ks)[:, None]
        return np.where(mask, self.lam + self.eta2, self.eta2)

    def tv(self, Ks):
        vp = self.variances(Ks)
        vq = self.lam + self.eta2
        Pp = self._proj(self._draw(self.cp, self.zp, vp))
        return tv_from_logs(self._logmix(Pp, vp), self._logmix(Pp, vq),
                            self._logmix(self.Pq, vp), self._logmix(self.Pq, vq))


def numerical_checks():
    """EM recovers a known mixture; with one component both estimators match the Gaussian TV."""
    rng = np.random.default_rng(0)
    d = 6
    A = [np.diag(np.r_[4.0, np.ones(d - 1) * 0.1]), np.diag(np.r_[np.ones(d - 1) * 0.1, 4.0])]
    X = np.concatenate([rng.multivariate_normal(np.zeros(d), A[0], 6000),
                        rng.multivariate_normal(np.zeros(d), A[1], 4000)])
    true = ZeroMeanGMM([0.6, 0.4], A)
    g, _ = fit_gmm(X, 2, 1, 200, 1e-8, 1e-6, 5)
    assert g.score(X) >= true.score(X) - 1e-2 and abs(sorted(g.w)[1] - 0.6) < 0.03, "EM check failed"

    lam = np.arange(1, d + 1.0) ** -1.5
    G = ZeroMeanGMM([1.0], [np.diag(lam)])
    eta2 = lam[1]
    ref = diag_tv(np.full(d - 2, eta2), lam[2:] + eta2, 40000, 3)[0]
    sb = SharedBasis(G, eta2, 40000, 4).tv(2)[0]
    cb = ComponentBases(G, eta2, 40000, 5).tv([2])[0]
    assert abs(sb - ref) < 0.01 and abs(cb - ref) < 0.01, (sb, cb, ref)
    assert SharedBasis(G, eta2, 2000, 6).tv(d)[0] == 0
    assert ComponentBases(G, eta2, 2000, 7).tv([d])[0] < 1e-12
    print(f"Checks passed: EM recovery; Gaussian reduction "
          f"(shared {sb:.4f}, component {cb:.4f}, reference {ref:.4f})")


# ---------------------------------------------------------------------------
# Parts 1-2: fit the mixtures, choose M, check Assumption 1 per component
# ---------------------------------------------------------------------------
def powerlaw_fit(lam, low, high):
    """Least-squares power law on log-spaced ranks in [low, high]; returns beta, C, envelope ratio."""
    i = np.unique(np.rint(np.geomspace(low, high, 40)).astype(int))
    c = np.polyfit(np.log(i), np.log(lam[i - 1]), 1)
    beta = -c[0]
    x = lam[i - 1] * i**beta
    return beta, math.exp(c[1]), float(x.max() / x.min())


def weighted_quantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(np.asarray(x)[o][np.searchsorted(c, q)])


def fit_all(cfg, data_dir, out):
    """Fit mixtures for every dataset and M; returns the selected models and the tables."""
    fits, fit_rows, comp_rows, em_log = {}, [], [], []
    for name in cfg["datasets"]:
        P, mean_patch = load_patches(name, data_dir, cfg, out)
        d = P["train"].shape[1]
        print(f"{name}: d={d}, patches train/val/test = "
              f"{len(P['train'])}/{len(P['val'])}/{len(P['test'])}")
        models = {}
        for M in cfg["M_grid"]:
            t = time.time()
            g, info = fit_gmm(P["train"], M, cfg["seed"] + M, cfg["em_max_iter"], cfg["em_tol"],
                              cfg["reg_rel"], cfg["init_iters"], em_log)
            row = dict(dataset=name, M=M, d=d, **info, seconds=time.time() - t,
                       params=M * d * (d + 1) // 2 + M - 1,
                       **{f"{s}_ll": g.score(P[s]) for s in ("train", "val", "test")})
            fit_rows.append(row)
            models[M] = g
            print(f"  M={M:3d}: val {row['val_ll'] / d:+.4f} nats/dim, "
                  f"test {row['test_ll'] / d:+.4f}, {info['iters']} it, {row['seconds']:.0f}s")
        F = pd.DataFrame([r for r in fit_rows if r["dataset"] == name])
        best = int(F.loc[F.val_ll.idxmax(), "M"])
        g = fits[name] = models[best]
        np.savez_compressed(out.dir / f"{name}_gmm_M{best}.npz", weights=g.w, covs=g.C,
                            mean_patch=mean_patch)

        lo, hi = cfg["window_low"], int(d * cfg["window_dim_fraction"])
        lp = eig_desc(g.pooled_cov())[0]
        bp, Cp, ccp = powerlaw_fit(lp, lo, hi)
        comp_rows.append(dict(dataset=name, M=best, component="pooled", weight=1.0, beta=bp,
                              C_sp=Cp, C_over_c=ccp, top_eig=lp[0]))
        for m in range(g.M):
            lam = eig_desc(g.C[m])[0]
            b, Cs, cc = powerlaw_fit(lam, lo, hi)
            comp_rows.append(dict(dataset=name, M=best, component=m, weight=g.w[m], beta=b,
                                  C_sp=Cs, C_over_c=cc, top_eig=lam[0]))
        print(f"  chosen M={best}; pooled beta={bp:.2f}")

    fit_df = pd.DataFrame(fit_rows)
    comp_df = out.table("component_spectra.csv", pd.DataFrame(comp_rows))
    out.table("em_trace.csv", pd.DataFrame(em_log))
    for _, F in fit_df.groupby("dataset"):
        base = F[F.M == 1].iloc[0]
        for s in ("val", "test"):
            fit_df.loc[F.index, f"{s}_gain_bits_per_dim"] = \
                (F[f"{s}_ll"] - base[f"{s}_ll"]) / (F.d * math.log(2))
    out.table("gmm_fit.csv", fit_df)
    print(fit_df[["dataset", "M", "iters", "val_gain_bits_per_dim",
                  "test_gain_bits_per_dim"]].round(4).to_string(index=False))

    summ = []
    for name, D in comp_df[comp_df.component != "pooled"].groupby("dataset"):
        b, w, cc = D.beta.to_numpy(), D.weight.to_numpy(), D.C_over_c.to_numpy()
        pooled = comp_df[(comp_df.dataset == name) & (comp_df.component == "pooled")]
        summ.append(dict(dataset=name, M=len(D), beta_w_median=weighted_quantile(b, w, 0.5),
                         beta_w_q10=weighted_quantile(b, w, 0.1),
                         beta_w_q90=weighted_quantile(b, w, 0.9),
                         weight_beta_gt_1=float(w[b > 1].sum()),
                         Cc_w_median=weighted_quantile(cc, w, 0.5),
                         Cc_w_q90=weighted_quantile(cc, w, 0.9),
                         beta_pooled=float(pooled.beta.iloc[0])))
    print(out.table("component_assumption1_summary.csv", pd.DataFrame(summ)).round(3)
          .to_string(index=False))
    plot_fit(fit_df, fits, cfg, out)
    return fits, fit_df, comp_df


def plot_fit(fit_df, fits, cfg, out):
    fig, ax = plt.subplots(1, 1 + len(fits), figsize=(5 + 4.5 * len(fits), 3.6))
    for name, F in fit_df.groupby("dataset"):
        ax[0].semilogx(F.M, F.test_gain_bits_per_dim, "o-", label=f"{name} test")
        ax[0].semilogx(F.M, F.val_gain_bits_per_dim, "s--", alpha=0.6, label=f"{name} val")
    ax[0].set(xlabel="components $M$", ylabel="log-lik. gain over $M=1$ (bits/dim)",
              title="Held-out fit")
    ax[0].legend(fontsize=7)
    ax[0].set_xticks(cfg["M_grid"])
    ax[0].set_xticklabels([str(m) for m in cfg["M_grid"]])
    ax[0].minorticks_off()
    for a, (name, g) in zip(ax[1:], fits.items()):
        for m in range(g.M):
            lam = eig_desc(g.C[m])[0]
            a.loglog(np.arange(1, len(lam) + 1), lam, color="tab:blue", lw=0.8,
                     alpha=float(0.15 + 0.85 * g.w[m] / g.w.max()))
        lp = eig_desc(g.pooled_cov())[0]
        a.loglog(np.arange(1, len(lp) + 1), lp, color="k", lw=1.8, label="pooled")
        a.axvspan(cfg["window_low"], len(lp) * cfg["window_dim_fraction"], color="gray", alpha=0.1)
        a.set(xlabel="rank", ylabel="eigenvalue", title=f"{name}: {g.M} component spectra")
        a.legend(fontsize=7)
    out.figure("gmm_fit_and_spectra")


# ---------------------------------------------------------------------------
# Part 3: shared (pooled) basis
# ---------------------------------------------------------------------------
def crossing(K, y, eps):
    """First K with y <= eps, interpolated log-linearly between grid points."""
    j = np.flatnonzero(y <= eps)
    if not len(j):
        return np.nan
    j = j[0]
    if j == 0 or y[j - 1] <= 0 or y[j] <= 0 or K[j - 1] == 0:
        return float(K[j])
    lk, ly = np.log([K[j - 1], K[j]]), np.log([y[j - 1], y[j]])
    return float(np.exp(lk[0] + (math.log(eps) - ly[0]) * (lk[1] - lk[0]) / (ly[1] - ly[0])))


def selectors(lam, eta2, eps, beta, C_sp):
    """K chosen by each rule from the pooled spectrum."""
    d = len(lam)
    cum = np.cumsum(lam)
    out = {"noise_level": int((lam >= eta2).sum())}
    for rho in (0.95, 0.99, 0.999):
        out[f"variance_{rho}"] = int(min(d, np.searchsorted(cum, rho * cum[-1]) + 1))
    out["tail_budget"] = int(np.flatnonzero(pinsker_suffix(lam, eta2) <= eps)[0])
    # Smallest K at which the right-hand side of bound (11) is at most eps.
    k2a = ((C_sp / eta2) / (eps * math.sqrt(8 * (2 * beta - 1)))) ** (2 / (2 * beta - 1))
    out["theorem2a_fitted"] = int(min(d, math.ceil(k2a)))
    return out


def shared_basis(fits, comp_df, cfg, out):
    curve_rows, req_rows, sel_rows, nong_rows = [], [], [], []
    for name, g in fits.items():
        d = g.C.shape[1]
        lam = eig_desc(g.pooled_cov())[0]
        cp = comp_df[(comp_df.dataset == name) & (comp_df.component == "pooled")].iloc[0]
        for j, k in enumerate(cfg["knees"]):
            eta2 = float(lam[k - 1])                 # exactly k pooled directions above eta
            t = time.time()
            SB = SharedBasis(g, eta2, cfg["mc"], cfg["seed"] + 100 * j)
            tg, tgse = SB.tv_vs_gaussian()
            nong_rows.append(dict(dataset=name, knee=k, eta2=eta2, tv_mixture_vs_gaussian=tg,
                                  se=tgse))
            Ks = np.arange(0, d + 1)
            res = [SB.tv(int(K)) for K in Ks]
            tv = np.array([r[0] for r in res])
            se = np.array([r[1] for r in res])
            G = wiener_suffix(lam, eta2)[Ks]
            Pn = pinsker_suffix(lam, eta2)[Ks]
            gauss = np.array([diag_tv(np.full(d - K, eta2), lam[K:] + eta2, cfg["mc"], 7)[0]
                              if K < d else 0.0 for K in Ks])
            curve_rows += [dict(dataset=name, M=g.M, knee=k, eta2=eta2, K=int(K), tv=a, se=b,
                                tv_gaussian=c, wiener_tail=x, pinsker=p)
                           for K, a, b, c, x, p in zip(Ks, tv, se, gauss, G, Pn)]
            for e in cfg["eps"]:
                Kint = int(Ks[np.flatnonzero(tv <= e)[0]])
                conservative = tv + 1.96 * se <= e
                req_rows.append(dict(dataset=name, knee=k, eps=e, K_star=crossing(Ks, tv, e),
                                     K_star_int=Kint, K_gaussian=crossing(Ks, gauss, e),
                                     K_lo=int(Ks[np.flatnonzero(conservative)[0]])
                                     if conservative.any() else d))
                for rule, K in selectors(lam, eta2, e, cp.beta, cp.C_sp).items():
                    sel_rows.append(dict(dataset=name, knee=k, eps=e, rule=rule, K=K,
                                         capped_at_d=K >= d, K_star=Kint,
                                         K_ratio=K / max(Kint, 1), tv=tv[K],
                                         tv_hi=tv[K] + 1.96 * se[K],
                                         meets=bool(tv[K] + 1.96 * se[K] <= e)))
            print(f"{name} k_eta={k}: shared-basis curve in {time.time() - t:.0f}s; "
                  f"TV(mixture, Gaussian) at this eta = {tg:.3f}")

    curves = out.table("shared_basis_curves.csv", pd.DataFrame(curve_rows))
    req = out.table("shared_basis_required_K.csv", pd.DataFrame(req_rows))
    sel = out.table("shared_basis_selectors.csv", pd.DataFrame(sel_rows))
    out.table("nongaussianity.csv", pd.DataFrame(nong_rows))

    print(sel.groupby(["dataset", "rule"]).agg(meets=("meets", "mean"),
                                               K_ratio_median=("K_ratio", "median"),
                                               K_ratio_max=("K_ratio", "max")).round(3))
    ratio = req.K_star / req.K_gaussian
    print("K* mixture / K* Gaussian prediction:\n",
          ratio.groupby(req.dataset).agg(["min", "median", "max"]).round(3))
    c = curves[(curves.wiener_tail < 0.5) & (curves.tv > 5 * curves.se)]
    print("TV / Wiener-gain tail (Gaussian small-gain constant 0.28):\n",
          (c.tv / c.wiener_tail).groupby(c.dataset).agg(["min", "median", "max"]).round(3))
    plot_shared(curves, sel, cfg, out)
    return req


def plot_shared(curves, sel, cfg, out):
    n = curves.dataset.nunique()
    fig, ax = plt.subplots(n, 3, figsize=(15, 3.6 * n), squeeze=False)
    for i, (name, C) in enumerate(curves.groupby("dataset", sort=False)):
        for j, (k, s) in enumerate(C.groupby("knee")):
            s = s[s.K < s.K.max()]
            col = plt.cm.viridis(j / max(1, len(cfg["knees"]) - 1))
            ax[i, 0].semilogy(s.K, s.tv, color=col, lw=1.5, label=f"$k_\\eta={k}$")
            ax[i, 0].semilogy(s.K, s.tv_gaussian, color=col, lw=0.8, ls="--")
            ax[i, 1].loglog(s.wiener_tail, s.tv, ".", color=col, ms=3)
        ax[i, 0].set(xlabel="$K$ (pooled basis)", ylabel="TV", ylim=(1e-4, 1.1),
                     title=f"{name}: mixture (solid) vs Gaussian prediction (dashed)")
        ax[i, 0].legend(fontsize=7)
        gg = np.geomspace(1e-4, 1, 20)
        ax[i, 1].loglog(gg, 0.28 * gg, "k--", lw=0.8, label="Gaussian: 0.28$\\times$")
        ax[i, 1].set(xlabel="pooled Wiener-gain tail norm", ylabel="TV",
                     title="Theorem 1 relation")
        ax[i, 1].legend(fontsize=7)
        S = sel[sel.dataset == name]
        rules = list(dict.fromkeys(S.rule))
        for r_i, rule in enumerate(rules):
            q = S[S.rule == rule]
            jitter = np.random.default_rng(r_i).uniform(-0.2, 0.2, len(q))
            ax[i, 2].semilogy(r_i + jitter, np.maximum(q.tv / q.eps, 1e-3), "o", ms=3)
        ax[i, 2].axhline(1, color="k", lw=0.8)
        ax[i, 2].set_xticks(range(len(rules)))
        ax[i, 2].set_xticklabels([r.replace("_", "\n") for r in rules], fontsize=7)
        ax[i, 2].set(ylabel="achieved TV / $\\varepsilon$",
                     title="selection rules from the pooled spectrum")
    out.figure("shared_basis")


# ---------------------------------------------------------------------------
# Part 4: component bases and Proposition 4
# ---------------------------------------------------------------------------
def component_bases(fits, req, cfg, out):
    """Each component keeps the smallest K_m whose Theorem 1(a) bound is at most a budget b.

    Then eps_m <= b and, by Proposition 4, the mixture error is at most
    sum_m pi_m eps_m <= b.
    """
    rows = []
    for name, g in fits.items():
        d = g.C.shape[1]
        lamp = eig_desc(g.pooled_cov())[0]
        for j, k in enumerate(cfg["knees"]):
            eta2 = float(lamp[k - 1])
            CB = ComponentBases(g, eta2, cfg["mc"], cfg["seed"] + 500 + j)
            suf = [pinsker_suffix(CB.lam[m], eta2) for m in range(CB.M)]
            budgets = np.unique(np.r_[np.geomspace(1e-3, 0.5, cfg["budget_points"]), cfg["eps"]])
            rules = {f"budget_{b:.4g}": [int(np.flatnonzero(s <= b)[0]) for s in suf]
                     for b in budgets}
            rules["noise_level"] = [int((CB.lam[m] >= eta2).sum()) for m in range(CB.M)]
            for rho in (0.95, 0.99, 0.999):
                rules[f"variance_{rho}"] = [
                    int(min(d, np.searchsorted(np.cumsum(CB.lam[m]), rho * CB.lam[m].sum()) + 1))
                    for m in range(CB.M)]
            for rule, Ks in rules.items():
                tv, se = CB.tv(Ks)
                eps_m = [diag_tv(np.full(d - K, eta2), CB.lam[m][K:] + eta2, 2000, 11 + m)[0]
                         if K < d else 0.0 for m, K in enumerate(Ks)]
                is_budget = rule.startswith("budget")
                rows.append(dict(dataset=name, M=g.M, knee=k, eta2=eta2,
                                 rule="budget" if is_budget else rule,
                                 budget=float(rule.split("_")[1]) if is_budget else np.nan,
                                 tv_mixture=tv, se=se, bound_prop4=float(np.dot(g.w, eps_m)),
                                 mean_K=float(np.dot(g.w, Ks)), max_K=int(max(Ks)),
                                 min_K=int(min(Ks))))
            print(f"{name} k_eta={k}: component-basis rules done")

    comp_tv = out.table("component_basis.csv", pd.DataFrame(rows))
    B = comp_tv[comp_tv.rule == "budget"].copy()
    print("Proposition 4 bound respected:",
          bool((B.tv_mixture - 1.96 * B.se <= B.bound_prop4 + 1e-9).all()))

    summary = []
    for (name, k), s in B.groupby(["dataset", "knee"]):
        s = s.sort_values("budget")
        for e in cfg["eps"]:
            nominal = s[np.isclose(s.budget, e)].iloc[0]
            ok = s[s.tv_mixture + 1.96 * s.se <= e]
            tuned = ok.iloc[-1] if len(ok) else None
            shared = req[(req.dataset == name) & (req.knee == k) & (req.eps == e)].K_star_int.iloc[0]
            summary.append(dict(dataset=name, knee=k, eps=e, tv_at_budget_eps=nominal.tv_mixture,
                                bound_at_budget_eps=nominal.bound_prop4,
                                meanK_budget_eps=nominal.mean_K,
                                largest_budget_meeting_eps=tuned.budget if tuned is not None else np.nan,
                                meanK_tuned=tuned.mean_K if tuned is not None else np.nan,
                                shared_basis_K_star=shared))
    print(out.table("component_basis_summary.csv", pd.DataFrame(summary)).round(4)
          .to_string(index=False))
    plot_components(B, req, out)


def plot_components(B, req, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for (name, k), s in B.groupby(["dataset", "knee"]):
        s = s.sort_values("budget")
        ax[0].loglog(s.budget, s.tv_mixture, "o-", ms=3, label=f"{name}, $k_\\eta={k}$")
        col = ax[0].lines[-1].get_color()
        ax[0].loglog(s.budget, s.bound_prop4, ":", color=col, lw=0.8)
        ax[1].loglog(s.tv_mixture.clip(lower=1e-4), s.mean_K, "o-", ms=3, color=col)
        r = req[(req.dataset == name) & (req.knee == k)]
        ax[1].loglog(r.eps, r.K_star_int, "s--", ms=3, color=col, alpha=0.6)
    bb = np.geomspace(1e-3, 0.5, 10)
    ax[0].loglog(bb, bb, "k-", lw=0.8)
    ax[0].set(xlabel="per-component budget $b$", ylabel="TV",
              title="mixture TV (solid), $\\sum_m\\pi_m\\varepsilon_m$ (dotted), $b$ (black)")
    ax[0].legend(fontsize=6)
    ax[1].set(xlabel="TV", ylabel="retained directions",
              title="component bases $\\sum_m\\pi_mK_m$ (solid) vs shared $K^\\star$ (dashed)")
    out.figure("component_bases")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", choices=PROFILES, default="paper",
                    help="compute budget (smoke = code test only)")
    ap.add_argument("--data", choices=["cifar", "synthetic"], default="cifar",
                    help="'synthetic' uses dead-leaves images: a code test, not a CIFAR result")
    ap.add_argument("--datasets", nargs="+", choices=["cifar10", "cifar100"],
                    help="override the profile's datasets")
    ap.add_argument("--output", type=Path, default=Path("results"), help="output root")
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="where CIFAR is stored or downloaded")
    ap.add_argument("--skip-checks", action="store_true", help="skip the numerical self-checks")
    args = ap.parse_args(argv)

    cfg = dict(seed=BASE_CFG["seed"], profile=args.profile, data=args.data,
               **{k: v for k, v in BASE_CFG.items() if k != "seed"}, **PROFILES[args.profile])
    if args.datasets:
        cfg["datasets"] = args.datasets
    out = Output(args.output, cfg)
    (out.dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out.dir / "environment.json").write_text(json.dumps(
        dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
             run_hash=out.run_hash), indent=2))
    if args.data != "cifar":
        print("WARNING: synthetic dead-leaves images: a code test, not a CIFAR result.")
    print("Output:", out.dir)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_checks:
        numerical_checks()
    fits, _, comp_df = fit_all(cfg, args.data_dir, out)
    req = shared_basis(fits, comp_df, cfg, out)
    component_bases(fits, req, cfg, out)

    archive = out.dir.parent / f"{out.dir.name}_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.dir.iterdir()):
            if p.is_file():
                z.write(p, p.name)
    print("Saved:", archive)


if __name__ == "__main__":
    main()
