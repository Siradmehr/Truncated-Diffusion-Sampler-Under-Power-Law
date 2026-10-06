"""Experiment 2: truncation of Gaussians with an exact power-law spectrum.

Synthetic test of Theorems 1 and 2. The data law is N(0, diag(sigma_i^2)) with
sigma_i^2 = i^(-beta), i = 1..d, and the evaluation level eta is set so that
exactly `knee` directions lie above it (eta^2 = knee^(-beta)). For every
(beta, knee) the script

  1. estimates the truncation error TV(q_K^eta, q^eta) for a grid of K, refined
     around every target accuracy, and checks it against the two-sided bound
     G_K/100 <= TV <= 3/2 G_K of Theorem 1(a) (G_K = min{1, ||g_{>K}||_2}) and
     against the explicit KL/Pinsker bound;
  2. compares rules for choosing K (noise level, variance fractions, the
     Pinsker tail budget) with the smallest K that meets each accuracy;
  3. fits the growth of that K in 1/epsilon and in 1/eta and compares the
     slopes with the exponents 2/(2 beta - 1) and 4/(2 beta - 1) of Theorem 2.

TV between diagonal Gaussians is estimated by Monte Carlo from the exact
densities, using TV(P, Q) = E_{(P+Q)/2} |tanh(log(dP/dQ) / 2)|, with common
random numbers across K. No dataset is needed.

Example:
    python 02_gaussian_truncation.py --profile pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from importlib import metadata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

EXPERIMENT = "02_gaussian_truncation"

PROFILES = {
    "smoke": dict(d=128, betas=[1.5], knees=[1, 4], budgets=[0.2, 0.1, 0.05], mc_draws=512,
                  k_points=10),
    "pilot": dict(d=4096, betas=[1.25, 1.5, 2.0], knees=[1, 4], budgets=[0.2, 0.1, 0.05, 0.02],
                  mc_draws=8000, k_points=30),
    "research": dict(d=65536, betas=[1.25, 1.5, 2.0], knees=[1, 4, 16],
                     budgets=[0.2, 0.1, 0.05, 0.02, 0.004], mc_draws=20000, k_points=60),
}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
class Output:
    """Writes tables, JSON and figures to one run directory."""

    def __init__(self, base: Path, cfg: dict):
        run_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
        self.dir = base / EXPERIMENT / f"{cfg['profile']}_{run_hash}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_hash = run_hash

    def json(self, name, obj):
        def default(x):
            return x.tolist() if isinstance(x, np.ndarray) else str(x)

        (self.dir / name).write_text(json.dumps(obj, indent=2, default=default))

    def table(self, name, rows):
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        df.to_csv(self.dir / name, index=False)
        return df

    def figure(self, name):
        plt.tight_layout()
        plt.savefig(self.dir / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Gaussian truncation
# ---------------------------------------------------------------------------
def suffix_sum(a):
    """s[K] = sum_{i >= K} a[i] for K = 0..len(a), with s[len(a)] = 0."""
    return np.r_[np.cumsum(a[::-1])[::-1], 0.0]


def tail_scores(v, eta):
    """Wiener-gain tail ||g_{>K}||_2 and the KL/Pinsker bound sqrt(KL/2), for every K.

    With x_i = sigma_i^2 / eta^2 and g_i = x_i / (1 + x_i),
    KL(q_K^eta || q^eta) = 1/2 sum_{i>K} [log(1 + x_i) - g_i].
    """
    if eta <= 0:
        raise ValueError("The noisy-tail formula requires eta > 0.")
    x = np.asarray(v, dtype=np.float64) / eta**2
    g = x / (1 + x)
    # Stable log(1+x) - x/(1+x) for very weak directions (Taylor series).
    terms = np.empty_like(x)
    small = x < 1e-3
    y = x[small]
    terms[small] = y * y * (0.5 - 2 * y / 3 + 3 * y * y / 4 - 4 * y**3 / 5)
    terms[~small] = np.log1p(x[~small]) - g[~small]
    return np.sqrt(suffix_sum(g * g)), np.sqrt(np.maximum(suffix_sum(terms), 0) / 4)


def selectors(v, eta, budgets=(0.05, 0.1, 0.2)):
    """Values of K chosen by each rule, computed from the spectrum alone."""
    v = np.asarray(v)
    d = len(v)
    _, pinsker = tail_scores(v, eta)
    result = {"full": d, "noise_threshold": int(np.count_nonzero(v >= eta**2))}
    total = v.sum()
    for rho in (0.95, 0.99, 0.999):
        result[f"variance_{rho}"] = (0 if total == 0 else
                                     min(d, int(np.searchsorted(np.cumsum(v), rho * total) + 1)))
    for tau in budgets:
        result[f"tail_{tau}"] = int(np.flatnonzero(pinsker <= tau)[0])
    return result


def mean_ci(s, ss, n):
    """Mean, 95% normal interval clipped to [0, 1], and standard error from running sums."""
    m = s / n
    se = np.sqrt(np.maximum((ss - n * m * m) / max(n - 1, 1), 0) / n)
    return m, np.maximum(0, m - 1.96 * se), np.minimum(1, m + 1.96 * se), se


def diagonal_tv(a, b, n=10000, seed=0, batch=128, chunk=1024):
    """Monte Carlo TV(N(0, diag a), N(0, diag b)) from the exact log-likelihood ratio."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Positive, equal-sized variance arrays required.")
    mask = a != b
    a, b = a[mask], b[mask]
    if not len(a):
        return dict(tv=0.0, lo=0.0, hi=0.0, se=0.0, n=n)
    rng = np.random.default_rng(seed)
    sm = ss = 0.0
    for start in range(0, n, batch):
        m = min(batch, n - start)
        coin = rng.integers(0, 2, size=(m, 1))           # draw from P or from Q
        ell = np.full(m, -0.5 * np.log(a / b).sum())
        for j in range(0, len(a), chunk):
            aa, bb = a[j:j + chunk], b[j:j + chunk]
            z = rng.standard_normal((m, len(aa)))
            var = np.where(coin, aa, bb)
            ell -= 0.5 * np.sum(z * z * var * (1 / aa - 1 / bb), axis=1)
        val = np.abs(np.tanh(ell / 2))
        sm += val.sum()
        ss += val @ val
    m, lo, hi, se = mean_ci(sm, ss, n)
    return dict(tv=float(m), lo=float(lo), hi=float(hi), se=float(se), n=n)


def truncation_curve(v, eta, ks, n=10000, seed=0, batch=64, chunk=1024):
    """TV(q_K^eta, q^eta) for every K in ks, in one common-random-number pass.

    The truncated law replaces variance v_i + eta^2 by eta^2 for i > K, so the
    log-likelihood ratio is a suffix sum over directions; cost O(n d).
    """
    v = np.asarray(v, float)
    ks = np.asarray(sorted(set(ks)), int)
    d = len(v)
    if np.any((ks < 0) | (ks > d)):
        raise ValueError("Invalid K")
    rng = np.random.default_rng(seed)
    sm = np.zeros(len(ks))
    ss = sm.copy()
    a = eta**2
    for start in range(0, n, batch):
        m = min(batch, n - start)
        coin = rng.integers(0, 2, size=(m, 1))
        ell = np.zeros((m, len(ks)))
        carry = np.zeros(m)
        for end in range(d, 0, -chunk):                   # chunks from the last direction back
            begin = max(0, end - chunk)
            vv = v[begin:end]
            b = a + vv
            z = rng.standard_normal((m, len(vv)))
            var = np.where(coin, b, a)
            terms = 0.5 * (np.log1p(vv / a) - z * z * var * (vv / (a * b)))
            suffix = np.cumsum(terms[:, ::-1], axis=1)[:, ::-1]
            ix = np.flatnonzero((ks >= begin) & (ks < end))
            if len(ix):
                ell[:, ix] = carry[:, None] + suffix[:, ks[ix] - begin]
            carry += suffix[:, 0]
        values = np.abs(np.tanh(ell / 2))
        sm += values.sum(0)
        ss += (values * values).sum(0)
    m, lo, hi, se = mean_ci(sm, ss, n)
    return pd.DataFrame({"K": ks, "tv": m, "lo": lo, "hi": hi, "se": se, "n": n})


def numerical_checks():
    """Sanity checks of the estimators against known values."""
    assert diagonal_tv([1, 2], [1, 2], n=32)["tv"] == 0
    v = np.array([4.0, 1.0, 0.2])
    S, B = tail_scores(v, 0.5)
    assert S[-1] == B[-1] == 0 and np.all(np.diff(B) <= 0)
    # Closed form for N(0, 1) vs N(0, 4): the densities cross at +-c.
    c = np.sqrt(2 * np.log(2) / (1 - 1 / 4))
    exact = 2 * (stats.norm.cdf(c) - stats.norm.cdf(c / 2))
    got = diagonal_tv([1], [4], n=12000, seed=19)
    assert abs(got["tv"] - exact) < max(0.012, 5 * got["se"])
    curve = truncation_curve(v, 0.5, [0, 1, 3], n=5000, seed=11)
    assert curve.iloc[-1].tv == 0
    print("Numerical checks passed.")


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_case(beta, knee, cfg, out):
    """Truncation curve and rule calibration for one (beta, knee)."""
    seed = cfg["seed"]
    v = np.arange(1, cfg["d"] + 1, dtype=float) ** (-beta)
    eta = np.sqrt(knee ** (-beta))
    S, B = tail_scores(v, eta)
    methods = selectors(v, eta, cfg["budgets"])

    # Coarse log-spaced grid, then a finer grid around every accuracy crossing.
    ks = (set(np.rint(np.geomspace(1, len(v), cfg["k_points"])).astype(int))
          | set(methods.values()) | {0, len(v)})
    coarse = truncation_curve(v, eta, ks, cfg["mc_draws"], seed)
    refined = set(ks)
    for tau in cfg["budgets"]:
        yes = coarse[coarse.tv <= tau]
        if len(yes):
            right = int(yes.iloc[0].K)
            left = int(coarse[coarse.K < right].K.max()) if right > 0 else 0
            refined.update(np.rint(np.linspace(left, right, 9)).astype(int))
    curve = truncation_curve(v, eta, refined, cfg["mc_draws"], seed + 1)
    curve["beta"] = beta
    curve["eta"] = eta
    curve["knee"] = knee
    curve["d"] = len(v)
    kval = curve.K.to_numpy(int)
    curve["G"] = np.minimum(1, S[kval])
    curve["B"] = np.minimum(1, B[kval])
    curve["bound_low"] = curve.G / 100
    curve["bound_high"] = np.minimum(1, 1.5 * curve.G)
    curve["possible_bound_violation"] = ((curve.hi < curve.bound_low - 1e-8)
                                         | (curve.lo > curve.bound_high + 1e-8)
                                         | (curve.lo > curve.B + 1e-8))

    choices = []
    for tau in cfg["budgets"]:
        possible = curve[curve.lo <= tau]
        confirmed = curve[curve.hi <= tau]
        klo = int(possible.iloc[0].K) if len(possible) else len(v)
        khi = int(confirmed.iloc[0].K) if len(confirmed) else len(v)
        for method, K in methods.items():
            if method.startswith("tail_") and method != f"tail_{tau}":
                continue
            check = diagonal_tv(np.full(len(v) - K, eta**2), v[K:] + eta**2, cfg["mc_draws"],
                                seed + 7)
            choices.append({"beta": beta, "knee": knee, "eta": eta, "d": len(v), "tau": tau,
                            "method": method, "K": K, **check,
                            "grid_oracle_possible": klo, "grid_oracle_confirmed": khi,
                            "full_retention": K == len(v),
                            "asymptotic_candidate": knee <= K <= len(v) // 8})

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].loglog(curve.K + 1, np.maximum(curve.tv, 1e-8), "o-", label="TV estimate")
    ax[0].fill_between(curve.K + 1, np.maximum(curve.lo, 1e-8), np.maximum(curve.hi, 1e-8),
                       alpha=0.2)
    ax[0].loglog(curve.K + 1, np.maximum(curve.B, 1e-8), "--", label="KL/Pinsker upper bound")
    ax[0].axvline(knee + 1, color="gray", ls=":")
    ax[0].set(xlabel="K + 1", ylabel="TV", title=f"beta={beta}, knee={knee}")
    ax[0].legend()
    ax[1].plot(curve.G, curve.tv, "o")
    ax[1].set(xlabel="min(1, aggregate Wiener tail)", ylabel="TV")
    out.figure(f"truncation_beta{beta}_knee{knee}")
    return curve, choices


def scaling_fits(cal):
    """Slopes of log K* in log epsilon and in log eta, against Theorem 2.

    Only points with 2 knee <= K* <= d/8 are used, away from the noise level and
    from saturation at finite d.
    """
    oracle = cal[["beta", "knee", "tau", "grid_oracle_confirmed", "d"]].drop_duplicates()
    accuracy = []
    for (beta, knee), g in oracle.groupby(["beta", "knee"]):
        g = g[(g.grid_oracle_confirmed >= 2 * knee) & (g.grid_oracle_confirmed <= g.d / 8)]
        row = {"beta": beta, "knee": knee, "predicted_slope": -2 / (2 * beta - 1),
               "points": len(g)}
        if g.tau.nunique() >= 3:
            row["measured_slope"] = np.polyfit(np.log(g.tau), np.log(g.grid_oracle_confirmed), 1)[0]
        else:
            row.update(measured_slope=np.nan, status="insufficient_asymptotic_points")
        accuracy.append(row)
    resolution = []
    for (beta, tau), g in oracle.groupby(["beta", "tau"]):
        g = g[(g.grid_oracle_confirmed >= 2 * g.knee) & (g.grid_oracle_confirmed <= g.d / 8)]
        measured = np.nan
        if g.knee.nunique() >= 3:
            eta = np.sqrt(g.knee.to_numpy(float) ** (-beta))
            measured = np.polyfit(np.log(eta), np.log(g.grid_oracle_confirmed), 1)[0]
        resolution.append({"beta": beta, "tau": tau, "measured_eta_slope": measured,
                           "predicted_eta_slope": -4 / (2 * beta - 1), "points": len(g)})
    return pd.DataFrame(accuracy), pd.DataFrame(resolution)


def write_archive(out):
    """Package versions, file inventory and a ZIP of all outputs up to 50 MiB each."""
    versions = {}
    for p in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    out.json("package_versions.json", versions)
    files = [p for p in out.dir.rglob("*") if p.is_file()]
    out.table("output_inventory.csv",
              [{"file": str(p.relative_to(out.dir)), "bytes": p.stat().st_size} for p in files])
    archive = out.dir.parent / f"{out.dir.name}_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.dir.rglob("*"):
            if p.is_file() and p.stat().st_size <= 50 * 1024**2:
                z.write(p, str(p.relative_to(out.dir)))
    return archive


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", choices=PROFILES, default="pilot",
                    help="compute budget (smoke = code test only)")
    ap.add_argument("--output", type=Path, default=Path("results"), help="output root")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args(argv)

    cfg = {"seed": args.seed, **PROFILES[args.profile], "profile": args.profile}
    out = Output(args.output, cfg)
    out.json("config.json", cfg)
    out.json("environment.json", {"python": sys.version, "platform": platform.platform(),
                                  "numpy": np.__version__, "profile": args.profile,
                                  "run_hash": out.run_hash})
    print("Output:", out.dir, "| profile:", args.profile)

    numerical_checks()
    rows, choices = [], []
    for beta in cfg["betas"]:
        for knee in cfg["knees"]:
            curve, ch = run_case(beta, knee, cfg, out)
            rows.extend(curve.to_dict("records"))
            choices.extend(ch)
            out.table("truncation_curves.csv", rows)
            out.table("selector_calibration.csv", choices)
            print(f"beta={beta}, knee={knee}: done")

    accuracy, resolution = scaling_fits(pd.DataFrame(choices))
    print(out.table("accuracy_scaling.csv", accuracy).to_string(index=False))
    print(out.table("resolution_scaling.csv", resolution).to_string(index=False))
    print("Archive:", write_archive(out))


if __name__ == "__main__":
    main()
