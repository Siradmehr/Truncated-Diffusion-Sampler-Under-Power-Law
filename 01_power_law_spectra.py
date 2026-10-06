"""Experiment 1: covariance spectra of natural images (and, optionally, audio).

Estimates the eigenvalue spectrum of patch covariances and tests whether it
follows a power law sigma_i^2 ~ i^(-beta) (Assumption 1 of the paper).

For every dataset and patch size the script
  1. computes covariance spectra on image-disjoint train / validation / test splits;
  2. fits power-law, exponential and broken-power-law models to the training
     spectrum over a grid of rank intervals, selects the model and interval on
     the validation spectrum, and reports its test error;
  3. reports the best power-law exponent with a bootstrap interval that
     resamples whole images (or recordings), never individual patches;
  4. checks how the stable rank depends on the number of training sources.

CIFAR-10/100 are downloaded through torchvision. Other image datasets
(e.g. DIV2K, BSDS500) and audio datasets (e.g. MAESTRO, FMA) can be added with
--image-root / --audio-root; a CSV manifest with columns path,split,group is
recommended so that official splits and recording/artist groups are respected.

Example:
    python 01_power_law_spectra.py --profile pilot
    python 01_power_law_spectra.py --profile research --cifar cifar10 cifar100
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
import scipy.linalg as sla  # noqa: E402
from PIL import Image  # noqa: E402

EXPERIMENT = "01_power_law_spectra"

PROFILES = {
    "smoke": dict(n_train=256, n_val=128, n_test=128, image_sizes=[4], bootstrap=2,
                  max_sources=10, vectors_per_source=4),
    "pilot": dict(n_train=6000, n_val=2000, n_test=2000, image_sizes=[8, 16], bootstrap=8),
    "research": dict(n_train=45000, n_val=5000, n_test=10000, image_sizes=[8, 16, 32],
                     bootstrap=200, max_sources=500, vectors_per_source=100,
                     audio_sizes=[512, 2048, 8192]),
}

DEFAULTS = dict(
    seed=2027,
    download_cifar=True,
    cifar_datasets=["cifar10"],
    image_roots={},
    audio_roots={},
    manifests={},
    external_image_sizes=[8, 16, 32],
    audio_sizes=[512, 2048],
    audio_seconds_per_source=60,
    vectors_per_source=16,
    max_sources=100,
    patches_per_image=1,
)


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
# Spectra
# ---------------------------------------------------------------------------
def pca(X):
    """Sample mean, eigenvalues (descending), eigenvectors and covariance of X."""
    X = np.asarray(X)
    if len(X) < 2:
        raise ValueError("PCA requires at least two observations.")
    mu = X.mean(0, dtype=np.float64)
    C = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for start in range(0, len(X), 2048):
        Z = np.asarray(X[start:start + 2048], dtype=np.float64) - mu
        C += Z.T @ Z
    C /= len(X) - 1
    v, V = sla.eigh(C, check_finite=False)
    v = np.maximum(v[::-1], 0)
    V = V[:, ::-1].copy()
    return mu, v, V, C


def spectrum(X):
    return pca(X)[1]


def grouped_indices(groups, rng):
    """Bootstrap whole groups (images or recordings), keeping all their observations."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    chosen = rng.choice(unique, len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == g) for g in chosen])


def design(rank, model, breakpoint=None):
    """Design matrix for log(eigenvalue) as a function of rank."""
    z = np.log(rank)
    if model == "power":
        return np.c_[np.ones(len(rank)), z]
    if model == "exponential":
        return np.c_[np.ones(len(rank)), rank]
    return np.c_[np.ones(len(rank)), z, np.maximum(z - np.log(breakpoint), 0)]


def fit_candidates(vtrain, vval):
    """Fit every (model, rank interval) on training eigenvalues; sort by validation error.

    Eigenvalues are not independent samples, so selection is done on a
    held-out spectrum rather than on goodness of fit.
    """
    threshold = max(vtrain[0], vval[0]) * 1e-10
    stable = min(np.count_nonzero(vtrain > threshold), np.count_nonzero(vval > threshold))
    if stable < 12:
        raise ValueError("Too few stable covariance ranks. Increase the number of sources.")
    candidates = []
    for low in (1, 2, 4, 8, 16):
        for high in sorted({stable // 4, stable // 2, 3 * stable // 4}):
            if high - low < 10 or high / low < 4:
                continue
            idx = np.unique(np.rint(np.geomspace(low, high, 40)).astype(int))
            for model in ("power", "exponential", "broken"):
                breaks = [None] if model != "broken" else np.geomspace(low, high, 5)[1:-1]
                for bp in breaks:
                    D = design(idx, model, bp)
                    coef = np.linalg.lstsq(D, np.log(vtrain[idx - 1]), rcond=None)[0]
                    err = np.mean((D @ coef - np.log(vval[idx - 1])) ** 2)
                    candidates.append(dict(model=model, low=low, high=high, breakpoint=bp,
                                           coef=coef, val_log_mse=err, rank_count=len(idx)))
    if not candidates:
        raise ValueError("No eligible fit range.")
    return sorted(candidates, key=lambda r: r["val_log_mse"])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def cifar_arrays(name, n_train, n_val, n_test, seed, data_dir, out, download=True):
    """CIFAR images in [0, 1] with a stratified train/validation split of the training set."""
    import torchvision.datasets as datasets
    from sklearn.model_selection import train_test_split

    cls = datasets.CIFAR10 if name.lower() == "cifar10" else datasets.CIFAR100
    data_dir.mkdir(parents=True, exist_ok=True)
    tr = cls(str(data_dir), train=True, download=download)
    te = cls(str(data_dir), train=False, download=download)
    ids = np.arange(len(tr))
    labels = np.asarray(tr.targets)
    ti, vi = train_test_split(ids, test_size=0.1, random_state=seed, stratify=labels)
    rng = np.random.default_rng(seed)
    ti = rng.permutation(ti)[:n_train]
    vi = rng.permutation(vi)[:n_val]
    ei = rng.permutation(len(te))[:n_test]
    out.json(f"{name}_split_ids.json",
             {"train": ti, "val": vi, "test": ei, "official_train_partition_seed": seed})
    return {
        "train": np.asarray(tr.data[ti], np.float32) / 255,
        "val": np.asarray(tr.data[vi], np.float32) / 255,
        "test": np.asarray(te.data[ei], np.float32) / 255,
        "train_groups": np.array([f"train_{i}" for i in ti]),
        "val_groups": np.array([f"train_{i}" for i in vi]),
        "test_groups": np.array([f"test_{i}" for i in ei]),
    }


def patch_vectors(images, groups, size, per_image=1, seed=0):
    """Random size x size patches, flattened, each tagged with the group of its image."""
    rng = np.random.default_rng(seed)
    xs, gs = [], []
    for image, g in zip(images, groups):
        h, w = image.shape[:2]
        if min(h, w) < size:
            continue
        for _ in range(per_image):
            y = rng.integers(0, h - size + 1)
            x = rng.integers(0, w - size + 1)
            xs.append(image[y:y + size, x:x + size].reshape(-1))
            gs.append(g)
    if not xs:
        raise ValueError("No images large enough for this patch size.")
    return np.asarray(xs, np.float32), np.asarray(gs)


def file_manifest(name, root, kind, seed, out, manifest_csv=None):
    """Table of files with columns path, split, group.

    Uses, in order: a provided manifest; the official MAESTRO split; train/val/test
    directory names; otherwise a random 70/15/15 split by file (flagged as custom).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{name}: dataset root not found: {root}")
    if manifest_csv:
        table = pd.read_csv(manifest_csv)
        if not {"path", "split", "group"}.issubset(table.columns):
            raise ValueError("Manifest requires path, split and group columns.")
        table["path"] = table.path.map(lambda p: str(Path(p) if Path(p).is_absolute() else root / p))
        table["split"] = table.split.replace(
            {"validation": "val", "valid": "val", "training": "train", "testing": "test"})
        method = "provided_manifest"
    else:
        meta = list(root.glob("**/maestro-v*.csv")) if kind == "audio" and "maestro" in name.lower() else []
        if meta:
            md = pd.read_csv(meta[0])
            table = pd.DataFrame({"path": [str(meta[0].parent / p) for p in md.audio_filename],
                                  "split": md.split.replace({"validation": "val"}),
                                  "group": md.audio_filename.astype(str)})
            method = "maestro_official_recording_split"
        else:
            ext = ({".wav", ".mp3", ".flac", ".ogg", ".m4a"} if kind == "audio"
                   else {".png", ".jpg", ".jpeg", ".bmp"})
            paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in ext)
            if not paths:
                raise ValueError(f"No {kind} files found under {root}.")
            alias = {"train": "train", "training": "train", "val": "val", "valid": "val",
                     "validation": "val", "test": "test", "testing": "test"}
            rows = []
            for p in paths:
                parts = [x.lower() for x in p.relative_to(root).parts[:-1]]
                labels = {alias[q] for q in parts if q in alias}
                split = next(iter(labels)) if len(labels) == 1 else None
                rows.append({"path": str(p), "split": split, "group": str(p.relative_to(root))})
            table = pd.DataFrame(rows)
            if table.split.isna().any() or set(table.split) != {"train", "val", "test"}:
                rng = np.random.default_rng(seed)
                ids = rng.permutation(len(table))
                labels = np.full(len(table), "test", dtype=object)
                labels[ids[:int(0.7 * len(ids))]] = "train"
                labels[ids[int(0.7 * len(ids)):int(0.85 * len(ids))]] = "val"
                table["split"] = labels
                method = "custom_source_70_15_15"
            else:
                method = "directory_split"
    if not set(table.split).issubset({"train", "val", "test"}):
        raise ValueError("Invalid split label in manifest.")
    if (table.groupby("group").split.nunique() > 1).any():
        raise ValueError("A source group occurs in more than one split.")
    if table.path.duplicated().any():
        raise ValueError("Duplicate file paths in manifest.")
    missing = table.loc[~table.path.map(lambda p: Path(p).is_file())]
    if len(missing):
        raise FileNotFoundError(f"Missing files, e.g. {missing.iloc[0].path}")
    out.table(f"{name}_manifest.csv", table)
    out.json(f"{name}_split_method.json", {"method": method})
    return table


def source_vectors(table, kind, size, per_source, max_sources, seed, out, audio_seconds=60):
    """Patches (images) or excerpts (audio) from up to max_sources files of one split."""
    rng = np.random.default_rng(seed)
    xs, gs, failed = [], [], []
    chosen = table.iloc[rng.permutation(len(table))[:max_sources]]
    for row in chosen.itertuples():
        try:
            if kind == "image":
                arr = np.asarray(Image.open(row.path).convert("RGB"), np.float32) / 255
                xx, gg = patch_vectors([arr], [row.group], size, per_source,
                                       seed=int(rng.integers(2**31)))
            else:
                import librosa
                import soundfile as sf

                try:
                    duration = sf.info(row.path).duration
                except Exception:
                    duration = librosa.get_duration(path=row.path)
                # A random bounded-length segment rather than always the opening.
                offset = float(rng.uniform(0, max(0, duration - audio_seconds)))
                y, _ = librosa.load(row.path, sr=16000, mono=True, offset=offset,
                                    duration=audio_seconds)
                if len(y) < size:
                    raise ValueError("recording too short")
                starts = rng.integers(0, len(y) - size + 1, size=per_source)
                xx = np.stack([y[j:j + size] for j in starts])
                gg = np.full(len(xx), str(row.group))
            xs.extend(xx)
            gs.extend(gg)
        except Exception as e:  # keep going, but record every unreadable file
            failed.append({"path": row.path, "error": str(e)})
    if failed:
        tag = hashlib.sha256(str(chosen.path.tolist()).encode()).hexdigest()[:8]
        out.table(f"read_failures_{tag}.csv", failed)
    if not xs:
        raise ValueError("No readable observations; see the read_failures CSV.")
    return np.asarray(xs, np.float32), np.asarray(gs)


def real_cases(cfg, data_dir, out):
    """Yield one (dataset, size) case at a time to bound memory."""
    seed = cfg["seed"]
    for name in cfg["cifar_datasets"]:
        a = cifar_arrays(name, cfg["n_train"], cfg["n_val"], cfg["n_test"], seed, data_dir, out,
                         cfg["download_cifar"])
        for size in cfg["image_sizes"]:
            case = {"name": f"{name}_rgb_{size}", "kind": "image", "shape": (size, size, 3)}
            for j, split in enumerate(("train", "val", "test")):
                X, g = patch_vectors(a[split], a[f"{split}_groups"], size,
                                     cfg["patches_per_image"], seed + j)
                case[split] = X
                case[f"{split}_groups"] = g
            yield case
    for kind, roots, sizes in [("image", cfg["image_roots"], cfg["external_image_sizes"]),
                               ("audio", cfg["audio_roots"], cfg["audio_sizes"])]:
        for name, root in roots.items():
            table = file_manifest(name, root, kind, seed, out, cfg["manifests"].get(name))
            for size in sizes:
                shape = (size, size, 3) if kind == "image" else (size,)
                case = {"name": f"{name}_{size}", "kind": kind, "shape": shape}
                for j, split in enumerate(("train", "val", "test")):
                    X, g = source_vectors(table[table.split == split], kind, size,
                                          cfg["vectors_per_source"], cfg["max_sources"], seed + j,
                                          out, cfg["audio_seconds_per_source"])
                    case[split] = X
                    case[f"{split}_groups"] = g
                yield case


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyse_case(case, cfg, out):
    """Fit, select and bootstrap one spectrum; returns the summary row and all candidate rows."""
    seed = cfg["seed"]
    name = case["name"]
    vs = {split: spectrum(case[split]) for split in ("train", "val", "test")}
    candidates = fit_candidates(vs["train"], vs["val"])
    best = candidates[0]
    # The best power law is reported even when another model wins.
    power = next(c for c in candidates if c["model"] == "power")

    rng = np.random.default_rng(seed)
    idx = np.unique(np.rint(np.geomspace(power["low"], power["high"], 40)).astype(int))
    boot = []
    for _ in range(cfg["bootstrap"]):
        sample = grouped_indices(case["train_groups"], rng)
        vb = spectrum(case["train"][sample])
        coef = np.linalg.lstsq(design(idx, "power"), np.log(np.maximum(vb[idx - 1], 1e-30)),
                               rcond=None)[0]
        boot.append(-coef[1])

    fit_rows = []
    for c in candidates:
        ranks = np.unique(np.rint(np.geomspace(c["low"], c["high"], 40)).astype(int))
        pred = design(ranks, c["model"], c["breakpoint"]) @ c["coef"]
        err = np.mean((pred - np.log(np.maximum(vs["test"][ranks - 1], 1e-30))) ** 2)
        fit_rows.append({**c, "coef": json.dumps(c["coef"].tolist()), "case": name,
                         "test_log_mse": err})

    lo, hi = np.quantile(boot, [0.025, 0.975]) if boot else (np.nan, np.nan)
    summary = {"case": name, "kind": case["kind"], "d": len(vs["train"]),
               "n_train": len(case["train"]),
               "sources_train": len(np.unique(case["train_groups"])),
               "best_model": best["model"], "power_beta": -power["coef"][1],
               "beta_lo": lo, "beta_hi": hi, "power_rank_low": power["low"],
               "power_rank_high": power["high"], "bootstrap": len(boot)}
    np.savez_compressed(out.dir / f"{name}_spectra.npz", **vs, beta_bootstrap=boot)
    out.json(f"{name}_groups.json", {s: case[f"{s}_groups"] for s in ("train", "val", "test")})

    # Spectra and local decay exponent.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for split, v in vs.items():
        axs[0].loglog(np.arange(1, len(v) + 1), np.maximum(v, 1e-20), label=split)
    ranks = np.arange(best["low"], best["high"] + 1)
    axs[0].loglog(ranks, np.exp(design(ranks, best["model"], best["breakpoint"]) @ best["coef"]),
                  "--", label="validation-selected " + best["model"])
    axs[0].set(xlabel="Covariance rank", ylabel="Eigenvalue", title=name)
    axs[0].legend()
    ids = np.unique(np.rint(np.geomspace(1, max(3, best["high"]), 40)).astype(int))
    slope = -np.gradient(np.log(np.maximum(vs["train"][ids - 1], 1e-30)), np.log(ids))
    axs[1].semilogx(ids, slope)
    axs[1].axhline(1, color="gray", ls=":")
    axs[1].set(xlabel="Rank", ylabel="Local decay exponent")
    out.figure(f"{name}_spectrum")

    # Stable rank against the number of independent training sources.
    learning = []
    groups = np.unique(case["train_groups"])
    order = rng.permutation(groups)
    for frac in (0.25, 0.5, 1.0):
        chosen = order[:max(2, int(len(order) * frac))]
        mask = np.isin(case["train_groups"], chosen)
        vv = spectrum(case["train"][mask])
        tail_rank = min(power["high"], np.count_nonzero(vv > vv[0] * 1e-10))
        learning.append({"fraction": frac, "sources": len(chosen), "vectors": int(mask.sum()),
                         "stable_rank": tail_rank})
    out.table(f"{name}_sample_stability.csv", learning)
    return summary, fit_rows


def write_archive(out):
    """Package versions, file inventory and a ZIP of all outputs up to 50 MiB each."""
    versions = {}
    for p in ("numpy", "scipy", "pandas", "torch", "torchvision", "librosa", "scikit-learn"):
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


def parse_pairs(items):
    """Turn ['name=path', ...] into {'name': 'path', ...}."""
    pairs = {}
    for item in items or []:
        name, sep, path = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {item!r}")
        pairs[name] = path
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", choices=PROFILES, default="pilot",
                    help="compute budget (smoke = code test only)")
    ap.add_argument("--output", type=Path, default=Path("results"), help="output root")
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="where CIFAR is stored or downloaded")
    ap.add_argument("--cifar", nargs="*", default=None,
                    help="CIFAR datasets to analyse (default: cifar10; pass none to skip)")
    ap.add_argument("--image-root", action="append", metavar="NAME=PATH",
                    help="extra image dataset, repeatable")
    ap.add_argument("--audio-root", action="append", metavar="NAME=PATH",
                    help="audio dataset, repeatable (needs librosa and soundfile)")
    ap.add_argument("--manifest", action="append", metavar="NAME=CSV",
                    help="CSV with path,split,group for a dataset, repeatable")
    ap.add_argument("--no-download", action="store_true", help="do not download CIFAR")
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = ap.parse_args(argv)

    cfg = dict(DEFAULTS)
    cfg.update(PROFILES[args.profile])
    cfg.update(profile=args.profile, seed=args.seed, download_cifar=not args.no_download,
               image_roots=parse_pairs(args.image_root), audio_roots=parse_pairs(args.audio_root),
               manifests=parse_pairs(args.manifest))
    if args.cifar is not None:
        cfg["cifar_datasets"] = [] if args.cifar in ([], ["none"]) else args.cifar

    out = Output(args.output, cfg)
    out.json("config.json", cfg)
    out.json("environment.json", {"python": sys.version, "platform": platform.platform(),
                                  "numpy": np.__version__, "profile": args.profile,
                                  "run_hash": out.run_hash})
    print("Output:", out.dir, "| profile:", args.profile)

    status = [{"dataset": n, "status": "configured" if n in cfg["image_roots"] or n in cfg["audio_roots"]
               else "not_configured"} for n in ("div2k", "bsds500", "maestro", "fma")]
    out.table("dataset_status.csv", status)

    summary, fit_rows = [], []
    for case in real_cases(cfg, args.data_dir, out):
        print("Analysing", case["name"], case["train"].shape)
        row, rows = analyse_case(case, cfg, out)
        summary.append(row)
        fit_rows.extend(rows)
        out.table("spectrum_summary.csv", summary)
        out.table("all_fit_candidates.csv", fit_rows)
    if summary:
        print(pd.DataFrame(summary).to_string(index=False))
    print("Archive:", write_archive(out))


if __name__ == "__main__":
    main()
