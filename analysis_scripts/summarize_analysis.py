"""Summarize analysis outputs and compare against reference data."""

import argparse
import math
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FES_BINS = 20
FES_PSEUDOCOUNT = 1.0
FES_NCOLS = 6


def parse_args():
    """Parse CLI arguments for the summary script."""
    parser = argparse.ArgumentParser(
        description="Summarize RMSD/contact maps and compare to references.",
    )
    parser.add_argument(
        "--root",
        default=os.getcwd(),
        help="Project root containing analysis_sys/ and analysis_ref/.",
    )
    parser.add_argument(
        "--analysis-dir",
        default="analysis_sys",
        help="Directory with simulation analysis outputs.",
    )
    parser.add_argument(
        "--analysis-ref",
        default="analysis_ref",
        help="Directory with reference analysis outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots and summary CSV (defaults to analysis-dir).",
    )
    parser.add_argument(
        "--drop-fraction",
        type=float,
        default=0.25,
        help="Fraction of initial RMSD data to drop before averaging.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.0,
        help="Threshold for binarizing contact maps before Jaccard.",
    )
    return parser.parse_args()


def load_run_pipeline_vars(path: Path):
    """Parse simple KEY=VALUE pairs from run_pipeline.sh."""
    if not path.exists():
        return {}
    vars_out = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "#" in stripped:
                stripped = stripped.split("#", 1)[0].strip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key.isidentifier():
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            vars_out[key] = value
    return vars_out


def maybe_run_get_motif_samples(root: Path):
    """Run getMotifSamples.py using run_pipeline.sh params if motif_pdbs.csv missing."""
    run_pipeline = root / "run_pipeline.sh"
    config = load_run_pipeline_vars(run_pipeline)
    if not config:
        return

    out_dir_cfg = config.get("OUT_DIR", "./")
    out_dir = Path(out_dir_cfg)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    motif_csv = out_dir / "motif_pdbs" / "motif_pdbs.csv"
    if motif_csv.exists():
        return

    eval_dirs_cfg = config.get("EVAL_DIRS") or config.get("EVAL_DIR")
    if not eval_dirs_cfg:
        print("motif_pdbs.csv missing, but EVAL_DIRS not set in run_pipeline.sh.")
        return

    eval_dirs_arg = eval_dirs_cfg
    if not eval_dirs_arg.lstrip().startswith("["):
        eval_dir = Path(eval_dirs_arg)
        if not eval_dir.is_absolute():
            eval_dir = (root / eval_dir).resolve()
        eval_dirs_arg = str(eval_dir)

    include_failures = config.get("INCLUDE_FAILURES", "0").lower() in {"1", "true", "yes"}

    cmd = [sys.executable, str(root / "getMotifSamples.py"), eval_dirs_arg, str(out_dir)]
    if include_failures:
        cmd.append("--include-failures")

    print("motif_pdbs.csv not found; running getMotifSamples.py from run_pipeline.sh.")
    subprocess.run(cmd, cwd=str(root), check=False)


def ensure_projection_pickle_support():
    """Import dependencies required to unpickle reference PCA/TICA models."""
    try:
        import deeptime.decomposition  # noqa: F401
        from deeptime.decomposition import TICA  # noqa: F401
        from sklearn.decomposition import PCA  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Projection analysis requires deeptime and scikit-learn to load "
            "analysis_ref/results_tica-pca_base.pkl."
        ) from exc


class NumPyCompatUnpickler(pickle.Unpickler):
    """Load NumPy 2.x pickles when running with NumPy 1.x."""

    MODULE_ALIASES = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
    }

    def find_class(self, module, name):
        module = self.MODULE_ALIASES.get(module, module)
        return super().find_class(module, name)


def load_pickle(path):
    """Load a pickle file from disk with NumPy compatibility aliases."""
    with path.open("rb") as handle:
        return NumPyCompatUnpickler(handle).load()


def as_array(value):
    """Convert value to a NumPy array if possible."""
    if value is None:
        return None
    return np.asarray(value)


def parse_rmsd_entry(entry):
    """Normalize an RMSD entry into a dict with time/rmsd arrays."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        rmsd = None
        for key in ("rmsd_A", "rmsd", "rmsd_values"):
            if key in entry:
                rmsd = as_array(entry[key])
                break
        if rmsd is None:
            return None
        time = None
        for key in ("time_ps", "time", "t"):
            if key in entry:
                time = as_array(entry[key])
                break
        return {"rmsd": rmsd, "time": time}

    return {"rmsd": as_array(entry), "time": None}


def load_ref_rmsd(path):
    """Load reference RMSD dicts supporting multiple pickle formats."""
    if not path.exists():
        return {}
    data = load_pickle(path)
    if not isinstance(data, dict):
        return {}

    if any(str(key).endswith("_t") for key in data):
        parsed = {}
        for key, value in data.items():
            if str(key).endswith("_t") or str(key).endswith("_normed"):
                continue
            parsed[key] = {
                "rmsd": as_array(value),
                "time": as_array(data.get(f"{key}_t")),
            }
        return parsed

    parsed = {}
    for key, value in data.items():
        entry = parse_rmsd_entry(value)
        if entry is not None:
            parsed[key] = entry
    return parsed


def load_sim_rmsd(analysis_dir):
    """Load simulation RMSD per-system pickles."""
    rmsd = {}
    for path in sorted(analysis_dir.glob("*_rmsd.pkl")):
        payload = load_pickle(path)
        system = payload.get("system") if isinstance(payload, dict) else None
        if not system:
            system = path.stem.replace("_rmsd", "")

        scaffold = None
        motif = None
        if isinstance(payload, dict):
            scaffold = parse_rmsd_entry(payload.get("scaffold"))
            motif = parse_rmsd_entry(payload.get("motif"))

        rmsd[system] = {"scaffold": scaffold, "motif": motif}
    return rmsd


def load_contactmaps_ref(ref_dir):
    """Load reference contact maps keyed by system name."""
    contactmaps = {}
    patterns = ["*_ref_motif_contactmap.npy", "*__ref_motif_contactmap.npy"]
    for pattern in patterns:
        for path in ref_dir.glob(pattern):
            name = path.name
            system = name.replace("__ref_motif_contactmap.npy", "")
            system = system.replace("_ref_motif_contactmap.npy", "")
            if system in contactmaps:
                continue
            contactmaps[system] = np.load(path)
    return contactmaps


def load_contactmaps_sim(analysis_dir):
    """Load simulation contact maps keyed by system name."""
    contactmaps = {}
    for path in analysis_dir.glob("*_motif_contactmap.npy"):
        system = path.stem.replace("_motif_contactmap", "")
        contactmaps[system] = np.load(path)
    return contactmaps


def load_projection_results(path, requires_model_support=False):
    """Load PCA/TICA projection results keyed by system name."""
    if not path.exists():
        return {}

    if requires_model_support:
        try:
            ensure_projection_pickle_support()
        except ImportError as exc:
            print(f"Skipping projection load for {path}: {exc}")
            return {}

    data = load_pickle(path)
    if not isinstance(data, dict):
        return {}

    parsed = {}
    for system, entry in data.items():
        if not isinstance(entry, dict):
            continue
        parsed_entry = {}
        for key in ("Y_tica", "Y_pca"):
            value = as_array(entry.get(key))
            if value is None:
                continue
            parsed_entry[key] = value
        if parsed_entry:
            parsed[system] = parsed_entry
    return parsed


def compute_stats(values, drop_fraction):
    """Compute mean, std, and count after dropping an initial fraction."""
    if values is None:
        return float("nan"), float("nan"), 0
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    start = int(arr.size * drop_fraction)
    if start >= arr.size:
        start = 0
    trimmed = arr[start:]
    trimmed = trimmed[np.isfinite(trimmed)]
    if trimmed.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(trimmed)), float(np.std(trimmed)), int(trimmed.size)


def mean_similarity_from_std(ref_mean, sim_mean, ref_std, sim_std):
    """Compute similarity based on mean difference normalized by std."""
    if not all(np.isfinite([ref_mean, sim_mean, ref_std, sim_std])):
        return float("nan")
    denom = math.sqrt(ref_std ** 2 + sim_std ** 2)
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    z_score = abs(ref_mean - sim_mean) / denom
    return float(1.0 / (1.0 + z_score))


def jaccard_similarity(ref_map, sim_map, threshold):
    """Compute Jaccard similarity on contact maps (excluding diagonal)."""
    if ref_map is None or sim_map is None:
        return float("nan")
    if ref_map.shape != sim_map.shape:
        return float("nan")

    n = ref_map.shape[0]
    mask = np.triu_indices(n, k=1)
    ref_bin = ref_map > threshold
    sim_bin = sim_map > threshold

    ref_vals = ref_bin[mask]
    sim_vals = sim_bin[mask]

    union = np.logical_or(ref_vals, sim_vals)
    if not np.any(union):
        return float("nan")
    inter = np.logical_and(ref_vals, sim_vals)
    j_sim = float(np.sum(inter) / np.sum(union))
    return j_sim


def pearson_similarity(ref_map, sim_map):
    """Compute Pearson correlation on contact maps (excluding diagonal)."""
    if ref_map is None or sim_map is None:
        return float("nan")
    if ref_map.shape != sim_map.shape:
        return float("nan")

    n = ref_map.shape[0]
    mask = np.triu_indices(n, k=1)
    ref_vals = ref_map[mask].astype(float)
    sim_vals = sim_map[mask].astype(float)
    valid = np.isfinite(ref_vals) & np.isfinite(sim_vals)
    if np.sum(valid) < 2:
        return float("nan")

    ref_vals = ref_vals[valid]
    sim_vals = sim_vals[valid]
    if np.all(ref_vals == ref_vals[0]) or np.all(sim_vals == sim_vals[0]):
        return float("nan")

    return float(np.corrcoef(ref_vals, sim_vals)[0, 1])


def contact_absdiff_metrics(ref_map, sim_map):
    """Compute sum of absolute differences (upper triangle, excl diagonal)."""
    if ref_map is None or sim_map is None:
        return float("nan"), float("nan")
    if ref_map.shape != sim_map.shape:
        return float("nan"), float("nan")

    n = ref_map.shape[0]
    if n == 0:
        return float("nan"), float("nan")

    mask = np.triu_indices(n, k=1)
    diff = np.abs(sim_map - ref_map)
    values = diff[mask]
    if values.size == 0:
        return float("nan"), float("nan")
    sum_abs = float(np.sum(values))
    sum_weighted = float(sum_abs / n)
    return sum_abs, sum_weighted


def _problem_sort_key(problem):
    """Sort problem identifiers numerically when possible."""
    try:
        return (0, int(str(problem).split("_")[0]), str(problem))
    except Exception:
        return (1, str(problem))


def _projection_label(key):
    """Return a human-readable label for a projection key."""
    return key.replace("Y_", "").upper()


def _valid_projection_array(results, problem, key):
    """Return a cleaned 2D projection array for a problem/key pair."""
    entry = results.get(problem)
    if not isinstance(entry, dict):
        return None
    value = entry.get(key)
    if value is None:
        return None

    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return None

    arr = arr[:, :2]
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    if not np.any(valid):
        return None
    return arr[valid]


def projection_problems(ref_results, sim_results, key):
    """Return overlapping systems with valid ref/sim projections."""
    problems = []
    for problem in sorted(set(ref_results) & set(sim_results), key=_problem_sort_key):
        if _valid_projection_array(ref_results, problem, key) is None:
            continue
        if _valid_projection_array(sim_results, problem, key) is None:
            continue
        problems.append(problem)
    return problems


def _compute_projection_limits(ref_results, sim_results, key, problems, q=(0.5, 99.5), pad=0.05):
    """Compute shared plot limits from ref+sim coordinates."""
    xs = []
    ys = []
    for results in (ref_results, sim_results):
        for problem in problems:
            Y = _valid_projection_array(results, problem, key)
            if Y is None:
                continue
            xs.append(Y[:, 0])
            ys.append(Y[:, 1])

    if not xs or not ys:
        raise ValueError(f"No valid projection data found for {key}.")

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    xmin, xmax = np.percentile(x, q)
    ymin, ymax = np.percentile(y, q)

    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0:
        dx = max(abs(xmin), abs(xmax), 1.0)
    if dy <= 0:
        dy = max(abs(ymin), abs(ymax), 1.0)

    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy

    return (xmin, xmax), (ymin, ymax)


def _fes_from_xy(x, y, xlim, ylim, bins=FES_BINS, pseudocount=FES_PSEUDOCOUNT):
    """Compute a masked free-energy surface from 2D coordinates."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    H = H.T
    occupied = H > 0

    P = np.full_like(H, np.nan, dtype=float)
    if occupied.any():
        denom = H[occupied].sum() + pseudocount * occupied.sum()
        P[occupied] = (H[occupied] + pseudocount) / denom

    F = np.full_like(P, np.nan, dtype=float)
    if occupied.any():
        F[occupied] = -np.log(P[occupied])
        F[occupied] -= np.nanmin(F[occupied])

    return np.ma.masked_invalid(F), xedges, yedges


def build_fes_lookup(results, key, problems, xlim, ylim, bins=FES_BINS):
    """Build per-system FES arrays on a shared grid."""
    lookup = {}
    for problem in problems:
        Y = _valid_projection_array(results, problem, key)
        if Y is None:
            continue
        lookup[problem] = _fes_from_xy(Y[:, 0], Y[:, 1], xlim, ylim, bins=bins)
    return lookup


def _grid_axes(n_panels, ncols=FES_NCOLS):
    """Create a sensible subplot grid."""
    ncols = max(1, min(ncols, n_panels))
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.2 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    return fig, axes.ravel()


def _masked_valid_values(values):
    """Return finite, unmasked values from an array-like object."""
    if np.ma.isMaskedArray(values):
        return values.compressed()
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def plot_projection_fes_grid(fes_lookup, key, title, out_path, cmap):
    """Plot simulation FES grids for PCA or TICA."""
    problems = sorted(fes_lookup.keys(), key=_problem_sort_key)
    if not problems:
        return

    vmax_vals = []
    for problem in problems:
        F, _, _ = fes_lookup[problem]
        valid = _masked_valid_values(F)
        if valid.size > 0:
            vmax_vals.append(np.percentile(valid, 95))
    if not vmax_vals:
        return

    vmax = max(vmax_vals)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = _grid_axes(len(problems))
    cmap_obj = plt.colormaps[cmap].copy()
    cmap_obj.set_bad("white")
    label = _projection_label(key)

    mappable = None
    for ax, problem in zip(axes, problems):
        F, xedges, yedges = fes_lookup[problem]
        mappable = ax.imshow(
            F,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            cmap=cmap_obj,
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(problem, fontsize=10)
        ax.set_xlabel(f"{label} 1")
        ax.set_ylabel(f"{label} 2")

    for ax in axes[len(problems):]:
        ax.axis("off")

    cbar = fig.colorbar(mappable, ax=axes[:len(problems)], shrink=0.95, pad=0.02)
    cbar.set_label(r"Free energy $-\ln P$")
    fig.suptitle(title, fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_projection_fes_diff_grid(ref_lookup, sim_lookup, key, title, out_path, cmap="RdBu_r"):
    """Plot FES difference grids using sys - ref on a shared scale."""
    problems = sorted(set(ref_lookup) & set(sim_lookup), key=_problem_sort_key)
    if not problems:
        return

    diffs = []
    abs_vals = []
    for problem in problems:
        ref_fes, xedges, yedges = ref_lookup[problem]
        sim_fes, _, _ = sim_lookup[problem]
        diff = np.ma.masked_invalid(np.ma.filled(sim_fes, np.nan) - np.ma.filled(ref_fes, np.nan))
        diffs.append((problem, diff, xedges, yedges))

        valid = _masked_valid_values(diff)
        if valid.size > 0:
            abs_vals.append(np.percentile(np.abs(valid), 95))

    if not abs_vals:
        return

    vmax = max(abs_vals)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = _grid_axes(len(problems))
    cmap_obj = plt.colormaps[cmap].copy()
    cmap_obj.set_bad("white")
    label = _projection_label(key)

    mappable = None
    for ax, (problem, diff, xedges, yedges) in zip(axes, diffs):
        mappable = ax.imshow(
            diff,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            cmap=cmap_obj,
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(problem, fontsize=10)
        ax.set_xlabel(f"{label} 1")
        ax.set_ylabel(f"{label} 2")

    for ax in axes[len(problems):]:
        ax.axis("off")

    cbar = fig.colorbar(mappable, ax=axes[:len(problems)], shrink=0.95, pad=0.02)
    cbar.set_label(r"$\Delta F$ (sys - ref)")
    fig.suptitle(title, fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _flatten_fes_overlap(ref_fes, sim_fes):
    """Return flattened overlapping finite bins from two FES arrays."""
    ref_values = np.asarray(np.ma.filled(ref_fes, np.nan), dtype=float)
    sim_values = np.asarray(np.ma.filled(sim_fes, np.nan), dtype=float)
    valid = np.isfinite(ref_values) & np.isfinite(sim_values)
    if np.sum(valid) == 0:
        return None, None
    return ref_values[valid].ravel(), sim_values[valid].ravel()


def fes_absdiff_metric(ref_fes, sim_fes):
    """Compute mean absolute FES difference across overlapping finite bins."""
    ref_values, sim_values = _flatten_fes_overlap(ref_fes, sim_fes)
    if ref_values is None:
        return float("nan")
    return float(np.mean(np.abs(sim_values - ref_values)))


def fes_pearson_similarity(ref_fes, sim_fes):
    """Compute Pearson correlation across overlapping finite FES bins."""
    ref_values, sim_values = _flatten_fes_overlap(ref_fes, sim_fes)
    if ref_values is None or ref_values.size < 2:
        return float("nan")
    if np.allclose(ref_values, ref_values[0]) or np.allclose(sim_values, sim_values[0]):
        return float("nan")
    return float(np.corrcoef(ref_values, sim_values)[0, 1])


def compute_fes_metrics(ref_lookup, sim_lookup):
    """Compute FES absdiff and Pearson metrics per problem."""
    metrics = {}
    for problem in sorted(set(ref_lookup) & set(sim_lookup), key=_problem_sort_key):
        ref_fes, _, _ = ref_lookup[problem]
        sim_fes, _, _ = sim_lookup[problem]
        metrics[problem] = {
            "absdiff": fes_absdiff_metric(ref_fes, sim_fes),
            "pearson": fes_pearson_similarity(ref_fes, sim_fes),
        }
    return metrics


def build_projection_summary(ref_results, sim_results, key, plots_dir, sys_name, diff_name, cmap):
    """Compute FES lookups, write plots, and return per-system metrics."""
    problems = projection_problems(ref_results, sim_results, key)
    if not problems:
        print(f"No overlapping {key} projections found; skipping FES plots.")
        return {}

    xlim, ylim = _compute_projection_limits(ref_results, sim_results, key, problems)
    ref_lookup = build_fes_lookup(ref_results, key, problems, xlim, ylim, bins=FES_BINS)
    sim_lookup = build_fes_lookup(sim_results, key, problems, xlim, ylim, bins=FES_BINS)

    label = _projection_label(key)
    plot_projection_fes_grid(
        sim_lookup,
        key,
        f"Simulation {label} Free Energy Surfaces",
        plots_dir / sys_name,
        cmap=cmap,
    )
    plot_projection_fes_diff_grid(
        ref_lookup,
        sim_lookup,
        key,
        f"{label} Free Energy Difference (sys - ref)",
        plots_dir / diff_name,
    )
    return compute_fes_metrics(ref_lookup, sim_lookup)


def plot_rmsd_overview(systems, sim_data, ref_data, out_path, title):
    """Plot RMSD vs time for each system with ref/sim overlays."""
    if not systems:
        return

    def mean_after_time(entry, min_time_ps):
        if not entry or entry.get("rmsd") is None:
            return None
        time = entry.get("time")
        if time is None:
            return None
        rmsd = np.asarray(entry["rmsd"], dtype=float)
        time = np.asarray(time, dtype=float)
        if rmsd.shape != time.shape:
            return None
        mask = np.isfinite(rmsd) & np.isfinite(time) & (time >= min_time_ps)
        if not np.any(mask):
            return None
        return float(np.mean(rmsd[mask]))

    n = len(systems)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.0 * nrows),
        squeeze=False,
    )

    handles = []
    labels = []

    for idx, system in enumerate(systems):
        ax = axes[idx // ncols][idx % ncols]
        sim_entry = sim_data.get(system)
        ref_entry = ref_data.get(system)

        if sim_entry and sim_entry.get("rmsd") is not None:
            x_sim = sim_entry.get("time")
            if x_sim is None:
                x_sim = np.arange(len(sim_entry["rmsd"]))
                x_label = "Frame"
            else:
                x_label = "Time (ps)"
            line_sim = ax.plot(
                x_sim,
                sim_entry["rmsd"],
                label="sim",
                color="tab:blue",
                alpha=0.6,
            )
            sim_mean = mean_after_time(sim_entry, 500.0)
            if sim_mean is not None:
                ax.axhline(
                    sim_mean,
                    color="tab:blue",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=1,
                    label="_nolegend_",
                )
            if not handles:
                handles.append(line_sim[0])
                labels.append("sim")
        else:
            x_label = "Time (ps)"

        if ref_entry and ref_entry.get("rmsd") is not None:
            x_ref = ref_entry.get("time")
            if x_ref is None:
                x_ref = np.arange(len(ref_entry["rmsd"]))
                x_label = "Frame"
            else:
                x_label = "Time (ps)"
            line_ref = ax.plot(
                x_ref,
                ref_entry["rmsd"],
                label="ref",
                color="tab:orange",
                alpha=0.6,
            )
            ref_mean = mean_after_time(ref_entry, 500.0)
            if ref_mean is not None:
                ax.axhline(
                    ref_mean,
                    color="tab:orange",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=1,
                    label="_nolegend_",
                )
            if len(handles) < 2:
                handles.append(line_ref[0])
                labels.append("ref")

        ax.set_title(system)
        ax.set_xlabel(x_label)
        ax.set_ylabel("RMSD (A)")

    total_axes = nrows * ncols
    for idx in range(n, total_axes):
        axes[idx // ncols][idx % ncols].axis("off")

    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2)
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.9,
        bottom=0.08,
        wspace=0.3,
        hspace=0.35,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_contactmaps(contactmaps, out_path, title):
    """Plot a grid of contact maps."""
    if not contactmaps:
        return

    systems = [item[0] for item in contactmaps]
    matrices = [item[1] for item in contactmaps]

    n = len(systems)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 3.6 * nrows),
        squeeze=False,
    )

    image = None
    for idx, (system, matrix) in enumerate(zip(systems, matrices)):
        ax = axes[idx // ncols][idx % ncols]
        image = ax.imshow(matrix, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        title_text = textwrap.fill(str(system), width=16)
        ax.set_title(title_text, fontsize=14, pad=1)

    total_axes = nrows * ncols
    for idx in range(n, total_axes):
        axes[idx // ncols][idx % ncols].axis("off")

    if image is not None:
        cax = fig.add_axes([0.91, 0.08, 0.015, 0.84])
        fig.colorbar(image, cax=cax)
    fig.suptitle(title, fontsize=26, y=0.97)
    fig.subplots_adjust(
        left=0.04,
        right=0.89,
        top=0.92,
        bottom=0.04,
        wspace=0.25,
        hspace=0.35,
    )

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_contactmaps_simdiff(ref_contact_items, sim_contact_items, out_path, title):
    """Plot sim contact maps in upper triangle and ref-sim in lower triangle."""
    if not ref_contact_items or not sim_contact_items:
        return

    ref_maps = {system: matrix for system, matrix in ref_contact_items}
    sim_maps = {system: matrix for system, matrix in sim_contact_items}

    systems = [system for system in sorted(sim_maps, key=_problem_sort_key) if system in ref_maps]
    entries = []
    for system in systems:
        ref_matrix = ref_maps.get(system)
        sim_matrix = sim_maps.get(system)
        if ref_matrix is None or sim_matrix is None:
            continue
        if ref_matrix.shape != sim_matrix.shape:
            continue
        entries.append((system, ref_matrix, sim_matrix))

    if not entries:
        return

    n = len(entries)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 3.6 * nrows),
        squeeze=False,
    )

    sim_cmap = plt.get_cmap("viridis").copy()
    sim_cmap.set_bad(alpha=0.0)
    diff_cmap = plt.get_cmap("RdBu").copy()
    diff_cmap.set_bad(alpha=0.0)
    diag_cmap = mcolors.ListedColormap(["black"])
    diag_cmap.set_bad(alpha=0.0)

    sim_image = None
    diff_image = None

    for idx, (system, ref_matrix, sim_matrix) in enumerate(entries):
        ax = axes[idx // ncols][idx % ncols]
        diff_matrix = sim_matrix - ref_matrix

        mask_upper = np.tril(np.ones_like(sim_matrix, dtype=bool), k=-1)
        mask_lower = np.triu(np.ones_like(sim_matrix, dtype=bool), k=0)

        sim_display = np.ma.array(sim_matrix, mask=~mask_upper)
        diff_display = np.ma.array(diff_matrix, mask=~mask_lower)

        sim_image = ax.imshow(
            sim_display,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap=sim_cmap,
        )
        diff_image = ax.imshow(
            diff_display,
            origin="lower",
            vmin=-1.0,
            vmax=1.0,
            cmap=diff_cmap,
        )
        diag_mask = np.eye(sim_matrix.shape[0], dtype=bool)
        diag_display = np.ma.array(np.ones_like(sim_matrix), mask=~diag_mask)
        ax.imshow(
            diag_display,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap=diag_cmap,
        )
        title_text = textwrap.fill(str(system), width=16)
        ax.set_title(title_text, fontsize=14, pad=1)

    total_axes = nrows * ncols
    for idx in range(n, total_axes):
        axes[idx // ncols][idx % ncols].axis("off")

    if sim_image is not None:
        cax_sim = fig.add_axes([0.88, 0.08, 0.015, 0.84])
        cbar_sim = fig.colorbar(sim_image, cax=cax_sim)
        cbar_sim.set_label("Sim contact", fontsize=18)
    if diff_image is not None:
        cax_diff = fig.add_axes([0.915, 0.08, 0.015, 0.84])
        cbar_diff = fig.colorbar(diff_image, cax=cax_diff)
        cbar_diff.set_label("Ref - Sim", fontsize=18)

    fig.suptitle(title, fontsize=26, y=0.97)
    fig.subplots_adjust(
        left=0.04,
        right=0.86,
        top=0.92,
        bottom=0.04,
        wspace=0.25,
        hspace=0.35,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(df, out_path, title):
    """Plot a correlation heatmap for numeric summary columns."""
    corr_pearson = df.corr()

    fig, ax = plt.subplots(
        figsize=(0.6 * len(corr_pearson.columns) + 4, 0.6 * len(corr_pearson.columns) + 4)
    )

    hm = sns.heatmap(
        corr_pearson,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar=False,
        ax=ax,
    )

    ax.set_title(title, fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    fig.colorbar(hm.collections[0], cax=cax, label="Pearson r")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run summary calculations and write plots/summary CSV."""
    args = parse_args()
    root = Path(args.root).resolve()
    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.is_absolute():
        analysis_dir = root / analysis_dir
    analysis_ref = Path(args.analysis_ref)
    if not analysis_ref.is_absolute():
        analysis_ref = root / analysis_ref

    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = out_dir / "summary_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ref_scaffold = load_ref_rmsd(analysis_ref / "rmsd_ref.pickle")
    ref_motif = load_ref_rmsd(analysis_ref / "rmsd_motif_ref.pickle")
    sim_rmsd = load_sim_rmsd(analysis_dir)

    ref_contacts = load_contactmaps_ref(analysis_ref)
    sim_contacts = load_contactmaps_sim(analysis_dir)

    ref_projections = load_projection_results(
        analysis_ref / "results_tica-pca.pkl",
        requires_model_support=True,
    )
    sim_projections = load_projection_results(
        analysis_dir / "results_tica-pca_sys.pkl",
        requires_model_support=False,
    )

    systems = sorted(
        set(sim_rmsd.keys()) | set(sim_contacts.keys()) | set(sim_projections.keys()),
        key=_problem_sort_key,
    )

    scaffold_sim = {system: sim_rmsd.get(system, {}).get("scaffold") for system in systems}
    motif_sim = {system: sim_rmsd.get(system, {}).get("motif") for system in systems}

    plot_rmsd_overview(
        systems,
        scaffold_sim,
        ref_scaffold,
        plots_dir / "scaffold_rmsd_overview.png",
        "Scaffold RMSD (sim vs ref)",
    )
    plot_rmsd_overview(
        systems,
        motif_sim,
        ref_motif,
        plots_dir / "motif_rmsd_overview.png",
        "Motif RMSD (sim vs ref)",
    )

    ref_contact_items = sorted(ref_contacts.items(), key=lambda item: _problem_sort_key(item[0]))
    sim_contact_items = sorted(sim_contacts.items(), key=lambda item: _problem_sort_key(item[0]))

    plot_contactmaps(
        ref_contact_items,
        plots_dir / "contactmaps_ref.png",
        "Reference motif contact maps",
    )
    plot_contactmaps_simdiff(
        ref_contact_items,
        sim_contact_items,
        plots_dir / "contactmaps_simdiff.png",
        "Simulation motif contact maps (upper) and ref - sim (lower)",
    )

    pca_fes_metrics = build_projection_summary(
        ref_projections,
        sim_projections,
        key="Y_pca",
        plots_dir=plots_dir,
        sys_name="fes_pca_sys.png",
        diff_name="fes_pca_diff_sys_minus_ref.png",
        cmap="turbo",
    )
    tica_fes_metrics = build_projection_summary(
        ref_projections,
        sim_projections,
        key="Y_tica",
        plots_dir=plots_dir,
        sys_name="fes_tica_sys.png",
        diff_name="fes_tica_diff_sys_minus_ref.png",
        cmap="magma",
    )

    rows = []
    for system in systems:
        scaffold_ref_entry = ref_scaffold.get(system)
        scaffold_sim_entry = scaffold_sim.get(system)
        motif_ref_entry = ref_motif.get(system)
        motif_sim_entry = motif_sim.get(system)

        scaffold_ref_mean, scaffold_ref_std, _ = compute_stats(
            None if scaffold_ref_entry is None else scaffold_ref_entry.get("rmsd"),
            args.drop_fraction,
        )
        scaffold_sim_mean, scaffold_sim_std, _ = compute_stats(
            None if scaffold_sim_entry is None else scaffold_sim_entry.get("rmsd"),
            args.drop_fraction,
        )
        motif_ref_mean, motif_ref_std, _ = compute_stats(
            None if motif_ref_entry is None else motif_ref_entry.get("rmsd"),
            args.drop_fraction,
        )
        motif_sim_mean, motif_sim_std, _ = compute_stats(
            None if motif_sim_entry is None else motif_sim_entry.get("rmsd"),
            args.drop_fraction,
        )
        scaffold_mean_similarity_std = mean_similarity_from_std(
            scaffold_ref_mean,
            scaffold_sim_mean,
            scaffold_ref_std,
            scaffold_sim_std,
        )
        motif_mean_similarity_std = mean_similarity_from_std(
            motif_ref_mean,
            motif_sim_mean,
            motif_ref_std,
            motif_sim_std,
        )

        contact_ref = ref_contacts.get(system)
        contact_sim = sim_contacts.get(system)
        contact_jaccard = jaccard_similarity(
            contact_ref,
            contact_sim,
            args.contact_threshold,
        )
        contact_pearson = pearson_similarity(
            contact_ref,
            contact_sim,
        )
        contact_absdiff_sum, contact_absdiff_weighted = contact_absdiff_metrics(
            contact_ref,
            contact_sim,
        )

        pca_metrics = pca_fes_metrics.get(system, {})
        tica_metrics = tica_fes_metrics.get(system, {})

        rows.append(
            {
                "system": system,
                "motif_rmsd_mean_dif": motif_ref_mean - motif_sim_mean,
                "motif_rmsd_mean_similarity_std": motif_mean_similarity_std,
                "scaffold_rmsd_mean_dif": scaffold_ref_mean - scaffold_sim_mean,
                "scaffold_rmsd_mean_similarity_std": scaffold_mean_similarity_std,
                "contact_jaccard": contact_jaccard,
                "contact_pearson": contact_pearson,
                "contact_absdiff_sum": contact_absdiff_sum,
                "contact_absdiff_weighted": contact_absdiff_weighted,
                "pca_fes_absdiff": pca_metrics.get("absdiff", float("nan")),
                "pca_fes_pearson": pca_metrics.get("pearson", float("nan")),
                "tica_fes_absdiff": tica_metrics.get("absdiff", float("nan")),
                "tica_fes_pearson": tica_metrics.get("pearson", float("nan")),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "summary_by_problem.csv"
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df.loc[:, numeric_cols] = summary_df.loc[:, numeric_cols].round(2)
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote plots to: {plots_dir}")

    maybe_run_get_motif_samples(root)
    motif_pdbs_info_path = root / "motif_pdbs/motif_pdbs.csv"
    if not motif_pdbs_info_path.is_file():
        print(f"ERROR: No motif_pdbs.csv at {motif_pdbs_info_path}")
    else:
        motif_pdbs_info = pd.read_csv(motif_pdbs_info_path)

        summary = pd.read_csv(summary_path)
        merged = motif_pdbs_info.merge(
            summary,
            left_on="problem",
            right_on="system",
            how="inner",
        )
        merged["solved"] = (
            merged["solved"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )

        # Save "solved" separately because it is needed to split the rows later.
        solved = merged["solved"].copy()

        # Remove identifiers and the Boolean grouping column.
        candidates = merged.drop(
            columns=["source_path", "problem", "system", "solved"],
            errors="ignore",
        ).copy()

        # Attempt to convert every value to a number.
        numeric_candidates = candidates.apply(
            pd.to_numeric,
            errors="coerce",
        )

        # Keep a column only when all its original non-null values were numeric.
        numeric_columns = [
            column
            for column in candidates.columns
            if numeric_candidates[column].notna().sum()
            == candidates[column].notna().sum()
        ]

        discarded_columns = [
            column
            for column in candidates.columns
            if column not in numeric_columns
        ]

        if discarded_columns:
            print(
                "Discarding columns containing non-numeric values: "
                + ", ".join(discarded_columns)
            )

        df_values = numeric_candidates[numeric_columns].copy()

        # Add solved back for splitting solved and failed cases.
        df_values.insert(0, "solved", solved.to_numpy(dtype=bool))

        df_values.to_csv(
            plots_dir / "correlationMap_all.csv",
            index=False,
        )

        df_true = (
            df_values.loc[df_values["solved"]]
            .drop(columns="solved")
            .copy()
        )

        df_false = (
            df_values.loc[~df_values["solved"]]
            .drop(columns="solved")
            .copy()
        )

        plot_correlation_heatmap(
            df_values,
            plots_dir / "correlationMap_all.png",
            "Correlations all problems",
        )
        plot_correlation_heatmap(
            df_true,
            plots_dir / "correlationMap_solvedProblems.png",
            "Correlations solved problems",
        )
        plot_correlation_heatmap(
            df_false,
            plots_dir / "correlationMap_failedProblems.png",
            "Correlations failed problems",
        )


if __name__ == "__main__":
    main()
