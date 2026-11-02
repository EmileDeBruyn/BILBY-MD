#!/usr/bin/env python3
"""Merge newly generated enzyme reference assets into analysis_ref."""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import shutil
from pathlib import Path


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
    path = Path(path)
    with path.open("rb") as handle:
        return NumPyCompatUnpickler(handle).load()


def write_pickle(path, payload):
    """Write a pickle file to disk."""
    path = Path(path)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    """Parse CLI arguments for reference merging."""
    parser = argparse.ArgumentParser(
        description="Merge newly computed enzyme reference assets into analysis_ref.",
    )
    parser.add_argument(
        "--root",
        default=os.getcwd(),
        help="Project root containing analysis_ref/.",
    )
    parser.add_argument(
        "--source-dir",
        default="analysis_ref/update_inputs",
        help="Directory containing freshly generated RMSD/contact/FES outputs.",
    )
    parser.add_argument(
        "--ref-dir",
        default="analysis_ref",
        help="Reference directory to amend.",
    )
    return parser.parse_args()


def ensure_dict_pickle(path):
    """Load a dict pickle if present, otherwise return an empty dict."""
    if not os.path.isfile(path):
        return {}
    data = load_pickle(path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict pickle at {path}, found {type(data).__name__}")
    return data


def merge_rmsd_payloads(source_dir, ref_dir):
    """Append per-system RMSD data into the legacy reference pickle layout."""
    scaffold_path = os.path.join(ref_dir, "rmsd_ref.pickle")
    motif_primary_path = os.path.join(ref_dir, "rmsd_motif_ref.pickle")
    motif_optional_path = os.path.join(ref_dir, "rmsd_motif.pickle")

    scaffold_ref = ensure_dict_pickle(scaffold_path)
    motif_ref = ensure_dict_pickle(motif_primary_path)
    motif_extra = ensure_dict_pickle(motif_optional_path) if os.path.isfile(motif_optional_path) else None

    rmsd_paths = sorted(glob.glob(os.path.join(source_dir, "*_rmsd.pkl")))
    for rmsd_path in rmsd_paths:
        payload = load_pickle(rmsd_path)
        if not isinstance(payload, dict):
            print(f"Skipping non-dict RMSD payload: {rmsd_path}")
            continue

        system = payload.get("system") or os.path.basename(rmsd_path).replace("_rmsd.pkl", "")

        scaffold = payload.get("scaffold") or {}
        if scaffold:
            scaffold_ref[f"{system}_t"] = scaffold.get("time_ps")
            scaffold_ref[system] = scaffold.get("rmsd_A")
            scaffold_ref[f"{system}_normed"] = scaffold.get("rmsd_normed")

        motif = payload.get("motif") or {}
        if motif:
            motif_ref[f"{system}_t"] = motif.get("time_ps")
            motif_ref[system] = motif.get("rmsd_A")
            motif_ref[f"{system}_normed"] = motif.get("rmsd_normed")
            if motif_extra is not None:
                motif_extra[f"{system}_t"] = motif.get("time_ps")
                motif_extra[system] = motif.get("rmsd_A")
                motif_extra[f"{system}_normed"] = motif.get("rmsd_normed")

        print(f"Merged RMSD reference data for {system}")

    write_pickle(scaffold_path, scaffold_ref)
    write_pickle(motif_primary_path, motif_ref)
    if motif_extra is not None:
        write_pickle(motif_optional_path, motif_extra)


def merge_contactmaps(source_dir, ref_dir):
    """Copy new motif contact maps into the reference directory."""
    contact_paths = sorted(glob.glob(os.path.join(source_dir, "*_motif_contactmap.npy")))
    for contact_path in contact_paths:
        filename = os.path.basename(contact_path)
        system = filename.replace("_motif_contactmap.npy", "")
        ref_path = os.path.join(ref_dir, f"{system}_ref_motif_contactmap.npy")
        shutil.copy2(contact_path, ref_path)
        print(f"Updated reference contact map: {ref_path}")


def merge_fes(source_dir, ref_dir):
    """Append new scratch FES models into the reference projection pickle."""
    source_path = os.path.join(source_dir, "results_tica-pca_ref_scratch.pkl")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Scratch FES pickle not found: {source_path}")

    new_results = ensure_dict_pickle(source_path)
    if not new_results:
        print("Scratch FES pickle is empty; no projection entries merged.")
        return

    target_names = ["results_tica-pca_base.pkl"]
    regular_target = os.path.join(ref_dir, "results_tica-pca.pkl")
    if os.path.isfile(regular_target):
        target_names.append("results_tica-pca.pkl")

    for target_name in target_names:
        target_path = os.path.join(ref_dir, target_name)
        target_results = ensure_dict_pickle(target_path)
        target_results.update(new_results)
        write_pickle(target_path, target_results)
        print(f"Merged FES reference entries into {target_path}")


def main():
    """Run the reference merge workflow."""
    args = parse_args()
    root = os.path.abspath(args.root)

    source_dir = args.source_dir
    if not os.path.isabs(source_dir):
        source_dir = os.path.join(root, source_dir)
    ref_dir = args.ref_dir
    if not os.path.isabs(ref_dir):
        ref_dir = os.path.join(root, ref_dir)

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")

    merge_rmsd_payloads(source_dir, ref_dir)
    merge_contactmaps(source_dir, ref_dir)
    merge_fes(source_dir, ref_dir)
    print("Reference update complete.")


if __name__ == "__main__":
    main()
