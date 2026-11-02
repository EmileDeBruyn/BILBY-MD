"""Compute RMSD time series for each system and motif subset."""

import argparse
import json
import os
import pickle

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD


def iter_systems(sim_root):
    """Yield system names from the simulations root in sorted order."""
    if not os.path.isdir(sim_root):
        raise FileNotFoundError(f"Simulation root not found: {sim_root}")
    for name in sorted(os.listdir(sim_root)):
        if os.path.isdir(os.path.join(sim_root, name)):
            yield name


def build_system_paths(root, sim_dir, system, prefer_aligned=True):
    """Return topology and trajectory paths for a system."""
    prod_dir = os.path.join(root, sim_dir, system, "production")
    top = os.path.join(prod_dir, f"md_{system}_noW.gro")
    aligned = os.path.join(prod_dir, f"md_{system}_noW_aligned.xtc")
    raw = os.path.join(prod_dir, f"md_{system}_noW.xtc")

    if prefer_aligned and os.path.isfile(aligned):
        traj = aligned
    elif os.path.isfile(raw):
        if prefer_aligned:
            print(f"Aligned trajectory missing; using raw: {raw}")
        traj = raw
    else:
        traj = None

    return top, traj


def load_motif_info(motif_info_path):
    """Load motif metadata from JSON, returning an empty dict if missing."""
    if not motif_info_path or not os.path.isfile(motif_info_path):
        return {}
    with open(motif_info_path, "r") as handle:
        return json.load(handle)


def build_resid_selection(residues):
    """Build an MDAnalysis resid selection string from a list of residues."""
    resid_list = " ".join(str(int(resid)) for resid in residues)
    return f"resid {resid_list}"


def compute_rmsd_series(universe, selection):
    """Compute RMSD time series for a given atom selection."""
    atoms = universe.select_atoms(selection)
    if atoms.n_atoms == 0:
        print(f"Skipping, selection has 0 atoms: {selection}")
        return None

    rmsd = RMSD(universe, select=selection)
    rmsd.run()

    rmsd_time_ps = rmsd.results.rmsd[:, 1]
    rmsd_A = rmsd.results.rmsd[:, 2]
    n_res = len(atoms.residues)
    if n_res:
        rmsd_normed = rmsd_A / n_res
    else:
        rmsd_normed = np.full_like(rmsd_A, np.nan)

    return {
        "time_ps": rmsd_time_ps,
        "rmsd_A": rmsd_A,
        "rmsd_normed": rmsd_normed,
        "n_residues": n_res,
        "selection": selection,
    }


def write_pickle(path, payload):
    """Write a dictionary payload to a pickle file."""
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    """Parse CLI arguments for RMSD calculation."""
    parser = argparse.ArgumentParser(
        description="Compute scaffold and motif RMSD for each system.",
    )
    parser.add_argument(
        "--root",
        default=os.getcwd(),
        help="Root directory containing simulations/.",
    )
    parser.add_argument(
        "--sim-dir",
        default="simulations",
        help="Relative simulations directory under root.",
    )
    parser.add_argument(
        "--analysis-dir",
        default="analysis_sys",
        help="Output directory for RMSD pickles.",
    )
    parser.add_argument(
        "--motif-info",
        default=None,
        help="Path to motif_info.json (defaults to root/motif_info.json).",
    )
    parser.add_argument(
        "--selection",
        default="backbone",
        help="Atom selection for RMSD (MDAnalysis syntax).",
    )
    parser.add_argument(
        "--use-unaligned",
        action="store_true",
        help="Use unaligned trajectories even if aligned ones exist.",
    )
    return parser.parse_args()


def main():
    """Run RMSD calculation for all systems."""
    args = parse_args()
    root = os.path.abspath(args.root)
    sim_root = os.path.join(root, args.sim_dir)

    analysis_dir = args.analysis_dir
    if not os.path.isabs(analysis_dir):
        analysis_dir = os.path.join(root, analysis_dir)
    os.makedirs(analysis_dir, exist_ok=True)

    motif_info_path = args.motif_info or os.path.join(root, "motif_info.json")
    motif_info = load_motif_info(motif_info_path)
    if not motif_info:
        print("Motif info not found; motif RMSD will be skipped.")

    for system in iter_systems(sim_root):
        top, traj = build_system_paths(
            root,
            args.sim_dir,
            system,
            prefer_aligned=not args.use_unaligned,
        )
        if not os.path.isfile(top):
            print(f"Skipping, missing topology: {top}")
            continue
        if traj is None:
            print(f"Skipping, missing trajectory for {system}")
            continue

        u = mda.Universe(top, traj)
        scaffold = compute_rmsd_series(u, args.selection)
        if scaffold is None:
            continue

        payload = {
            "system": system,
            "topology": top,
            "trajectory": traj,
            "scaffold": scaffold,
        }

        motif_entry = motif_info.get(system)
        if motif_entry and motif_entry.get("motif_idx"):
            motif_sel = build_resid_selection(motif_entry["motif_idx"])
            motif_sel = f"({motif_sel}) and ({args.selection})"
            motif_data = compute_rmsd_series(u, motif_sel)
            if motif_data is not None:
                payload["motif"] = motif_data
        else:
            if motif_info:
                print(f"No motif indices for {system}; skipping motif RMSD.")

        out_path = os.path.join(analysis_dir, f"{system}_rmsd.pkl")
        write_pickle(out_path, payload)
        print(f"Wrote RMSD results: {out_path}")


if __name__ == "__main__":
    main()
