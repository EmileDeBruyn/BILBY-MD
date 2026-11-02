#!/usr/bin/env python3
"""Fit per-system PCA/TICA models from enzyme trajectories for reference use."""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from deeptime.decomposition import TICA
from sklearn.decomposition import PCA


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
    raw_xtc = os.path.join(prod_dir, f"md_{system}_noW.xtc")
    raw_trr = os.path.join(prod_dir, f"md_{system}_noW.trr")

    if prefer_aligned and os.path.isfile(aligned):
        traj = aligned
    elif os.path.isfile(raw_xtc):
        if prefer_aligned:
            print(f"Aligned trajectory missing; using raw xtc: {raw_xtc}")
        traj = raw_xtc
    elif os.path.isfile(raw_trr):
        if prefer_aligned:
            print(f"Aligned trajectory missing; using raw trr: {raw_trr}")
        traj = raw_trr
    else:
        traj = None

    return top, traj


def load_motif_info(motif_info_path):
    """Load motif metadata from JSON, returning an empty dict if missing."""
    if not motif_info_path or not os.path.isfile(motif_info_path):
        return {}
    with open(motif_info_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_resid_selection(residues):
    """Build an MDAnalysis resid selection string from a list of residues."""
    resid_list = " ".join(str(int(resid)) for resid in residues)
    return f"resid {resid_list}"


def resolve_motif_residues(universe, motif_entry):
    """Resolve motif residues, preferring chain-specific selections when available."""
    motif_idx = motif_entry.get("motif_idx") or []
    motif_chain = motif_entry.get("motif_chain")
    resid_sel = build_resid_selection(motif_idx)
    expected = len(motif_idx)

    candidates = []
    if motif_chain:
        topology = universe._topology
        if hasattr(topology, "chainIDs"):
            candidates.append(f"({resid_sel}) and backbone and chainID {motif_chain}")
        if hasattr(topology, "segids"):
            candidates.append(f"({resid_sel}) and backbone and segid {motif_chain}")
    candidates.append(f"({resid_sel}) and backbone")

    fallback = None
    for selection in candidates:
        try:
            atoms = universe.select_atoms(selection)
        except (AttributeError, ValueError) as exc:
            print(
                f"Skipping unsupported motif selection for chain filtering: "
                f"{selection} ({exc})"
            )
            continue
        if atoms.n_residues == 0:
            continue
        if fallback is None:
            fallback = (selection, atoms)
        if atoms.n_residues == expected:
            return selection, atoms.residues

    if fallback is None:
        return None, None
    return fallback[0], fallback[1].residues


def collect_dihedral_selections(residues):
    """Collect valid phi/psi selections for motif residues."""
    phi_selections = []
    psi_selections = []
    for residue in residues:
        phi_sel = residue.phi_selection()
        psi_sel = residue.psi_selection()
        if phi_sel is not None:
            phi_selections.append(phi_sel)
        if psi_sel is not None:
            psi_selections.append(psi_sel)
    return phi_selections, psi_selections


def compute_dihedral_features(universe, motif_entry):
    """Compute stacked phi/psi angles for the motif."""
    selection, residues = resolve_motif_residues(universe, motif_entry)
    if residues is None or len(residues) == 0:
        return None, "motif selection empty", None

    phi_selections, psi_selections = collect_dihedral_selections(residues)
    if not phi_selections or not psi_selections:
        return None, "missing phi/psi selections for motif residues", selection

    phis = dihedrals.Dihedral(phi_selections).run()
    psis = dihedrals.Dihedral(psi_selections).run()

    phi_angles = np.asarray(phis.results.angles, dtype=float)
    psi_angles = np.asarray(psis.results.angles, dtype=float)
    if phi_angles.ndim != 2 or psi_angles.ndim != 2:
        return None, "unexpected dihedral result shape", selection
    if phi_angles.shape[0] != psi_angles.shape[0]:
        return None, "phi/psi frame count mismatch", selection

    X = np.hstack((phi_angles, psi_angles))
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return None, "empty dihedral feature matrix", selection
    return X, None, selection


def write_pickle(path, payload):
    """Write a dictionary payload to a pickle file."""
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    """Parse CLI arguments for reference FES fitting."""
    parser = argparse.ArgumentParser(
        description="Fit per-system PCA/TICA models from motif dihedrals.",
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
        "--motif-info",
        default=None,
        help="Path to motif_info.json (defaults to root/motif_info.json).",
    )
    parser.add_argument(
        "--output",
        default="analysis_ref/update_inputs/results_tica-pca_ref_scratch.pkl",
        help="Output pickle path, relative to root if needed.",
    )
    parser.add_argument(
        "--tica-lagtime",
        type=int,
        default=10,
        help="Lagtime for TICA fitting.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Number of output dimensions to retain.",
    )
    parser.add_argument(
        "--use-unaligned",
        action="store_true",
        help="Use unaligned trajectories even if aligned ones exist.",
    )
    return parser.parse_args()


def main():
    """Run reference FES fitting for all systems."""
    args = parse_args()
    if args.tica_lagtime <= 0:
        raise ValueError("--tica-lagtime must be positive.")
    if args.dim < 2:
        raise ValueError("--dim must be at least 2.")

    root = os.path.abspath(args.root)
    sim_root = os.path.join(root, args.sim_dir)

    motif_info_path = args.motif_info or os.path.join(root, "motif_info.json")
    motif_info = load_motif_info(motif_info_path)
    if not motif_info:
        raise FileNotFoundError(
            f"Motif info not found or empty: {motif_info_path}"
        )

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = {}
    for system in iter_systems(sim_root):
        motif_entry = motif_info.get(system)
        if not motif_entry or not motif_entry.get("motif_idx"):
            print(f"No motif indices for {system}; skipping FES fitting.")
            continue

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
        X, error, selection = compute_dihedral_features(u, motif_entry)
        if X is None:
            print(f"Skipping FES fitting for {system}: {error}")
            continue

        tica = TICA(lagtime=args.tica_lagtime, dim=args.dim).fit(X).fetch_model()
        pca = PCA(n_components=args.dim).fit(X)

        Y_tica = np.asarray(tica.transform(X), dtype=float)
        Y_pca = np.asarray(pca.transform(X), dtype=float)
        if Y_tica.ndim != 2 or Y_tica.shape[1] < 2:
            print(f"Skipping FES fitting for {system}: invalid TICA projection shape")
            continue
        if Y_pca.ndim != 2 or Y_pca.shape[1] < 2:
            print(f"Skipping FES fitting for {system}: invalid PCA projection shape")
            continue

        results[system] = {
            "tica_model": tica,
            "Y_tica": Y_tica[:, :2],
            "pca_model": pca,
            "Y_pca": Y_pca[:, :2],
            "motif_selection": selection,
        }
        print(f"Fitted scratch PCA/TICA reference for {system}")

    write_pickle(output_path, results)
    print(f"Wrote scratch FES reference pickle: {output_path}")


if __name__ == "__main__":
    main()
