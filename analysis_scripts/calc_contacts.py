"""Compute average motif contact maps for each system."""

import argparse
import json
import os

import numpy as np
import MDAnalysis as mda


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


def residues_in_contact(pos_a, pos_b, cutoff2):
    """Return True if any atom pair is within the squared cutoff distance."""
    for atom_a in pos_a:
        diffs = pos_b - atom_a
        dist2 = np.sum(diffs * diffs, axis=1)
        if np.any(dist2 <= cutoff2):
            return True
    return False


def compute_contact_matrix(residue_positions, cutoff):
    """Compute a symmetric residue contact matrix for a single frame."""
    n_res = len(residue_positions)
    contact_mat = np.zeros((n_res, n_res), dtype=np.float32)
    cutoff2 = cutoff * cutoff

    for i in range(n_res):
        pos_i = residue_positions[i]
        for j in range(i, n_res):
            pos_j = residue_positions[j]
            if residues_in_contact(pos_i, pos_j, cutoff2):
                contact_mat[i, j] = 1.0
                if i != j:
                    contact_mat[j, i] = 1.0

    return contact_mat


def parse_args():
    """Parse CLI arguments for contact map calculation."""
    parser = argparse.ArgumentParser(
        description="Compute average motif contact maps for each system.",
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
        help="Output directory for contact maps.",
    )
    parser.add_argument(
        "--motif-info",
        default=None,
        help="Path to motif_info.json (defaults to root/motif_info.json).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=3.0,
        help="Distance cutoff in Angstrom for a residue contact.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Frame stride for sampling the trajectory.",
    )
    parser.add_argument(
        "--use-unaligned",
        action="store_true",
        help="Use unaligned trajectories even if aligned ones exist.",
    )
    return parser.parse_args()


def main():
    """Run motif contact map calculation for all systems."""
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
        print("Motif info not found; no contact maps will be generated.")
        return

    for system in iter_systems(sim_root):
        motif_entry = motif_info.get(system)
        if not motif_entry or not motif_entry.get("motif_idx"):
            print(f"No motif indices for {system}; skipping contact map.")
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
        motif_sel = build_resid_selection(motif_entry["motif_idx"])
        motif_atoms = u.select_atoms(motif_sel)
        if motif_atoms.n_residues == 0:
            print(f"Skipping, motif selection empty for {system}")
            continue

        n_res = len(motif_atoms.residues)
        contact_accum = np.zeros((n_res, n_res), dtype=np.float32)
        n_sampled = 0

        for _ in u.trajectory[::args.interval]:
            positions = [res.atoms.positions.copy() for res in motif_atoms.residues]
            contact_accum += compute_contact_matrix(positions, args.cutoff)
            n_sampled += 1

        if n_sampled:
            contact_avg = contact_accum / n_sampled
        else:
            contact_avg = contact_accum

        out_path = os.path.join(analysis_dir, f"{system}_motif_contactmap.npy")
        np.save(out_path, contact_avg)
        print(f"Wrote contact map: {out_path}")


if __name__ == "__main__":
    main()
