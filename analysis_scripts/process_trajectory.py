"""Align and center trajectories for all systems in a BILBY-MD run."""

import argparse
import os

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.transformations import center_in_box


def iter_systems(sim_root):
    """Yield system names from the simulations root in sorted order."""
    if not os.path.isdir(sim_root):
        raise FileNotFoundError(f"Simulation root not found: {sim_root}")
    for name in sorted(os.listdir(sim_root)):
        if os.path.isdir(os.path.join(sim_root, name)):
            yield name


def build_system_paths(root, sim_dir, system):
    """Return topology and trajectory paths for a system."""
    prod_dir = os.path.join(root, sim_dir, system, "production")
    top = os.path.join(prod_dir, f"md_{system}_noW.gro")
    traj = os.path.join(prod_dir, f"md_{system}_noW.xtc")
    if not os.path.isfile(traj):
        traj = os.path.join(prod_dir, f"md_{system}_noW.trr")
    traj_out = os.path.join(prod_dir, f"md_{system}_noW_aligned.xtc")
    return top, traj, traj_out


def align_trajectory(top, traj, traj_out, selection, weights, overwrite=False):
    """Align a trajectory to frame 0 and write an aligned XTC."""
    if not os.path.isfile(top):
        print(f"Skipping, missing topology: {top}")
        return False
    if not os.path.isfile(traj):
        print(f"Skipping, missing trajectory: {traj}")
        return False
    if os.path.isfile(traj_out) and not overwrite:
        print(f"Skipping, aligned trajectory exists: {traj_out}")
        return False

    u = mda.Universe(top, traj)
    center_atoms = u.select_atoms("protein or nucleic")
    if center_atoms.n_atoms == 0:
        center_atoms = u.atoms
    u.trajectory.add_transformations(center_in_box(center_atoms, wrap=True))

    if u.select_atoms(selection).n_atoms == 0:
        print(f"Skipping, alignment selection empty: {selection} for {traj}")
        return False

    ref = mda.Universe(top, traj)
    ref.trajectory[0]
    aligner = align.AlignTraj(
        u,
        ref,
        select=selection,
        weights=weights,
        filename=traj_out,
        in_memory=False,
    )
    aligner.run()
    print(f"Wrote aligned trajectory: {traj_out}")
    return True


def parse_args():
    """Parse CLI arguments for alignment settings."""
    parser = argparse.ArgumentParser(
        description="Align and center trajectories for all systems.",
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
        "--align-selection",
        default="backbone",
        help="Atom selection for alignment (MDAnalysis syntax).",
    )
    parser.add_argument(
        "--weights",
        default="mass",
        help="Weights for alignment (e.g., mass or None).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing aligned trajectories.",
    )
    return parser.parse_args()


def main():
    """Run alignment over all simulation systems."""
    args = parse_args()
    root = os.path.abspath(args.root)
    sim_root = os.path.join(root, args.sim_dir)
    for system in iter_systems(sim_root):
        top, traj, traj_out = build_system_paths(root, args.sim_dir, system)
        align_trajectory(
            top,
            traj,
            traj_out,
            args.align_selection,
            args.weights,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
