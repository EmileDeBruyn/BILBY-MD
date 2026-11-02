"""Project motif dihedral trajectories into reference PCA/TICA spaces."""

import argparse
import json
import os
import pickle

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals


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
    with open(motif_info_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_resid_selection(residues):
    """Build an MDAnalysis resid selection string from a list of residues."""
    resid_list = " ".join(str(int(resid)) for resid in residues)
    return f"resid {resid_list}"


def ensure_projection_pickle_support():
    """Import dependencies required to unpickle projection models."""
    try:
        import deeptime.decomposition  # noqa: F401
        from deeptime.decomposition import TICA  # noqa: F401
        from sklearn.decomposition import PCA  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "calc_fes.py requires deeptime and scikit-learn to load "
            "analysis_ref/results_tica-pca_base.pkl."
        ) from exc


class NumPyCompatUnpickler(pickle.Unpickler):
    """Load NumPy 2.x pickles in an environment using NumPy 1.x."""

    MODULE_ALIASES = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
    }

    def find_class(self, module, name):
        module = self.MODULE_ALIASES.get(module, module)
        return super().find_class(module, name)

def load_reference_models(path):
    """Load reference PCA/TICA models keyed by system."""
    if not os.path.isfile(path):
        print(f"Reference projection pickle not found: {path}")
        return {}

    ensure_projection_pickle_support()
    with open(path, "rb") as handle:
        data = NumPyCompatUnpickler(handle).load()
    if not isinstance(data, dict):
        print(f"Unexpected reference projection format in {path}; skipping.")
        return {}
    return data


def expected_feature_width(entry):
    """Infer the feature width required by the reference models."""
    pca_model = entry.get("pca_model")
    n_features = getattr(pca_model, "n_features_in_", None)
    if n_features is not None:
        return int(n_features)

    components = np.asarray(getattr(pca_model, "components_", None))
    if components.ndim == 2 and components.shape[1] > 0:
        return int(components.shape[1])

    tica_model = entry.get("tica_model")
    for attr in (
        "_instantaneous_coefficients",
        "_timelagged_coefficients",
        "_koopman_matrix",
    ):
        values = np.asarray(getattr(tica_model, attr, None))
        if values.ndim == 2 and values.shape[0] > 0:
            return int(values.shape[0])

    return None


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


def compute_dihedral_features(universe, motif_residues):
    """Compute stacked phi/psi angles for the motif."""
    motif_sel = build_resid_selection(motif_residues)
    motif_atoms = universe.select_atoms(f"({motif_sel}) and backbone")
    if motif_atoms.n_residues == 0:
        return None, "motif selection empty"

    phi_selections, psi_selections = collect_dihedral_selections(motif_atoms.residues)
    if not phi_selections or not psi_selections:
        return None, "missing phi/psi selections for motif residues"

    phis = dihedrals.Dihedral(phi_selections).run()
    psis = dihedrals.Dihedral(psi_selections).run()

    phi_angles = np.asarray(phis.results.angles, dtype=float)
    psi_angles = np.asarray(psis.results.angles, dtype=float)
    if phi_angles.ndim != 2 or psi_angles.ndim != 2:
        return None, "unexpected dihedral result shape"
    if phi_angles.shape[0] != psi_angles.shape[0]:
        return None, "phi/psi frame count mismatch"

    X = np.hstack((phi_angles, psi_angles))
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return None, "empty dihedral feature matrix"
    return X, None


def write_pickle(path, payload):
    """Write a dictionary payload to a pickle file."""
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    """Parse CLI arguments for FES projection calculation."""
    parser = argparse.ArgumentParser(
        description="Project motif dihedrals into reference PCA/TICA spaces.",
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
        help="Output directory for projection pickle.",
    )
    parser.add_argument(
        "--motif-info",
        default=None,
        help="Path to motif_info.json (defaults to root/motif_info.json).",
    )
    parser.add_argument(
        "--reference-results",
        default="analysis_ref/results_tica-pca.pkl",
        help="Reference PCA/TICA model pickle, relative to root if needed.",
    )
    parser.add_argument(
        "--use-unaligned",
        action="store_true",
        help="Use unaligned trajectories even if aligned ones exist.",
    )
    return parser.parse_args()


def main():
    """Run projection calculation for all systems."""
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
        print("Motif info not found; FES projection will be skipped.")
        return

    reference_results = args.reference_results
    if not os.path.isabs(reference_results):
        reference_results = os.path.join(root, reference_results)
    ref_models = load_reference_models(reference_results)
    if not ref_models:
        return

    results = {}
    for system in iter_systems(sim_root):
        motif_entry = motif_info.get(system)
        if not motif_entry or not motif_entry.get("motif_idx"):
            print(f"No motif indices for {system}; skipping FES projection.")
            continue
        if system not in ref_models:
            print(f"No reference PCA/TICA models for {system}; skipping.")
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
        X, error = compute_dihedral_features(u, motif_entry["motif_idx"])
        if X is None:
            print(f"Skipping FES projection for {system}: {error}")
            continue

        ref_entry = ref_models.get(system, {})
        feature_width = expected_feature_width(ref_entry)
        if feature_width is not None and X.shape[1] != feature_width:
            print(
                f"Skipping FES projection for {system}: feature width "
                f"{X.shape[1]} does not match reference width {feature_width}"
            )
            continue

        tica_model = ref_entry.get("tica_model")
        pca_model = ref_entry.get("pca_model")
        if tica_model is None or pca_model is None:
            print(f"Skipping FES projection for {system}: missing reference models")
            continue

        Y_tica_sys = np.asarray(tica_model.transform(X), dtype=float)
        Y_pca_sys = np.asarray(pca_model.transform(X), dtype=float)
        if Y_tica_sys.ndim != 2 or Y_tica_sys.shape[1] < 2:
            print(f"Skipping FES projection for {system}: invalid TICA projection shape")
            continue
        if Y_pca_sys.ndim != 2 or Y_pca_sys.shape[1] < 2:
            print(f"Skipping FES projection for {system}: invalid PCA projection shape")
            continue

        results[system] = {
            "Y_tica": Y_tica_sys[:, :2],
            "Y_pca": Y_pca_sys[:, :2],
        }
        print(f"Projected FES coordinates for {system}")

    out_path = os.path.join(analysis_dir, "results_tica-pca_sys.pkl")
    write_pickle(out_path, results)
    print(f"Wrote FES projections: {out_path}")


if __name__ == "__main__":
    main()
