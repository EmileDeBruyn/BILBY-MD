#!/usr/bin/env python3
"""
Get MotifBench Best-Scoring Motif Scaffolds

Usage:
    python getMotifSamples.py EVAL_DIRS [OUT_DIR] [--include-failures]
        [--problem-config PROBLEM_CONFIG]

Arguments:
    EVAL_DIRS : Path to a MotifBench evaluation directory or a JSON list of
                evaluation directories. Each directory must contain:
                    - summary_by_problem.csv
                    - problem subfolders with:
                         * esm_complete_results.csv
                         * motif_info.json

    OUT_DIR   : Optional. Directory to store results.
                Default: current working directory.

Options:
    --include-failures : Include problems with Num_Solutions <= 0 and select the
                         best run by tm_score even if Success == 0.
    --problem-config   : Optional JSON file with a `problems` list containing
                         exact problem folder names to include.

Description:
    This script identifies, for each problem in one or more MotifBench evaluations,
    the best-scoring scaffold based on TM-score.

    By default, only successful runs (Success == 1) are considered. When
    --include-failures is set, problems with Num_Solutions <= 0 are still
    processed and the best run is selected regardless of Success.

    For each problem:
        - Load esm_complete_results.csv
        - Optionally zero out TM scores of failed runs
        - Select the run with highest TM-score
        - Extract its motif metadata from motif_info.json
        - Copy its sample PDB file to OUT_DIR/motif_pdbs/<problem>.pdb

    The script writes:
        OUT_DIR/motif_pdbs/       - collected PDBs
        OUT_DIR/motif_info.json   - metadata per selected problem
"""

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import MDAnalysis as mda
import pandas as pd


logger = logging.getLogger(__name__)
STANDARD_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def configure_logging() -> None:
    """Configure basic logging for the script."""
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Select best MotifBench scaffolds from evaluation results.",
    )
    parser.add_argument(
        "eval_dirs",
        help=(
            "Path to a MotifBench evaluation directory, or a JSON list "
            "of evaluation directories."
        ),
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=Path.cwd(),
        help="Output directory for motif_pdbs/ and motif_info.json.",
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help=(
            "Include problems with Num_Solutions <= 0 and select the best run "
            "regardless of Success."
        ),
    )
    parser.add_argument(
        "--include-successes",
        type=int,
        choices=(0, 1),
        default=1,
        help="Set to 0 to skip problems with Num_Solutions > 0.",
    )
    parser.add_argument(
        "--problem-config",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with a `problems` list containing exact "
            "problem folder names to include."
        ),
    )
    parser.add_argument(
        "--failures-mode",
        choices=("high_tm", "low_tm"),
        default="high_tm",
        help=(
            "When --include-failures is set, select the failed backbone with "
            "the highest or lowest TM-score before PULCHRA reconstruction."
        ),
    )
    parser.add_argument(
        "--pulchra-bin",
        type=Path,
        default=Path("/e/project1/profound/schaeffler1/bin/pulchra"),
        help="Path to the PULCHRA executable used to reconstruct failed cases.",
    )
    parser.add_argument(
        "--failure-backbone-log-dir",
        type=Path,
        default=None,
        help=(
            "Directory for failed-case backbone logs. Default: "
            "OUT_DIR/failure_backbones."
        ),
    )
    return parser.parse_args()


def normalize_eval_dirs(eval_dirs_arg: str | list[str]) -> list[Path]:
    """Normalize input into a de-duplicated list of evaluation directories."""

    if isinstance(eval_dirs_arg, list):
        raw_eval_dirs = eval_dirs_arg
    else:
        stripped = eval_dirs_arg.strip()
        if stripped.startswith("["):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.error(f"Failed to parse EVAL_DIRS JSON list: {exc}")
                raise SystemExit(1) from exc

            if not isinstance(payload, list) or not all(
                isinstance(path, str) for path in payload
            ):
                logger.error("EVAL_DIRS JSON must be a list of strings.")
                raise SystemExit(1)
            raw_eval_dirs = payload
        else:
            raw_eval_dirs = [eval_dirs_arg]

    eval_dirs = []
    seen = set()
    for raw_dir in raw_eval_dirs:
        folder = raw_dir.strip()
        if not folder:
            continue

        resolved = Path(folder).expanduser().resolve()
        if resolved in seen:
            continue

        eval_dirs.append(resolved)
        seen.add(resolved)

    if not eval_dirs:
        logger.error("No evaluation directories were provided.")
        raise SystemExit(1)

    return eval_dirs


def load_problem_folders(eval_dirs: list[Path], config_path: Path | None) -> list[Path]:
    """Return the problem folders to iterate over."""

    if config_path is None:
        problem_folders = []
        seen = {}
        for eval_dir in eval_dirs:
            candidates = sorted(entry for entry in eval_dir.iterdir() if entry.is_dir())
            for entry in candidates:
                if entry.name in seen:
                    logger.warning(
                        f"Duplicate problem folder '{entry.name}' found in {entry.parent}; "
                        f"using first match from {seen[entry.name]}."
                    )
                    continue

                problem_folders.append(entry)
                seen[entry.name] = entry.parent

        return problem_folders

    resolved = config_path.resolve()
    if not resolved.exists():
        logger.error(f"Problem config does not exist: {resolved}")
        raise SystemExit(1)

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse problem config {resolved}: {exc}")
        raise SystemExit(1) from exc

    if isinstance(payload, list):
        problems = payload
    elif isinstance(payload, dict):
        problems = payload.get("problems")
    else:
        logger.error(
            "Problem config must be either a JSON list or an object with a "
            "`problems` list."
        )
        raise SystemExit(1)

    if not isinstance(problems, list) or not all(
        isinstance(problem, str) for problem in problems
    ):
        logger.error(
            "Problem config must define `problems` as a JSON list of strings."
        )
        raise SystemExit(1)

    problem_folders = []
    seen = set()
    for problem in problems:
        folder_name = problem.strip()
        if not folder_name or folder_name in seen:
            continue

        folder_path = None
        for eval_dir in eval_dirs:
            candidate = eval_dir / folder_name
            if candidate.is_dir():
                folder_path = candidate
                break

        if folder_path is None:
            logger.warning(
                f"Configured problem folder not found in any evaluation directory: {folder_name}"
            )
            continue

        problem_folders.append(folder_path)
        seen.add(folder_name)

    return problem_folders


def load_summary_tables(eval_dirs: list[Path]) -> dict[Path, pd.DataFrame]:
    """Load summary tables for each evaluation directory."""

    summary_tables = {}
    for eval_dir in eval_dirs:
        summary_csv = eval_dir / "summary_by_problem.csv"
        if not summary_csv.exists():
            logger.error(f"summary_by_problem.csv not found in {eval_dir}")
            raise SystemExit(1)

        summary_tables[eval_dir] = pd.read_csv(summary_csv)

    return summary_tables


def pdb_passes_checks(pdb_path: Path) -> bool:
    """Validate PDB using MDAnalysis before copying for simulation."""

    try:
        universe = mda.Universe(str(pdb_path))
    except Exception as exc:
        logger.warning(f"Failed to read PDB with MDAnalysis: {pdb_path} ({exc})")
        return False

    # 1) Only standard residues (no UNK/nonstandard resnames)
    resnames = {resname.upper() for resname in universe.residues.resnames}
    nonstandard = sorted(resnames - STANDARD_RESIDUES)
    if nonstandard:
        logger.info(
            f"Skipping {pdb_path}: nonstandard residues found {nonstandard}."
        )
        return False

    # 2) Must contain sidechain atoms (not backbone-only)
    atom_names = {name.upper() for name in universe.atoms.names}
    if atom_names.issubset(BACKBONE_ATOMS):
        logger.info(f"Skipping {pdb_path}: backbone-only atoms detected.")
        return False

    # 3) Only one chain and exactly one TER record in the file
    chain_ids = getattr(universe.atoms, "chainIDs", None)
    if chain_ids is None:
        chain_ids = universe.atoms.segids
    chain_ids = {str(cid).strip() for cid in chain_ids if cid is not None}
    if len(chain_ids) > 1:
        logger.info(
            f"Skipping {pdb_path}: multiple chains detected {sorted(chain_ids)}."
        )
        return False

    try:
        with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
            ter_count = sum(1 for line in handle if line.startswith("TER"))
    except OSError as exc:
        logger.warning(f"Failed to read PDB for TER count: {pdb_path} ({exc})")
        return False

    if ter_count != 1:
        logger.info(
            f"Skipping {pdb_path}: expected 1 TER record, found {ter_count}."
        )
        return False

    return True



def success_mask_from_column(column: pd.Series) -> pd.Series:
    """Return a boolean Success mask that tolerates bool, numeric, and string CSV values."""

    def parse_success(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))

        normalized = str(value).strip().lower()
        return normalized in {"1", "true", "t", "yes", "y"}

    return column.map(parse_success)


def resolve_result_path(
    raw_path: object,
    problem_dir: Path,
    problem_name: str,
    tail_parts: int | None = None,
) -> Path:
    """Resolve result paths even when evaluation folders were moved."""

    path = Path(str(raw_path)).expanduser()
    if path.exists():
        return path

    if not path.is_absolute():
        return problem_dir / path

    parts = path.parts
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == problem_name:
            rel_parts = parts[idx + 1 :]
            if rel_parts:
                return problem_dir / Path(*rel_parts)
            return problem_dir

    if tail_parts is not None and len(parts) >= tail_parts:
        return problem_dir / Path(*parts[-tail_parts:])

    return path


def derive_backbone_path_from_sample(sample_path: Path) -> Path | None:
    """Derive <variant>/<variant>.pdb from <variant>/self_consistency/esmf/sample_X.pdb."""

    try:
        variant_dir = sample_path.parents[2]
    except IndexError:
        return None

    return variant_dir / f"{variant_dir.name}.pdb"


def substitute_backbone_residue_names(
    source_pdb: Path,
    sequence: str,
    output_pdb: Path,
) -> None:
    """Copy a backbone PDB while replacing residue names from a one-letter sequence."""

    sequence = sequence.strip().upper()
    invalid = sorted(set(sequence) - set(ONE_TO_THREE))
    if invalid:
        raise ValueError(f"Sequence contains unsupported residue codes: {invalid}")

    lines = source_pdb.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    residue_keys = []
    seen = set()
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        key = (line[21:22], line[22:26], line[26:27])
        if key not in seen:
            residue_keys.append(key)
            seen.add(key)

    if len(residue_keys) != len(sequence):
        raise ValueError(
            f"Backbone residue count ({len(residue_keys)}) does not match "
            f"selected sequence length ({len(sequence)}) for {source_pdb}."
        )

    resname_by_key = {
        residue_key: ONE_TO_THREE[aa]
        for residue_key, aa in zip(residue_keys, sequence, strict=True)
    }

    output_lines = []
    for line in lines:
        if line.startswith("ATOM"):
            key = (line[21:22], line[22:26], line[26:27])
            resname = resname_by_key.get(key)
            if resname is not None:
                line = f"{line[:17]}{resname:>3}{line[20:]}"
        output_lines.append(line)

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("".join(output_lines), encoding="utf-8")


def normalize_single_ter(pdb_path: Path) -> None:
    """Ensure a PDB has exactly one TER after the final atom record."""

    lines = pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    body = [line for line in lines if not line.startswith(("TER", "END"))]

    last_atom_idx = None
    for idx, line in enumerate(body):
        if line.startswith(("ATOM", "HETATM")):
            last_atom_idx = idx

    if last_atom_idx is None:
        pdb_path.write_text("".join(body), encoding="utf-8")
        return

    body.insert(last_atom_idx + 1, "TER\n")
    body.append("END\n")
    pdb_path.write_text("".join(body), encoding="utf-8")


def remove_glycine_cb_atoms(pdb_path: Path) -> int:
    """Remove impossible glycine CB atoms from a PDB file."""

    lines = pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    cleaned_lines = []
    removed = 0

    for line in lines:
        if (
            line.startswith("ATOM")
            and line[12:16].strip().upper() == "CB"
            and line[17:20].strip().upper() == "GLY"
        ):
            removed += 1
            continue
        cleaned_lines.append(line)

    if removed:
        pdb_path.write_text("".join(cleaned_lines), encoding="utf-8")

    return removed


def run_pulchra(pulchra_bin: Path, input_pdb: Path, log_file: Path) -> Path:
    """Run PULCHRA and return its rebuilt PDB path."""

    if input_pdb.suffix.lower() == ".pdb":
        rebuilt_pdb = input_pdb.with_name(f"{input_pdb.stem}.rebuilt.pdb")
    else:
        rebuilt_pdb = input_pdb.with_name(f"{input_pdb.name}.rebuilt.pdb")

    if rebuilt_pdb.exists():
        rebuilt_pdb.unlink()

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [str(pulchra_bin), str(input_pdb)],
            cwd=input_pdb.parent,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )

    if not rebuilt_pdb.exists():
        raise FileNotFoundError(f"PULCHRA did not create expected output: {rebuilt_pdb}")

    normalize_single_ter(rebuilt_pdb)
    return rebuilt_pdb


def main() -> None:
    # -----------------------------
    # Parse command-line arguments
    # -----------------------------
    args = parse_args()
    eval_dirs = normalize_eval_dirs(args.eval_dirs)
    out_dir = Path(args.out_dir).resolve()
    include_failures = args.include_failures
    include_successes = bool(args.include_successes)
    failure_backbone_log_dir = (
        args.failure_backbone_log_dir.resolve()
        if args.failure_backbone_log_dir is not None
        else out_dir / "failure_backbones"
    )
    pulchra_bin = args.pulchra_bin.expanduser().resolve()

    if include_failures and not pulchra_bin.is_file():
        logger.error(f"PULCHRA executable not found: {pulchra_bin}")
        raise SystemExit(1)
    if include_failures and not pulchra_bin.stat().st_mode & 0o111:
        logger.error(f"PULCHRA path is not executable: {pulchra_bin}")
        raise SystemExit(1)

    missing_eval_dirs = [eval_dir for eval_dir in eval_dirs if not eval_dir.exists()]
    if missing_eval_dirs:
        for eval_dir in missing_eval_dirs:
            logger.error(f"EVAL_DIR does not exist: {eval_dir}")
        raise SystemExit(1)

    non_directory_eval_dirs = [eval_dir for eval_dir in eval_dirs if not eval_dir.is_dir()]
    if non_directory_eval_dirs:
        for eval_dir in non_directory_eval_dirs:
            logger.error(f"EVAL_DIR is not a directory: {eval_dir}")
        raise SystemExit(1)

    problem_dirs = load_problem_folders(eval_dirs, args.problem_config)
    summary_tables = load_summary_tables(eval_dirs)

    logger.info(f"Using {len(eval_dirs)} evaluation directories:")
    for eval_dir in eval_dirs:
        logger.info(f"  - {eval_dir}")
    logger.info(f"Using output directory    : {out_dir}")
    if not include_successes:
        logger.info("Skipping problems with Num_Solutions > 0.")
    if include_failures:
        logger.info("Including problems with Num_Solutions <= 0.")
        logger.info(f"Failure selection mode : {args.failures_mode}")
        logger.info(f"PULCHRA executable    : {pulchra_bin}")
        logger.info(f"Failure backbone logs : {failure_backbone_log_dir}")
    if args.problem_config is not None:
        logger.info(
            f"Restricting run to {len(problem_dirs)} problem(s) from config."
        )

    motif_pdbs_dir = out_dir / "motif_pdbs"
    motif_pdbs_dir.mkdir(parents=True, exist_ok=True)
    if include_failures:
        failure_backbone_log_dir.mkdir(parents=True, exist_ok=True)

    motif_info = {}
    selection_rows = []

    # -----------------------------
    # Iterate through problems
    # -----------------------------
    for entry in problem_dirs:
        problem_name = entry.name
        logger.info(f"Processing {problem_name}...")

        df_summary = summary_tables.get(entry.parent)
        if df_summary is None:
            logger.warning(
                f"Skipping {problem_name}: no summary_by_problem.csv loaded for {entry.parent}."
            )
            continue

        # check if folder exists
        if not entry.is_dir():
            logger.debug(
                f"Skipping {problem_name}: {entry} does not exist."
            )
            continue

        results_path = entry / "esm_complete_results.csv"
        if not results_path.exists():
            logger.warning(
                f"Skipping {problem_name}: esm_complete_results.csv not found."
            )
            continue

        results = pd.read_csv(results_path)

        # Check if problem is in summary and has solutions
        match = df_summary.index[df_summary["Problem"] == problem_name]
        if len(match) == 0:
            logger.warning(
                f"Skipping {problem_name}: not found in summary_by_problem.csv."
            )
            continue

        summary_idx = match[0]
        num_solutions = df_summary.loc[summary_idx, "Num_Solutions"]
        use_success_filter = num_solutions > 0
        if use_success_filter and not include_successes:
            logger.info(
                f"Skipping {problem_name}: Num_Solutions = {num_solutions} "
                "and --include-successes is 0."
            )
            continue
        if not use_success_filter and not include_failures:
            logger.info(
                f"Skipping {problem_name}: Num_Solutions = {num_solutions} "
                "(no successes)."
            )
            continue
        if not use_success_filter and include_failures:
            logger.info(
                f"Including {problem_name} with Num_Solutions = {num_solutions}."
            )

        # Ensure required columns
        required_cols = {"tm_score", "Success", "sample_path", "backbone_path"}
        if not required_cols.issubset(results.columns):
            missing = required_cols - set(results.columns)
            logger.warning(
                f"Skipping {problem_name}: missing required columns {missing}."
            )
            continue
        if not use_success_filter and "sequence" not in results.columns:
            logger.warning(
                f"Skipping failed problem {problem_name}: missing required column 'sequence'."
            )
            continue

        results = results.copy()
        results["tm_score"] = pd.to_numeric(results["tm_score"], errors="coerce")
        results = results.dropna(subset=["tm_score"])
        if results.empty:
            logger.warning(f"Skipping {problem_name}: no numeric tm_score values found.")
            continue

        success_mask = success_mask_from_column(results["Success"])
        if use_success_filter:
            candidate_results = results.loc[success_mask].copy()
            if candidate_results.empty:
                logger.warning(
                    f"Skipping {problem_name}: Num_Solutions > 0 but no successful rows found."
                )
                continue
            best_idx = candidate_results["tm_score"].idxmax()
            selection_mode = "success_high_tm"
        elif args.failures_mode == "low_tm":
            best_idx = results["tm_score"].idxmin()
            selection_mode = "failure_low_tm"
        else:
            best_idx = results["tm_score"].idxmax()
            selection_mode = "failure_high_tm"

        selected_row = results.loc[best_idx]
        sample_path = Path(str(selected_row["sample_path"]))
        backbone_path = Path(str(selected_row["backbone_path"]))
        backbone_key = backbone_path.stem
        best_tm_score = float(selected_row["tm_score"])
        solved = bool(success_mask.loc[best_idx])
        selected_sequence = (
            str(selected_row["sequence"]).strip()
            if "sequence" in results.columns and not pd.isna(selected_row.get("sequence"))
            else ""
        )

        # Load motif metadata
        motif_info_file = entry / "motif_info.json"
        if not motif_info_file.exists():
            logger.warning(
                f"Skipping {problem_name}: motif_info.json not found."
            )
            continue

        with motif_info_file.open("r", encoding="utf-8") as f:
            motif_data = json.load(f)

        if backbone_key not in motif_data:
            logger.warning(
                f"Skipping {problem_name}: backbone key '{backbone_key}' "
                f"not found in motif_info.json."
            )
            continue

        motif_info[problem_name] = motif_data[backbone_key]

        out_file = motif_pdbs_dir / f"{problem_name}.pdb"
        source_path = ""
        source_backbone_path = ""
        reconstructed_pdb = ""
        backbone_log_dir = ""
        reconstruction = "none"

        if use_success_filter:
            sample_path = resolve_result_path(
                sample_path, entry, problem_name, tail_parts=4
            )

            if not sample_path.exists():
                logger.warning(
                    f"Sample PDB for {problem_name} not found at {sample_path}; skipping."
                )
                continue

            if not pdb_passes_checks(sample_path):
                logger.info(
                    f"Skipping {problem_name}: PDB failed validation checks at {sample_path}."
                )
                continue

            shutil.copy(sample_path, out_file)
            source_path = str(sample_path)
            source_backbone_path = str(
                resolve_result_path(backbone_path, entry, problem_name)
            )
            logger.info(f"Copied best PDB to {out_file}")
        else:
            if not selected_sequence:
                logger.warning(
                    f"Skipping failed problem {problem_name}: selected row has no sequence."
                )
                continue

            backbone_path = resolve_result_path(backbone_path, entry, problem_name)
            if not backbone_path.exists():
                sample_for_fallback = resolve_result_path(
                    sample_path, entry, problem_name, tail_parts=4
                )
                fallback_backbone = derive_backbone_path_from_sample(sample_for_fallback)
                if fallback_backbone is not None and fallback_backbone.exists():
                    backbone_path = fallback_backbone

            if not backbone_path.exists():
                logger.warning(
                    f"Backbone PDB for failed problem {problem_name} not found at "
                    f"{backbone_path}; skipping."
                )
                continue

            problem_log_dir = failure_backbone_log_dir / problem_name
            problem_log_dir.mkdir(parents=True, exist_ok=True)
            backbone_log_dir = str(problem_log_dir)

            selected_payload = {
                "problem": problem_name,
                "selection_mode": selection_mode,
                "row_index": int(best_idx) if isinstance(best_idx, int) else str(best_idx),
                "tm_score": best_tm_score,
                "sample_path": str(sample_path),
                "resolved_backbone_path": str(backbone_path),
                "sequence": selected_sequence,
                "row": selected_row.to_dict(),
            }
            with (problem_log_dir / "selected_row.json").open("w", encoding="utf-8") as handle:
                json.dump(selected_payload, handle, ensure_ascii=False, indent=2, default=str)

            original_bb = problem_log_dir / f"{problem_name}.{selection_mode}.bb_original.pdb"
            substituted_bb = problem_log_dir / f"{problem_name}.{selection_mode}.bb_substituted.pdb"
            shutil.copy(backbone_path, original_bb)

            try:
                substitute_backbone_residue_names(
                    original_bb, selected_sequence, substituted_bb
                )
                removed = remove_glycine_cb_atoms(substituted_bb)
                if removed:
                    logger.info(
                        f"Removed {removed} glycine CB atom(s) before PULCHRA: "
                        f"{substituted_bb}"
                    )
                rebuilt_pdb = run_pulchra(
                    pulchra_bin, substituted_bb, problem_log_dir / "pulchra.log"
                )
                removed = remove_glycine_cb_atoms(rebuilt_pdb)
                if removed:
                    logger.info(
                        f"Removed {removed} glycine CB atom(s) after PULCHRA: "
                        f"{rebuilt_pdb}"
                    )
                    normalize_single_ter(rebuilt_pdb)
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                logger.warning(
                    f"Skipping failed problem {problem_name}: PULCHRA reconstruction "
                    f"failed ({exc})."
                )
                continue

            if not pdb_passes_checks(rebuilt_pdb):
                logger.info(
                    f"Skipping {problem_name}: PULCHRA rebuilt PDB failed validation "
                    f"checks at {rebuilt_pdb}."
                )
                continue

            if out_file.exists():
                out_file.unlink()
            shutil.move(str(rebuilt_pdb), out_file)
            source_path = str(sample_path)
            source_backbone_path = str(backbone_path)
            reconstructed_pdb = str(out_file)
            reconstruction = "pulchra"
            logger.info(
                f"Reconstructed failed-case backbone with PULCHRA and moved to {out_file}"
            )

        selection_rows.append(
            {
                "problem": problem_name,
                "tm_score": best_tm_score,
                "solved": solved,
                "selection_mode": selection_mode,
                "source_path": source_path,
                "source_sample_path": str(sample_path),
                "source_backbone_path": source_backbone_path,
                "sequence": selected_sequence,
                "reconstruction": reconstruction,
                "reconstructed_pdb": reconstructed_pdb,
                "backbone_log_dir": backbone_log_dir,
            }
        )

    # Write combined motif metadata
    out_info_file = out_dir / "motif_info.json"
    with out_info_file.open("w", encoding="utf-8") as f:
        json.dump(motif_info, f, ensure_ascii=False, indent=4)

    out_csv = motif_pdbs_dir / "motif_pdbs.csv"
    if selection_rows:
        pd.DataFrame(selection_rows).to_csv(out_csv, index=False)
        logger.info(f"Wrote selection CSV to {out_csv}")
    else:
        logger.info("No selections were made; CSV not written.")

    logger.info(
        f"Done. Wrote {len(motif_info)} motif metadata entries to {out_info_file}"
    )


if __name__ == "__main__":
    configure_logging()
    main()
