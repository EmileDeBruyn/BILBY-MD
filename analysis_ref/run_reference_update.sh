#!/bin/bash
#SBATCH --job-name="analysis_mbMD_ref"
#SBATCH --account=loki
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=48
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=06:00:00
#SBATCH --partition=booster

source /e/project1/profound/schaeffler1/venv_MB-pipeline/activate.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DEFAULT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
ROOT="${1:-${SLURM_SUBMIT_DIR:-$ROOT_DEFAULT}}"
PYTHON="${PYTHON:-python}"

ALIGN_SELECTION="${ALIGN_SELECTION:-backbone}"
ALIGN_WEIGHTS="${ALIGN_WEIGHTS:-mass}"
RMSD_SELECTION="${RMSD_SELECTION:-backbone}"
CONTACT_CUTOFF="${CONTACT_CUTOFF:-3.0}"
CONTACT_INTERVAL="${CONTACT_INTERVAL:-10}"
TICA_LAGTIME="${TICA_LAGTIME:-10}"
REF_BUILD_DIR="${REF_BUILD_DIR:-$ROOT/analysis_ref/update_inputs}"

mkdir -p "$REF_BUILD_DIR"

"$PYTHON" "$ROOT/analysis_scripts/process_trajectory.py" \
  --root "$ROOT" \
  --align-selection "$ALIGN_SELECTION" \
  --weights "$ALIGN_WEIGHTS"

"$PYTHON" "$ROOT/analysis_scripts/calc_rmsd.py" \
  --root "$ROOT" \
  --analysis-dir "$REF_BUILD_DIR" \
  --selection "$RMSD_SELECTION"

"$PYTHON" "$ROOT/analysis_scripts/calc_contacts.py" \
  --root "$ROOT" \
  --analysis-dir "$REF_BUILD_DIR" \
  --cutoff "$CONTACT_CUTOFF" \
  --interval "$CONTACT_INTERVAL"

"$PYTHON" "$ROOT/analysis_ref/calc_fes_scratch.py" \
  --root "$ROOT" \
  --output "$REF_BUILD_DIR/results_tica-pca_ref_scratch.pkl" \
  --tica-lagtime "$TICA_LAGTIME"

"$PYTHON" "$ROOT/analysis_ref/merge_reference_updates.py" \
  --root "$ROOT" \
  --source-dir "$REF_BUILD_DIR"
