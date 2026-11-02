#!/bin/bash
#SBATCH --job-name="analysis_mbMD"
#SBATCH --account=profound
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=12:00:00       
#SBATCH --partition=booster

module purge
module load Stages/2025
module load GCC/13.3.0  ParaStationMPI/5.11.0-1
module load GROMACS/2024.3

source /e/project1/profound/schaeffler1/venv_MB-pipeline/activate.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="${1:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
PYTHON="${PYTHON:-python}"
SIM_DIR="${SIM_DIR:-simulations}"

ALIGN_SELECTION="${ALIGN_SELECTION:-backbone}"
ALIGN_WEIGHTS="${ALIGN_WEIGHTS:-mass}"
RMSD_SELECTION="${RMSD_SELECTION:-backbone}"
CONTACT_CUTOFF="${CONTACT_CUTOFF:-3.0}"
CONTACT_INTERVAL="${CONTACT_INTERVAL:-10}"
JACCARD_CONTACT_THRESHOLD="${JACCARD_CONTACT_THRESHOLD:-0.5}"

have_no_water_outputs() {
  local prod_dir="$1"
  local system="$2"

  [[ -f "$prod_dir/md_${system}_noW.tpr" ]] &&
    [[ -f "$prod_dir/md_${system}_noW.gro" ]] &&
    [[ -f "$prod_dir/md_${system}_noW.trr" ]]
}

prepare_no_water_outputs() {
  local sim_root="$ROOT/$SIM_DIR"
  local remove_water_template="$ROOT/templates/remove_water.sh"
  local sim_dir prod_dir system escaped_system input

  if [[ ! -d "$sim_root" ]]; then
    echo "[analysis] ERROR: Simulation directory not found: $sim_root" >&2
    exit 1
  fi
  if [[ ! -f "$remove_water_template" ]]; then
    echo "[analysis] ERROR: remove_water template not found: $remove_water_template" >&2
    exit 1
  fi

  shopt -s nullglob
  for sim_dir in "$sim_root"/*; do
    [[ -d "$sim_dir" ]] || continue
    system="$(basename "$sim_dir")"
    prod_dir="$sim_dir/production"

    if [[ ! -d "$prod_dir" ]]; then
      echo "[analysis] Skipping $system, missing production directory: $prod_dir"
      continue
    fi

    if have_no_water_outputs "$prod_dir" "$system"; then
      echo "[analysis] Found no-water outputs for $system."
      continue
    fi

    echo "[analysis] Missing no-water outputs for $system; running remove_water.sh."
    for input in "md_${system}.tpr" "md_${system}.trr"; do
      if [[ ! -f "$prod_dir/$input" ]]; then
        echo "[analysis] ERROR: Cannot remove water for $system; missing $prod_dir/$input" >&2
        exit 1
      fi
    done

    cp "$remove_water_template" "$prod_dir/remove_water.sh"
    (
      cd "$prod_dir"
      escaped_system="${system//&/\&}"
      sed -i "s|^sysname=.*|sysname='${escaped_system}'|" remove_water.sh
      sed -i "0,/^#SBATCH --job-name=/s|^#SBATCH --job-name=.*|#SBATCH --job-name=noW_${escaped_system}|" remove_water.sh
      bash ./remove_water.sh
    )

    if ! have_no_water_outputs "$prod_dir" "$system"; then
      echo "[analysis] ERROR: remove_water.sh did not produce all no-water outputs for $system" >&2
      exit 1
    fi
  done
  shopt -u nullglob
}

prepare_no_water_outputs

"$PYTHON" "$ROOT/analysis_scripts/process_trajectory.py" \
  --root "$ROOT" \
  --sim-dir "$SIM_DIR" \
  --align-selection "$ALIGN_SELECTION" \
  --weights "$ALIGN_WEIGHTS"

"$PYTHON" "$ROOT/analysis_scripts/calc_rmsd.py" \
  --root "$ROOT" \
  --sim-dir "$SIM_DIR" \
  --selection "$RMSD_SELECTION"

"$PYTHON" "$ROOT/analysis_scripts/calc_contacts.py" \
  --root "$ROOT" \
  --sim-dir "$SIM_DIR" \
  --cutoff "$CONTACT_CUTOFF" \
  --interval "$CONTACT_INTERVAL"

"$PYTHON" "$ROOT/analysis_scripts/calc_fes.py" \
  --root "$ROOT" \
  --sim-dir "$SIM_DIR"

"$PYTHON" "$ROOT/analysis_scripts/summarize_analysis.py" \
  --root "$ROOT" \
  --contact-threshold "$JACCARD_CONTACT_THRESHOLD"
