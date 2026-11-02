#!/bin/bash
# md_scheduler.sh
#
# Simple scheduler to keep at most N simulations running at the same time.
#
# Usage:
#   ./md_scheduler.sh [SIM_ROOT] [MAX_PARALLEL]
#
# States per simulation directory:
#   - ERROR      → permanently skip this sample
#   - DONE       → finished, ignore
#   - RUNNING    → currently active
#   - none       → PENDING (needs to be started)
#
# This scheduler should be called when:
#   1) Beginning the pipeline (initial seeding)
#   2) At the END of the final MD job for each sample.
#

set -euo pipefail

SIM_ROOT="${1:-./simulations}"
MAX_PARALLEL="${2:-4}"

SIM_ROOT="$(readlink -f "$SIM_ROOT")"
LOCKFILE="${SIM_ROOT}/.md_scheduler.lock"

mkdir -p "$SIM_ROOT"

(
    # Acquire lock so only one scheduler runs at a time.
    flock -n 200 || exit 0

    echo "[md_scheduler] SIM_ROOT=${SIM_ROOT}, MAX_PARALLEL=${MAX_PARALLEL}"

    # Snapshot Slurm job names for RUNNING flag reconciliation.
    have_squeue=0
    job_names=()
    if command -v squeue >/dev/null 2>&1; then
        if squeue -u "$USER" -h -o "%j" >/dev/null 2>&1; then
            mapfile -t job_names < <(squeue -u "$USER" -h -o "%j")
            have_squeue=1
        else
            echo "[md_scheduler] WARNING: squeue failed; skipping RUNNING reconciliation."
        fi
    else
        echo "[md_scheduler] WARNING: squeue not found; skipping RUNNING reconciliation."
    fi

    has_job_for_folder() {
        local folder="$1"
        local name
        for name in "${job_names[@]}"; do
            if [[ "$name" == "setup_${folder}" || "$name" == "eq_${folder}" || "$name" == "md_${folder}" || "$name" == "${folder}" ]]; then
                return 0
            fi
        done
        return 1
    }

    # Count how many samples are in RUNNING state (not DONE yet)
    current_running=0
    mapfile -t sim_dirs < <(find "$SIM_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

    # Count RUNNING samples (skip ERROR and DONE)
    for sim_dir in "${sim_dirs[@]}"; do
        folder_name="$(basename "$sim_dir")"

        if [[ -f "${sim_dir}/RUNNING" && "$have_squeue" -eq 1 ]]; then
            if ! has_job_for_folder "$folder_name"; then
                echo "[md_scheduler] RUNNING set but no job in queue; resubmitting: ${sim_dir}"
                resubmit_script="${sim_dir}/production/check_and_resubmit.sh"
                if [[ -f "$resubmit_script" ]]; then
                    if [[ ! -x "$resubmit_script" ]]; then
                        chmod +x "$resubmit_script"
                    fi
                    (
                        cd "$(dirname "$resubmit_script")" || exit 1
                        bash $resubmit_script
                    )
                else
                    echo "[md_scheduler] WARNING: ${resubmit_script} not found; keeping RUNNING flag."
                fi
            fi
        fi

        
        # Skip ERROR folders completely
        if [[ -f "${sim_dir}/ERROR" ]]; then
            echo "[md_scheduler] Skipping ERROR sample: ${sim_dir}"
            continue
        fi

        # Skip DONE
        if [[ -f "${sim_dir}/DONE" ]]; then
            continue
        fi

        # Count RUNNING
        if [[ -f "${sim_dir}/RUNNING" ]]; then
            current_running=$(( current_running + 1 ))
        fi
    done

    echo "[md_scheduler] Currently running samples: ${current_running}"

    # If we are already at or above the limit, nothing to do.
    if (( current_running >= MAX_PARALLEL )); then
        echo "[md_scheduler] Max parallel limit reached; exiting."
        exit 0
    fi

    # How many more samples can we start?
    free_slots=$((MAX_PARALLEL - current_running))
    echo "[md_scheduler] Free slots: ${free_slots}"

    # Start up to `free_slots` PENDING simulations (no RUNNING or DONE markers)
    for sim_dir in "${sim_dirs[@]}"; do
        [[ "$free_slots" -le 0 ]] && break

        # Skip ERROR folders completely
        if [[ -f "${sim_dir}/ERROR" ]]; then
            echo "[md_scheduler] Skipping ERROR sample: ${sim_dir}"
            continue
        fi

        # Skip already finished
        if [[ -f "${sim_dir}/DONE" ]]; then
            continue
        fi

        # Skip already running
        if [[ -f "${sim_dir}/RUNNING" ]]; then
            continue
        fi

        # This is a PENDING sample: submit its make_structure.sh
        job_script="${sim_dir}/make_structure.sh"
        if [[ ! -x "$job_script" && -f "$job_script" ]]; then
            chmod +x "$job_script"
        fi

        if [[ ! -f "$job_script" ]]; then
            echo "[md_scheduler] WARNING: No make_structure.sh in ${sim_dir}, skipping."
            continue
        fi

        echo "[md_scheduler] Submitting: ${job_script}"
        (
            cd "$sim_dir"
            sbatch make_structure.sh
        )

        # Mark it as RUNNING (sample-wide, not just single job)
        touch "${sim_dir}/RUNNING"

        free_slots=$((free_slots - 1))
        current_running=$((current_running + 1))
    done

    echo "[md_scheduler] Done. Currently running: ${current_running}"

) 200>"${LOCKFILE}"
