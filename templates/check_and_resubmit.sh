#!/bin/bash
# --- configuration ---------------------------------------------------------
sysname="system"                     # or pass as arg: sysname="${1:-system}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$SCRIPT_DIR" || exit 1

logfile="md_${sysname}.log"          # GROMACS stdout/stderr file
mdpfile="NPT_production.mdp"         # production .mdp with nsteps
continue_script="${IDP_CONTINUE_SCRIPT:-jupiter_run.sh}" # what to sbatch if we need to continue
MAX_PARALLEL=10
# ---------------------------------------------------------------------------

# Basic sanity checks
if [[ ! -f "$logfile" ]]; then
    echo "[check] ERROR: Log file '$logfile' not found. Not resubmitting."
    rm -f ../RUNNING
    touch ../ERROR
    # restart scheduler
    echo "[check] Running scheduler."
    bash "$SIM_ROOT/md_scheduler.sh" "$SIM_ROOT/simulations" "$MAX_PARALLEL" >> $SIM_ROOT/md_scheduler.log 2>&1
    exit 1
fi

if [[ ! -f "$mdpfile" ]]; then
    echo "[check] ERROR: MDP file '$mdpfile' not found. Not resubmitting."
    rm -f ../RUNNING
    touch ../ERROR
    # restart scheduler
    echo "[check] Running scheduler."
    bash "$SIM_ROOT/md_scheduler.sh" "$SIM_ROOT/simulations" "$MAX_PARALLEL" >> $SIM_ROOT/md_scheduler.log 2>&1
    exit 1
fi

# 1. Fatal error? -> never automatically resubmit
if grep -q "Fatal error" "$logfile"; then
    echo "[check] Fatal error detected in $logfile, not resubmitting."
    rm -f ../RUNNING
    touch ../ERROR
    # restart scheduler
    echo "[check] Running scheduler."
    bash "$SIM_ROOT/md_scheduler.sh" "$SIM_ROOT/simulations" "$MAX_PARALLEL" >> $SIM_ROOT/md_scheduler.log 2>&1
    exit 1
fi

# 2. Get target step from nsteps in the .mdp
#    e.g. line:  nsteps = 50000000000 ; 10 us
target_step=$(awk '
    $1 == "nsteps" {
        # collect numeric field after "nsteps" and "=" (robust against comments)
        for (i = 1; i <= NF; i++) {
            if ($i ~ /^[0-9]+$/) {
                print $i;
                exit;
            }
        }
    }
' "$mdpfile")

if [[ -z "$target_step" ]]; then
    echo "[check] ERROR: Could not extract nsteps from $mdpfile. Not resubmitting."
    rm -f ../RUNNING
    touch ../ERROR
    # restart scheduler
    echo "[check] Running scheduler."
    bash "$SIM_ROOT/md_scheduler.sh" "$SIM_ROOT/simulations" "$MAX_PARALLEL" >> $SIM_ROOT/md_scheduler.log 2>&1
    exit 1
fi

# 3. Get last reported MD step from the log
#    We look for lines like:
#       50799999   101599.99840  ...
#    i.e. leading spaces, an integer step, then a time.
current_step=$(awk '
    # match lines starting with optional spaces, then an integer, then a number (time)
    /^[[:space:]]*[0-9]+[[:space:]]+[0-9.]+/ {
        step = $1
    }
    END {
        if (step == "") step = 0
        print step
    }
' "$logfile")

echo "[check] target_step = $target_step, current_step = $current_step"

# 4. Decide: finished vs. continue
if (( current_step >= target_step )); then
    echo "[check] Target step reached (or exceeded). Simulation finished, no resubmission."
    rm -f ../RUNNING
    touch ../DONE
    # restart scheduler
    echo "[check] Running scheduler."
    bash "$SIM_ROOT/md_scheduler.sh" "$SIM_ROOT/simulations" "$MAX_PARALLEL" >> $SIM_ROOT/md_scheduler.log 2>&1
    sbatch remove_water.sh
    exit 0
fi

# Optional: informative note if we *also* see 'Finished mdrun', but not done yet
if grep -q "Finished mdrun" "$logfile"; then
    echo "[check] NOTE: 'Finished mdrun' found, but current_step < target_step."
    echo "[check]       Likely stopped due to -maxh. Continuing from checkpoint."
fi

# 5. Not finished, no fatal error -> resubmit continuation
echo "[check] Not yet at target step. Submitting continuation job..."
sbatch "$continue_script"
submit_status=$?

if (( submit_status != 0 )); then
    echo "[check] ERROR: sbatch failed with status $submit_status."
    rm -f ../RUNNING
    touch ../ERROR
    # restart scheduler
    echo "[check] Running scheduler."
    bash "$SIM_ROOT/md_scheduler.sh" "$SIM_ROOT/simulations" "$MAX_PARALLEL" >> $SIM_ROOT/md_scheduler.log 2>&1
    exit $submit_status
fi

exit 0