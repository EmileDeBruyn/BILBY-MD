#!/bin/bash -x
#SBATCH --job-name=j4_system
#SBATCH --account=profound
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=11:55:00
#SBATCH --partition=booster
#SBATCH --exclusive

# Run or continue one production simulation on a full JUWELS/JUPITER booster node.

sysname="${SYSNAME:-system}"
out_file="md_${sysname}.out"
maxh="${IDP_MAXH:-11.5}"
mdrun_bin="${IDP_MDRUN_BIN:-gmx}"
mdrun_thread_mode="${IDP_MDRUN_THREAD_MODE:-nt}"
mdrun_threads="${IDP_MDRUN_THREADS:-${SLURM_CPUS_PER_TASK:-288}}"
gpu_ids="${IDP_GPU_IDS:-0123}"

module purge
module load Stages/2025
module load GCC/13.3.0  ParaStationMPI/5.11.0-1
module load GROMACS/2024.3

export SYSNAME="$sysname"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GMX_ENABLE_DIRECT_GPU_COMM="${GMX_ENABLE_DIRECT_GPU_COMM:-1}"
export GMX_GPU_DD_COMMS="${GMX_GPU_DD_COMMS:-true}"
export GMX_GPU_PME_PP_COMMS="${GMX_GPU_PME_PP_COMMS:-true}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-288}}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

write_error() {
    local message="$1"
    {
        echo "timestamp=$(date -Is)"
        echo "script=$(basename "$0")"
        echo "directory=$(pwd)"
        echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
        echo "gpu_ids=${gpu_ids}"
        echo "message=${message}"
        echo
    } >> ../ERROR
    rm -f ../RUNNING
}

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        write_error "Required file missing: ${path}"
        exit 1
    fi
}

{
    echo "timestamp=$(date -Is)"
    echo "state=running"
    echo "script=$(basename "$0")"
    echo "directory=$(cd .. && pwd)"
    echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
    echo "gpu_ids=${gpu_ids}"
    echo "maxh=${maxh}"
    echo
} >> ../RUNNING

echo "[jupiter] Starting ${PWD} on GPUs ${gpu_ids} with ${mdrun_threads} MD threads using ${mdrun_bin}." &>> "$out_file"

require_file "NPT_production.mdp"

if [[ ! -f "md_${sysname}.tpr" ]]; then
    require_file "eq_NPT_${sysname}.gro"
    require_file "eq_NPT_${sysname}.cpt"
    require_file "../topol.top"

    gmx grompp -f NPT_production.mdp \
               -c eq_NPT_${sysname}.gro \
               -p ../topol.top \
               -t eq_NPT_${sysname}.cpt \
               -o md_${sysname}.tpr \
               -maxwarn 1 &>> "$out_file"
    status=$?
    if (( status != 0 )); then
        write_error "gmx grompp failed with exit code ${status}."
        exit $status
    fi
fi

mdrun_args=(
    -s md_${sysname}.tpr
    -deffnm md_${sysname}
    -nb gpu
    -pme gpu
    -bonded gpu
    -pin on
    -gpu_id "$gpu_ids"
    -maxh "$maxh"
)

case "$mdrun_thread_mode" in
    nt)
        mdrun_args+=(-nt "$mdrun_threads")
        ;;
    ntomp)
        mdrun_args+=(-ntomp "$mdrun_threads")
        ;;
    none)
        ;;
    *)
        write_error "Unsupported IDP_MDRUN_THREAD_MODE=${mdrun_thread_mode}. Use nt, ntomp, or none."
        exit 1
        ;;
esac

if [[ -f "md_${sysname}.cpt" ]]; then
    mdrun_args+=(-cpi md_${sysname}.cpt -append)
fi

"$mdrun_bin" mdrun "${mdrun_args[@]}" &>> "$out_file"
status=$?
if (( status != 0 )); then
    write_error "${mdrun_bin} mdrun failed with exit code ${status}."
    exit $status
fi

grep "mdrun" "$out_file" >> performance_md_${sysname}.out
grep "Performance" "$out_file" -B 4 >> performance_md_${sysname}.out
echo >> performance_md_${sysname}.out

if [[ -x "./post_processing.sh" ]]; then
    ./post_processing.sh &>> "$out_file"
elif [[ -f "./post_processing.sh" ]]; then
    bash ./post_processing.sh &>> "$out_file"
elif [[ -x "./check_and_resubmit.sh" ]]; then
    ./check_and_resubmit.sh &>> "$out_file"
elif [[ -f "./check_and_resubmit.sh" ]]; then
    bash ./check_and_resubmit.sh &>> "$out_file"
else
    write_error "Missing check_and_resubmit.sh."
    exit 1
fi

status=$?
if (( status != 0 )); then
    if [[ ! -f "../ERROR" ]]; then
        write_error "Check/resubmit step failed with exit code ${status}."
    fi
    exit $status
fi

exit 0
