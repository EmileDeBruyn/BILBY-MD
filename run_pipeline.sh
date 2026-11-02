#!/bin/bash

### file paths
EVAL_DIRS='["/p/scratch/profound/schaeffler1/motifbench_results_proteina/evaluation_XL"]'
OUT_DIR="./"
LOG_FILE="md_pipeline.log"

### motif specification
PROBLEM_CONFIG="./problem_selection.json"
INCLUDE_SUCCESSES=1
INCLUDE_FAILURES=1
FAILURES_MODE="high_tm"  # choices: high_tm, low_tm
PULCHRA_BIN="/e/project1/profound/schaeffler1/bin/pulchra"

### MD scheduler arguments
COMPUTE_PROJECT="profound"
MAX_PARALLEL=10

### run pipeline
source /path/to/venv/venv_BILBY-pipeline/activate.sh

# save old log files
if [ -e "$LOG_FILE" ]; then
    i=1
    while [ -e "$LOG_FILE-$i" ]; do
        i=$((i + 1))
    done
    mv "$LOG_FILE" "$LOG_FILE-$i"
fi

if [ -n "$PROBLEM_CONFIG" ]; then
    if [ ! -f "$PROBLEM_CONFIG" ]; then
        echo "ERROR: PROBLEM_CONFIG does not exist or is not a file: $PROBLEM_CONFIG" | tee -a "$LOG_FILE" >&2
        exit 1
    fi
fi

case "$INCLUDE_SUCCESSES" in
    0|1) ;;
    *)
        echo "ERROR: INCLUDE_SUCCESSES must be 0 or 1, got: $INCLUDE_SUCCESSES" | tee -a "$LOG_FILE" >&2
        exit 1
        ;;
esac

case "$FAILURES_MODE" in
    high_tm|low_tm) ;;
    *)
        echo "ERROR: FAILURES_MODE must be high_tm or low_tm, got: $FAILURES_MODE" | tee -a "$LOG_FILE" >&2
        exit 1
        ;;
esac

if [ "$INCLUDE_FAILURES" -eq 1 ] && [ ! -x "$PULCHRA_BIN" ]; then
    echo "ERROR: PULCHRA_BIN does not exist or is not executable: $PULCHRA_BIN" | tee -a "$LOG_FILE" >&2
    exit 1
fi

# evaluate MotifBench output and get best scaffold of solved cases
cmd=(python getMotifSamples.py "$EVAL_DIRS" "$OUT_DIR" "--include-successes" "$INCLUDE_SUCCESSES")
if [ -n "$PROBLEM_CONFIG" ]; then
    cmd+=("--problem-config" "$PROBLEM_CONFIG")
fi
if [ "$INCLUDE_FAILURES" -eq 1 ]; then
    cmd+=("--include-failures" "--failures-mode" "$FAILURES_MODE" "--pulchra-bin" "$PULCHRA_BIN")
fi
"${cmd[@]}" >> "$LOG_FILE" 2>&1

# set compute project in batch scripts
sed -i "/#SBATCH --account=/c\#SBATCH --account=${COMPUTE_PROJECT}" ./templates/make_structure.sh
sed -i "/#SBATCH --account=/c\#SBATCH --account=${COMPUTE_PROJECT}" ./templates/script_min_eqNVT.sh
sed -i "/#SBATCH --account=/c\#SBATCH --account=${COMPUTE_PROJECT}" ./templates/jupiter_run.sh
sed -i "/#SBATCH --account=/c\#SBATCH --account=${COMPUTE_PROJECT}" ./templates/remove_water.sh

# build md folder structure
mkdir -p simulations
for file in ./motif_pdbs/*
do
    filename="$(basename "$file")"
    basename_no_ext="${filename%.*}"
    extension="${filename##*.}"

    if [ "$extension" = "pdb" ]; then
        folder_name=$(printf '%s\n' "$basename_no_ext")
        sim_dir="./simulations/$folder_name"

        echo "$sim_dir"

        if [ -d "$sim_dir" ]; then
            echo "Skipping $sim_dir: already exists." | tee -a "$LOG_FILE"
            continue
        fi

        mkdir "$sim_dir"
        cp ./templates/make_structure.sh "$sim_dir"
        sed -i "/sysname='system'/c\sysname='$folder_name'" "$sim_dir/make_structure.sh"
        sed -i "0,/^#SBATCH --job-name=/s|^#SBATCH --job-name=.*|#SBATCH --job-name=setup_${folder_name}|" "$sim_dir/make_structure.sh"

        cp "$file" "$sim_dir/"
    fi

done

wait

# write MAX_PARALLEL also to check_and_submit.sh
sed -i "s/^MAX_PARALLEL=.*/MAX_PARALLEL=${MAX_PARALLEL}/" ./templates/check_and_resubmit.sh

bash ./md_scheduler.sh ./simulations $MAX_PARALLEL >> md_scheduler.log 2>&1
