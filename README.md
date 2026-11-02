# BILBY-MD

`BILBY-MD` is the molecular-dynamics validation layer for MotifBench
scaffold-design runs. Its main job is to take MotifBench evaluation output,
select one scaffold per benchmark problem, prepare MD simulations for those
scaffolds, and analyze the resulting trajectories against a precomputed
reference pool in `analysis_ref/`.

The standard workflow is:

1. run a scaffold-generation model and evaluate the designs with MotifBench
2. use `run_pipeline.sh` to extract motif-bearing scaffold structures and build
   GROMACS-ready systems in `simulations/`
3. run MD for those systems
4. use `run_analysis.sh` to compute RMSD, motif contact maps, and FES-based
   comparison metrics relative to the reference set

![BILBY-md workflow](docs/images/bilby-pipeline.png)

The repository is organized around three pieces of state:

- `motif_pdbs/`: selected scaffold structures used to seed simulations
- `simulations/`: per-problem MD folders and trajectories
- `analysis_ref/`: the reusable reference baseline consumed during analysis

`motif_info.json` is the central metadata file for all motif-aware setup and
analysis. Each top-level key must match a simulation folder name under
`simulations/`. For each system it defines the motif residue range, the residue
indices used by RMSD/contact/FES analysis, and optionally the motif chain for
multi-chain systems. A typical entry looks like:

```
{
    "1AGY_PS00155_c2": {
        "contig": "A110-122",
        "motif_chain": "A",
        "motif_idx": [
            110,
            ...
            122
        ],
        "redesign_info": "A111;A113;A116"
    }
}
```

The reference pool stored in `analysis_ref/` contains the baseline artifacts
consumed by `run_analysis.sh`: scaffold RMSD (`rmsd_ref.pickle`), motif RMSD
(`rmsd_motif_ref.pickle`), motif contact maps
(`*_ref_motif_contactmap.npy`), and PCA/TICA reference models for FES
projection (`results_tica-pca_base.pkl`, plus `results_tica-pca.pkl` when a
combined reference set is available).

The scripts are written for SLURM-based execution with environment-provided
GROMACS modules on JSC systems. The provided job scripts currently target the
JUPITER software environment and the project-specific Python environment at
`/p/scratch/profound/schaeffler1/venv_BILBY-MD/activate.sh`.
The used packages are defined in the requirements.txt for reproducability.

To evaluate cases which are tagged unsolved by the MotifBench evaluation a local installation 
of PULCHRA for backbone reconstruction is needed which has to be specified in `run_pipeline.sh`.

Extending the reference pool with new benchmark cases is supported, but it is
an add-on workflow rather than the default use case. When needed, new cases can
be simulated locally and merged into the existing reference baseline with
`analysis_ref/run_reference_update.sh`.

## Requirements
Install the Python analysis dependencies with:

```bash
pip install -r requirements.txt
```

The analysis pipeline now depends on `MDAnalysis`, `numpy`, `pandas`,
`matplotlib`, `scikit-learn`, and `deeptime`.

Running the full simulation workflow also assumes access to the site modules
used by the Slurm templates (`GROMACS/2024.3` on the booster partition in the
current setup). Failed-case reconstruction additionally requires an executable
PULCHRA binary; set its path with `PULCHRA_BIN` in `run_pipeline.sh`.

## Simulation pipeline
Use `run_pipeline.sh` as the main entry point. It calls `getMotifSamples.py` to
collect simulation-ready PDBs, creates `simulations/<problem>/` folders from the
templates, and starts `md_scheduler.sh` to keep a bounded number of Slurm jobs
active.

Each MotifBench evaluation directory is expected to contain:

- `summary_by_problem.csv`
- one subfolder per problem
- each problem subfolder with `esm_complete_results.csv` and `motif_info.json`

`esm_complete_results.csv` must include `tm_score`, `Success`, `sample_path`,
and `backbone_path`. Failed-case reconstruction also needs a `sequence` column.

`run_pipeline.sh` is configured via the variables at the top of the script:
- `EVAL_DIRS`: JSON list of one or more MotifBench evaluation directories containing the per-problem results.
- `OUT_DIR`: output location for `motif_pdbs/` and `motif_info.json` (used to build `simulations/`).
- `LOG_FILE`: log file written by the setup pipeline.
- `MAX_PARALLEL`: maximum number of concurrent jobs for the MD scheduler.
- `INCLUDE_SUCCESSES`: set to `1` to include problems with `Num_Solutions > 0`, or `0` to skip solved problems.
- `INCLUDE_FAILURES`: set to `1` to include problems with `Num_Solutions <= 0`, or `0` to skip them.
- `FAILURES_MODE`: `high_tm` selects the failed backbone with the highest TM-score; `low_tm` selects the lowest.
- `PULCHRA_BIN`: path to the PULCHRA executable used to reconstruct failed backbones.
- `PROBLEM_CONFIG`: optional path to a JSON file listing exact problem folder names to include.
- `COMPUTE_PROJECT`: Slurm project/account used for the `#SBATCH --account=` flag in the templates.

If you only want a subset of problems, create or edit `problem_selection.json`:

```json
{
  "problems": [
    "01_1LDB",
    "03_2CGA",
    "05_5ZE9"
  ]
}
```

Then set `PROBLEM_CONFIG` in `run_pipeline.sh` to that file path. `getMotifSamples.py`
also accepts the same file directly via `--problem-config /path/to/problem_selection.json`,
and `EVAL_DIRS` should be provided as a JSON list string such as
`["/path/to/evaluation_XL", "/path/to/evaluation_other"]`.

### Sample selection
`getMotifSamples.py` selects one scaffold per problem and writes the selected
structures and metadata into `OUT_DIR`:

- `motif_pdbs/<problem>.pdb`: the PDB used to start the MD simulation.
- `motif_pdbs/motif_pdbs.csv`: audit table with the selected TM-score, solved
  flag, selection mode, source paths, sequence, and reconstruction method.
- `motif_info.json`: motif residue metadata for the selected backbone, keyed by
  problem name. The analysis scripts use this file to identify motif residues.

For solved problems (`Num_Solutions > 0`), only rows with `Success == 1` are
considered and the highest `tm_score` row is copied from `sample_path`.

For failed problems (`Num_Solutions <= 0`), set `INCLUDE_FAILURES=1`. The script
selects either the highest- or lowest-TM failed row according to
`FAILURES_MODE`, resolves its raw generated backbone from `backbone_path`, and
reconstructs an all-atom amino-acid representation with PULCHRA:

1. Copy the raw backbone into `failure_backbones/<problem>/`.
2. Rewrite residue names from the selected one-letter `sequence`.
3. Run PULCHRA on the substituted backbone.
4. Normalize the rebuilt PDB to a single `TER` record plus `END`.
5. Validate the rebuilt PDB with MDAnalysis and move it to
   `motif_pdbs/<problem>.pdb`.

The failed-case audit directory contains:

- `selected_row.json`: the exact CSV row and resolved paths used for selection.
- `<problem>.<selection_mode>.bb_original.pdb`: raw backbone before residue-name substitution.
- `<problem>.<selection_mode>.bb_substituted.pdb`: backbone passed to PULCHRA.
- `pulchra.log`: PULCHRA stdout/stderr.

Both solved samples and PULCHRA-rebuilt failed samples must pass the same basic
PDB checks before simulation: standard amino-acid residue names, side-chain atoms
present, a single chain, and exactly one `TER` record.

### MD setup and execution
After `motif_pdbs/` is written, `run_pipeline.sh`:

1. Updates the Slurm account in the template scripts from `COMPUTE_PROJECT`.
2. Creates `simulations/<problem>/` for each `motif_pdbs/<problem>.pdb`.
3. Copies `templates/make_structure.sh` into each simulation folder and patches
   `sysname` plus the setup job name.
4. Writes `MAX_PARALLEL` into `templates/check_and_resubmit.sh`.
5. Starts `md_scheduler.sh ./simulations $MAX_PARALLEL`.

For each system, the Slurm chain is:

1. `make_structure.sh`: runs `gmx pdb2gmx`, builds the box, solvates, ionizes,
   and submits equilibration.
2. `script_min_eqNVT.sh`: minimizes, runs NVT and NPT equilibration, then
   submits production.
3. `jupiter_run.sh`: builds or reuses the production TPR, runs or continues
   production MD from checkpoint, and calls `check_and_resubmit.sh`.
4. `check_and_resubmit.sh`: marks the system `DONE`, `ERROR`, or resubmits
   production until the target `nsteps` is reached. When complete, it submits
   `remove_water.sh`.
5. `remove_water.sh`: writes no-water production outputs and removes the full
   water trajectory after successful conversion.

Important logs and flags:

- `md_pipeline.log`: sample selection and setup log from `run_pipeline.sh`.
- `md_scheduler.log`: scheduler submissions and backfill events.
- `simulations/<problem>/RUNNING`, `DONE`, and `ERROR`: sample-level scheduler
  state.
- `simulations/<problem>/production/md_<problem>.out` and `.log`: production
  GROMACS output used by resubmission and monitoring.

## Evaluation pipeline
The standardized analysis is run with `run_analysis.sh`, which calls five scripts in
`analysis_scripts/` in sequence.

- `analysis_scripts/process_trajectory.py`: centers and aligns each trajectory to frame 0,
  writing `md_<system>_noW_aligned.xtc` in each
  `simulations/<system>/production` directory.
- `analysis_scripts/calc_rmsd.py`: computes backbone RMSD for the full scaffold and motif
  residues and writes one pickle per system to
  `analysis_sys/<system>_rmsd.pkl`.
- `analysis_scripts/calc_contacts.py`: computes the average motif contact map and writes
  `analysis_sys/<system>_motif_contactmap.npy`.
- `analysis_scripts/calc_fes.py`: computes motif backbone `phi`/`psi` dihedrals,
  projects them into the reference PCA/TICA spaces stored in
  `analysis_ref/results_tica-pca_base.pkl`, and writes
  `analysis_sys/results_tica-pca_sys.pkl`.
- `analysis_scripts/summarize_analysis.py`: compares simulation results to `analysis_ref/`,
  writes overview plots to `analysis_sys/summary_plots/`, and produces
  `analysis_sys/summary_by_problem.csv` with RMSD, contact-map, and FES similarity
  metrics.

`run_analysis.sh` currently runs five analysis steps in order:

1. `process_trajectory.py`
2. `calc_rmsd.py`
3. `calc_contacts.py`
4. `calc_fes.py`
5. `summarize_analysis.py`

### Usage
```bash
bash run_analysis.sh /p/scratch/profound/schaeffler1/BILBY-MD
```

`run_analysis.sh` accepts environment overrides:
- `PYTHON` (defaults to `python`)
- `ALIGN_SELECTION` and `ALIGN_WEIGHTS` for alignment
- `RMSD_SELECTION` for RMSD calculation
- `CONTACT_CUTOFF` and `CONTACT_INTERVAL` for contact maps
- `JACCARD_CONTACT_THRESHOLD` for contact-map comparison in the summary step

The script resolves `ROOT` from the first positional argument, otherwise from
`$SLURM_SUBMIT_DIR`, otherwise from the directory containing `run_analysis.sh`.

The analysis scripts expect `motif_info.json` at the project root (or pass
`--motif-info` directly to the Python scripts).

## Analysis outputs
`analysis_sys/` contains the per-system and summary outputs produced by the
analysis pipeline:

- `*_rmsd.pkl`: scaffold and motif RMSD traces per system.
- `*_motif_contactmap.npy`: average motif contact map per system.
- `results_tica-pca_sys.pkl`: per-system PCA/TICA projections in the reference
  latent spaces.
- `summary_by_problem.csv`: combined RMSD, contact, and FES similarity metrics.

`analysis_sys/summary_plots/` contains the aggregate plots, including:

- `scaffold_rmsd_overview.png`
- `motif_rmsd_overview.png`
- `contactmaps_ref.png`
- `contactmaps_simdiff.png`
- `fes_pca_sys.png`
- `fes_pca_diff_sys_minus_ref.png`
- `fes_tica_sys.png`
- `fes_tica_diff_sys_minus_ref.png`

The FES comparison uses 20-bin free-energy surfaces in both PCA and TICA space
and adds the following columns to `summary_by_problem.csv`:

- `pca_fes_absdiff`
- `pca_fes_pearson`
- `tica_fes_absdiff`
- `tica_fes_pearson`

## Reference Update Workflow
To turn the local enzyme simulations into new reference entries, use
`analysis_ref/run_reference_update.sh`. This workflow mirrors the structure of
`run_analysis.sh`, but writes intermediate outputs into
`analysis_ref/update_inputs/` and then merges the new systems into the existing
reference pool in `analysis_ref/`.

`analysis_ref/run_reference_update.sh` runs five steps in order:

1. `analysis_scripts/process_trajectory.py`
2. `analysis_scripts/calc_rmsd.py`
3. `analysis_scripts/calc_contacts.py`
4. `analysis_ref/calc_fes_scratch.py`
5. `analysis_ref/merge_reference_updates.py`

The helper scripts are:

- `analysis_ref/calc_fes_scratch.py`: fits PCA and TICA models from scratch for
  each local system using motif backbone `phi`/`psi` dihedrals and writes
  `analysis_ref/update_inputs/results_tica-pca_ref_scratch.pkl`.
- `analysis_ref/merge_reference_updates.py`: takes the newly generated
  `*_rmsd.pkl`, `*_motif_contactmap.npy`, and
  `results_tica-pca_ref_scratch.pkl` files from `analysis_ref/update_inputs/`,
  appends their RMSD and FES entries into the existing reference pickles, and
  copies the contact maps into `analysis_ref/` as
  `*_ref_motif_contactmap.npy`.

The reference update workflow uses the root-level `motif_info.json` for motif
definitions. For FES fitting, `calc_fes_scratch.py` uses both `motif_idx` and
`motif_chain` when present, so multi-chain systems such as
`4KRX_PS01173_c2` can be resolved to the intended chain.

### Usage
```bash
bash analysis_ref/run_reference_update.sh /p/scratch/profound/schaeffler1/BILBY-MD
```

`analysis_ref/run_reference_update.sh` accepts environment overrides:
- `PYTHON` (defaults to `python`)
- `ALIGN_SELECTION` and `ALIGN_WEIGHTS` for alignment
- `RMSD_SELECTION` for RMSD calculation
- `CONTACT_CUTOFF` and `CONTACT_INTERVAL` for contact maps
- `TICA_LAGTIME` for scratch TICA fitting
- `REF_BUILD_DIR` to change the intermediate output directory

The merge step updates these reference files when present:

- `analysis_ref/rmsd_ref.pickle`
- `analysis_ref/rmsd_motif_ref.pickle`
- `analysis_ref/rmsd_motif.pickle`
- `analysis_ref/results_tica-pca_base.pkl`
- `analysis_ref/results_tica-pca.pkl`

## Monitoring progress
Use `monitor_progress.sh` to get a quick overview of simulation state, progress,
and queue status. The script sources the project environment and runs
`analysis_scripts/monitor_md.py`, which:
- scans `./simulations/<system>/` for `RUNNING`, `ERROR`, or `DONE` flags and
  reports those systems in the main table
- lists unflagged systems as `not scheduled`
- queries Slurm via `squeue --me` to map job names to states (production jobs
  match `<system>`, equilibration jobs match `eq_<system>`)
- computes percent complete and simulated time from the newest GROMACS log
  (e.g., `production/*.log`) plus the matching `.mdp` (e.g.,
  `production/NPT_production.mdp`), falling back to `n/a` when files are missing

Usage (run from the project root so `./simulations` resolves correctly):
```bash
bash monitor_progress.sh
```

## License

BILBY-MD is licensed under the Apache License,
Version 2.0. See [LICENSE](LICENSE) for details.

Third-party software and datasets used by the pipeline
remain subject to their respective licenses.

# Open Todo's
- calc foldseek cluster and check if within same cluster as original structure