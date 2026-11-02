#!/usr/bin/env python3
"""
Monitor MD simulations in ./simulations.

- Collects job states from Slurm (squeue).
- Computes progress from GROMACS log + mdp.
- Only includes systems that have a flag file: RUNNING / ERROR / DONE
  in their simulation folder (./simulations/<system>/).

Usage:
    python monitor_md.py
"""

import os
import re
import glob
import math
import subprocess
from collections import deque


def get_full_job_name(jobid):
    """Return full job name for a given Slurm job ID (unused helper)."""
    out = subprocess.check_output(["scontrol", "show", "job", str(jobid)], text=True)
    match = re.search(r"JobName=(.*)", out)
    return match.group(1) if match else None


def get_progress(log_path, mdp_path, tail=180):
    """
    Returns progress of simulation in percent (float 0–100),
    based on last reported MD step vs. nsteps in the mdp file.
    """

    # --- 1. Extract nsteps from mdp ---------------------------------------
    with open(mdp_path, "r") as f:
        target_step = None
        dt_ps = None
        for line in f:
            # match "nsteps = XXXXX"
            m = re.match(r"\s*nsteps\s*=\s*([0-9]+)", line)
            if m:
                target_step = int(m.group(1))
                continue
            m = re.match(r"\s*dt\s*=\s*([0-9Ee.+-]+)", line)
            if m:
                dt_ps = float(m.group(1))

    if target_step is None:
        raise ValueError(f"Could not find nsteps in {mdp_path}")
    if dt_ps is None:
        raise ValueError(f"Could not find dt in {mdp_path}")

    # --- 2. Read last lines of the log -----------------------------------
    with open(log_path, "r") as f:
        lines = list(deque(f, maxlen=tail))

    # --- 3. Find the last valid MD step ----------------------------------
    last_step = None
    step_pattern = re.compile(r"^\s*([0-9]+)\s+[0-9Ee.+-]+")

    for line in reversed(lines):
        m = step_pattern.match(line)
        if m:
            last_step = int(m.group(1))
            break

    if last_step is None:
        return 0.0, 0.0  # could not detect progress → assume 0%

    # --- 4. Compute percentage -------------------------------------------
    pct = (last_step / target_step) * 100.0

    # Bound to 0–100
    pct = max(0.0, min(100.0, pct))

    time_ps = last_step * dt_ps
    time_us = time_ps / 1_000_000.0

    return pct, time_us

def find_latest_log(prod_dir):
    """
    Return the path to the newest / highest-numbered md_system log file.
    Examples:
        md_system.log
        md_system.part0001.log
        md_system.part0010.log
    """
    pattern = os.path.join(prod_dir, "*.log")
    logs = glob.glob(pattern)

    if not logs:
        return None

    def extract_partnum(path):
        fname = os.path.basename(path)
        m = re.search(r"\.part(\d+)\.log$", fname)
        if m:
            return int(m.group(1))
        else:
            # plain md_system.log counts as part 0
            return 0

    # choose by highest part number
    latest = max(logs, key=extract_partnum)
    return latest

def main():
    ###################### get names + states of running jobs #############

    # %i = job id, %T = state (RUNNING, PENDING, etc), %j = job name
    try:
        squeue_lines = subprocess.check_output(
            ["squeue", "--me", "--noheader", "-o", "%i %T %j"], text=True
        ).splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error calling squeue: {e}")
        return

    # maps job name -> state (e.g. "system_0001" -> "RUNNING")
    job_state_by_name = {}
    for line in squeue_lines:
        # allow job names with spaces: split into 3 fields max
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        jid, state, name = parts
        job_state_by_name[name] = state

    ###################### scan simulation folders ########################

    base_folder = "./simulations"
    sys_folder_paths = [
        sf for sf in glob.glob(os.path.join(base_folder, "*")) if os.path.isdir(sf)
    ]

    # Only include systems that have a flag file RUNNING / ERROR / DONE
    systems = []
    systems_queue = [] # systems not started yet
    for sf in sys_folder_paths:
        if any(
            os.path.isfile(os.path.join(sf, flag))
            for flag in ("RUNNING", "ERROR", "DONE")
        ):
            systems.append(os.path.basename(sf))
        else:
            systems_queue.append(os.path.basename(sf))

    systems = sorted(systems)
    systems_queue = sorted(systems_queue)

    status = []
    sim_progress = []
    sim_time_us = []

    for s in systems:
        sf = os.path.join(base_folder, s)

        # ------- flag override --------
        flag_status = None
        if os.path.isfile(os.path.join(sf, "ERROR")):
            flag_status = "ERROR"
        elif os.path.isfile(os.path.join(sf, "RUNNING")):
            flag_status = "RUNNING"
        elif os.path.isfile(os.path.join(sf, "DONE")):
            flag_status = "DONE"

        # ERROR flag → override everything
        if flag_status == "ERROR":
            status.append("ERROR")
            prod_dir = os.path.join(sf, "production")
            log_path = find_latest_log(prod_dir)
            mdp_path = os.path.join(sf, "production/NPT_production.mdp")
            if log_path and os.path.exists(log_path) and os.path.exists(mdp_path):
                prog, time_us = get_progress(log_path, mdp_path)
            else:
                prog = math.nan  # signify "no production"
                time_us = math.nan
            sim_progress.append(prog)
            sim_time_us.append(time_us)
            continue
        
        # default
        st_label = "idle"
        prog = math.nan

        # ---- 1) Production jobs: job name == system name ----------------
        # if job in queue
        if s in job_state_by_name:
            slurm_state = job_state_by_name[s]
            if slurm_state.startswith("R"):      # RUNNING
                st_label = "running"
            elif slurm_state.startswith("PD"):   # PENDING
                st_label = "queued"
            else:
                st_label = slurm_state.lower()

            prod_dir = os.path.join(sf, "production")
            log_path = find_latest_log(prod_dir)
            mdp_path = os.path.join(sf, "production/NPT_production.mdp")
            if log_path and os.path.exists(log_path) and os.path.exists(mdp_path):
                prog, time_us = get_progress(log_path, mdp_path)
            else:
                prog = 0.0
                time_us = 0.0

        elif f"md_{s}" in job_state_by_name:
            slurm_state = job_state_by_name[f"md_{s}"]
            if slurm_state.startswith("R"):      # RUNNING
                st_label = "running"
            elif slurm_state.startswith("PD"):   # PENDING
                st_label = "queued"
            else:
                st_label = slurm_state.lower()

            prod_dir = os.path.join(sf, "production")
            log_path = find_latest_log(prod_dir)
            mdp_path = os.path.join(sf, "production/NPT_production.mdp")
            if log_path and os.path.exists(log_path) and os.path.exists(mdp_path):
                prog, time_us = get_progress(log_path, mdp_path)
            else:
                prog = 0.0
                time_us = 0.0

        # ---- 2) Equilibration jobs: job name "eq_<system>" --------------
        elif f"eq_{s}" in job_state_by_name:
            slurm_state = job_state_by_name[f"eq_{s}"]
            if slurm_state.startswith("R"):
                st_label = "equil_running"
            elif slurm_state.startswith("PD"):
                st_label = "equil_queued"
            else:
                st_label = "equil_" + slurm_state.lower()

            # Example: first NVT, then NPT; adapt as needed
            log_path_nvt = os.path.join(sf, "equilibrate/eq_NVT_system.log")
            mdp_path_nvt = os.path.join(sf, "equilibrate/Eq_NVT.mdp")
            log_path_npt = os.path.join(sf, "equilibrate/eq_NPT_system.log")
            mdp_path_npt = os.path.join(sf, "equilibrate/Eq_NPT.mdp")

            if os.path.exists(log_path_npt) and os.path.exists(mdp_path_npt):
                prog, time_us = get_progress(log_path_npt, mdp_path_npt)
            elif os.path.exists(log_path_nvt) and os.path.exists(mdp_path_nvt):
                prog, time_us = get_progress(log_path_nvt, mdp_path_nvt)
            else:
                prog = 0.0
                time_us = 0.0

        # ---- 3) Idle: no job in queue ----------------------------------
        else:
            if flag_status == "DONE":
                st_label = "DONE"
            else:
                st_label = "idle"

            # check if production run exists and report its progress if so
            prod_dir = os.path.join(sf, "production")
            log_path = find_latest_log(prod_dir)
            #print(prod_dir,log_path)
            mdp_path = os.path.join(sf, "production/NPT_production.mdp")
            if log_path and os.path.exists(log_path) and os.path.exists(mdp_path):
                prog, time_us = get_progress(log_path, mdp_path)
            else:
                prog = math.nan  # signify "no production"
                time_us = math.nan

        status.append(st_label)
        sim_progress.append(prog)
        sim_time_us.append(time_us)

    ###################### printing #######################################

    if not systems:
        print("No systems with RUNNING / ERROR / DONE flag found in ./simulations.")
        return

    # Header
    print("-" * 53)
    print(f"{'idx':>3} {'system':<15} {'status':<13} {'progress [%]':>10} {'time [ns]':>10}")
    print("-" * 53)

    # systems simulating
    for idx, (name, st, prog, time_us) in enumerate(zip(systems, status, sim_progress, sim_time_us)):
        # progress formatting: allow NaN / missing
        if isinstance(prog, (int, float)):
            if prog == prog:  # not NaN
                prog_str = f"{prog:6.2f}%"
            else:
                prog_str = "   n/a "
        else:
            prog_str = f"{str(prog):>8}"

        if isinstance(time_us, (int, float)):
            if time_us == time_us:  # not NaN
                time_str = f"{time_us*1e3:10.2f}"
            else:
                time_str = "     n/a "
        else:
            time_str = f"{str(time_us*1e3):>10}"

        print(f"{idx:3d} {name:<15} {st:<13} {prog_str:>10} {time_str:>10}")

    print("-" * 53)

    # systems in queue
    for idx2, name in enumerate(systems_queue):
        st = "not scheduled"
        prog_str = "   n/a "
        time_str = "     n/a "
        print(f"{idx+idx2+1:3d} {name:<15} {st:<13} {prog_str:>10} {time_str:>10}")
    
    print("-" * 53)

if __name__ == "__main__":
    main()
