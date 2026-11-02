#!/bin/bash -x
#SBATCH --job-name=O08816
#SBATCH --account=profound
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=06:00:00       
#SBATCH --partition=booster

sysname='system'

# Modules and programs
module purge
module load Stages/2025
module load GCC/13.3.0  ParaStationMPI/5.11.0-1
module load GROMACS/2024.3


#############################################################
#get topology without water
if [[ ! -f "md_${sysname}_noW.tpr" ]]; then
    echo 14 | gmx convert-tpr -s md_${sysname}.tpr -o md_${sysname}_noW.tpr
fi

### remove water from trajectory
if [[ ! -f "md_${sysname}_noW.trr" ]]; then
    echo 14 | gmx trjconv \
            -f md_${sysname}.trr \
            -s md_${sysname}.tpr \
            -pbc mol -ur compact \
            -dump 0 \
            -o md_${sysname}_noW.gro &>> md_$sysname.out

    echo 14 | gmx trjconv \
            -f md_${sysname}.trr \
            -s md_${sysname}.tpr \
            -pbc mol -ur compact \
            -force yes \
            -o md_${sysname}_noW.trr &>> md_$sysname.out
fi

# delete full traj if conv was successfull
if [ -f md_${sysname}_noW.trr ]; then
    rm md_${sysname}.trr
else
    echo "[check] ERROR: Failed to generate stripped trajectories." &>> md_$sysname.out
    touch ../ERROR
    exit 1
fi

