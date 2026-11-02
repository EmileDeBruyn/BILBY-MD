#!/bin/bash -x
#SBATCH --job-name=f_equi
#SBATCH --account=profound
#SBATCH --nodes=1        
#SBATCH --ntasks=48     
#SBATCH --ntasks-per-node=48 
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=02:00:00       
#SBATCH --partition=booster

sysname=''

# Modules and programs
module purge
module load Stages/2025
module load GCC/13.3.0  ParaStationMPI/5.11.0-1
module load GROMACS/2024.3

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Working directory
#cd /p/project/idp/PTH/fibril_model/P4_fibril_PiR_BtB_UpUp

# minimize

cd minimize

gmx grompp -f em.mdp -c "$sysname"_ions.gro -p ../topol.top -o em_$sysname.tpr
gmx mdrun -deffnm em_$sysname > em_$sysname.out

# NVT equilibration

cp em_$sysname.gro ../equilibrate
cd ../equilibrate

#cd equilibrate

gmx grompp -f Eq_NVT.mdp -c em_"$sysname".gro -r em_"$sysname".gro -p ../topol.top -o eq_NVT_$sysname.tpr

gmx mdrun -gpu_id 0123 -pin on -nb gpu -bonded gpu -ntmpi 8 -ntomp 6 -deffnm eq_NVT_$sysname > eq_NVT_$sysname.out


#cd equilibrate

# NPT equilibration
gmx grompp -f Eq_NPT.mdp -c eq_NVT_"$sysname".gro -r eq_NVT_"$sysname".gro -t eq_NVT_$sysname.cpt -p ../topol.top -o eq_NPT_$sysname.tpr

gmx mdrun -gpu_id 0123 -pin on -nb gpu -bonded gpu -ntmpi 8 -ntomp 6 -deffnm eq_NPT_$sysname > eq_NPT_$sysname.out

wait

# run md

cp eq_NPT_$sysname.gro ../production
cp eq_NPT_$sysname.cpt ../production

cd ../production

sbatch jupiter_run.sh



























