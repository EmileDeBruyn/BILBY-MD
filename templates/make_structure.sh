#!/bin/bash -x
#SBATCH --job-name=make_struct
#SBATCH --account=profound
#SBATCH --nodes=1       
#SBATCH --ntasks=48   
#SBATCH --ntasks-per-node=48
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=00:30:00       
#SBATCH --partition=booster

module purge
module load Stages/2025
module load GCC/13.3.0  ParaStationMPI/5.11.0-1
module load GROMACS/2024.3

###################################
# folder must contain sysname.pdb #
###################################

sysname='system'

templates='../../templates'

### sheduler
touch ./RUNNING

####### folder structure


cp $templates/script_min_eqNVT.sh .

mkdir minimize
cp $templates/em.mdp minimize

mkdir equilibrate
cp $templates/Eq* equilibrate

mkdir production
cp $templates/NPT* production
cp $templates/jupiter_run.sh production
cp $templates/check_and_resubmit.sh production
cp $templates/remove_water.sh production

mkdir structure
cp $templates/charmm36-feb2021_caps.ff structure -r
cp $templates/generic.mdp structure
mkdir structure/em_vac
cp $templates/em.mdp ./structure/em_vac 

mv *pdb structure



###################################
#            Monomer              #
###################################
cd ./structure

# copy ff and stuff
# amber 2010 ff
gmx pdb2gmx -f $sysname.pdb -o $sysname.gro -ignh <<EOF
6
1
EOF

wait 

gmx editconf -bt dodecahedron -d 3 -f $sysname.gro -o "$sysname"_box.gro > setup.out

# em
cd em_vac

gmx grompp -f em.mdp -p ../topol.top -maxwarn 2 -c ../"$sysname"_box.gro -o em_vac_"$sysname".tpr
gmx mdrun -v -deffnm em_vac_"$sysname" >> setup.out 

cd ..

#solv
gmx solvate -cs spc216.gro -p topol.top -cp ./em_vac/em_vac_"$sysname".gro -o "$sysname"_solv.gro >> setup.out

# ionize
gmx grompp -f generic.mdp -p topol.top -o ions.tpr -maxwarn 1 -c "$sysname"_solv.gro >> setup.out
echo 13 | gmx genion -s ions.tpr -p topol.top -pname NA -nname CL -neutral -conc 0.15 -o "$sysname"_ions.gro >> setup.out

rm *#

mv topol.top ..
mv *.itp ..
mv charmm36-feb2021_caps.ff ..
cp "$sysname"_ions.gro ../minimize

cd ..

# substitute the systemname
folder="${PWD##*/}"

sed -i "/sysname=/c\sysname='$sysname'" ./script_min_eqNVT.sh
sed -i "/#SBATCH --job-name=/c\#SBATCH --job-name=eq_$folder" ./script_min_eqNVT.sh
sed -i "/sysname=/c\sysname='$sysname'" ./production/jupiter_run.sh
sed -i "/#SBATCH --job-name=/c\#SBATCH --job-name=md_$folder" ./production/jupiter_run.sh
sed -i "/sysname=/c\sysname='$sysname'" ./production/remove_water.sh
sed -i "/#SBATCH --job-name=/c\#SBATCH --job-name=noW_$folder" ./production/remove_water.sh
sed -i "/sysname=/c\sysname='$sysname'" ./production/check_and_resubmit.sh

sbatch script_min_eqNVT.sh
