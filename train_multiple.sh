#!/bin/bash

#SBATCH --job-name=vi_ensemble
#SBATCH --constraint=A100
#SBATCH --time=0-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2/dmohan/bbb/RadioGalaxies-BBB/logs/nonlinear_shuffle/out-slurm_%j.out
#SBATCH --array=1-10
#SBATCH --cpus-per-task=17
#SBATCH --exclude=compute-0-116,compute-0-117,compute-0-118,compute-0-119,compute-0-7,compute-0-4

pwd;

nvidia-smi
echo ">>>start task array $SLURM_ARRAY_TASK_ID"
echo "Running on:"
hostname
source /share/nas2/dmohan/bbb/RadioGalaxies-BBB/venv/bin/activate 
echo ">>>training"
python /share/nas2/dmohan/bbb/RadioGalaxies-BBB/bbb_ensemble.py $SLURM_ARRAY_TASK_ID
