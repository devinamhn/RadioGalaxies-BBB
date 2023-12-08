#!/bin/bash

#SBATCH --job-name=vi
#SBATCH --constraint=A100
#SBATCH --time=0-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/logs/out-slurm_%j.out
#SBATCH --exclude=compute-0-116,compute-0-117,compute-0-118,compute-0-119,compute-0-7

pwd;

nvidia-smi
echo ">>>start"
source /share/nas2/dmohan/bbb/RadioGalaxies-BBB/venv/bin/activate 
echo ">>>training"
python /share/nas2/dmohan/bbb/RadioGalaxies-BBB/mirabest_bbb.py
