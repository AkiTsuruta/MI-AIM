#!/bin/bash 
#SBATCH --job-name=colinv_test 
#SBATCH --account=project_462000039
#SBATCH --time=00:20:00
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=18
#SBATCH --mem=2G 
#SBATCH --partition=small 

module load LUMI/22.06 partition/L
module load cray-python/3.9.12.1

./col_inv_multiprocessing.py
