#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=48:00:00
#SBATCH -p standard
#SBATCH --output=job_output/out
#SBATCH --error=job_output/error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sak2km@virginia.edu
#SBATCH -A hcdm
module load anaconda2
python scripts/wrapper_k3_std.py --data_sets web2018 --click_models inf nav per --log_folder outdir/logs/ --average_folder outdir/average --output_folder outdir/fullruns/ --n_runs 10 --n_proc 15 --n_impr 10000 --log_folder log_folder

