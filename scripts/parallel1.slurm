#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH -p parallel
#SBATCH --output=job_output/out
#SBATCH --error=job_output/error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sak2km@virginia.edu
#SBATCH -A hcdm

module load anaconda2
python scripts/test1.py --data_sets test_new  --click_models inf nav per --log_folder outdir/logs/ --average_folder outdir/average --output_folder outdir/fullruns/ --n_runs 50 --n_proc 50 --n_impr 5000 --log_folder log_folder

