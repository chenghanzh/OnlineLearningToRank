#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=120:00:00
#SBATCH -p standard
#SBATCH --output=job_output/out
#SBATCH --error=job_output/error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sak2km@virginia.edu
#SBATCH -A hcdm

module load anaconda2
python scripts/wrappers/mgd_p.py --data_sets MQ2007 --click_models inf nav per --log_folder log_folder --average_folder outdir/average --output_folder outdir/fullruns/ --n_runs 50 --n_proc 25 --n_impr 5000