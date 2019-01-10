#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH -p largemem
#SBATCH --output=job_output/out
#SBATCH --error=job_output/error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sak2km@virginia.edu
#SBATCH -A hcdm

module load anaconda2
python scripts/detail_exp/test_hist_viewed.py --data_sets MQ2007 MQ2008 --click_models inf per --log_folder log_folder --average_folder outdir/average --output_folder outdir/fullruns/ --n_runs 15 --n_proc 20 --n_impr 5000