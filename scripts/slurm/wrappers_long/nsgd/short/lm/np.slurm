#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH -p largemem
#SBATCH --output=job_output/out
#SBATCH --error=job_output/error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sak2km@virginia.edu
#SBATCH -A hcdm

module load anaconda/5.2.0-py2.7
python scripts/wrappers_long/nsgd_tb_lm.py --data_sets NP2003 --click_models inf nav per --log_folder ~/../../../../scratch/sak2km/l2r_data/log_folder --average_folder ~/../../../../scratch/sak2km/l2r_data/outdir/average --output_folder ~/../../../../scratch/sak2km/l2r_data/outdir/fullruns/ --n_runs 20 --n_proc 8 --n_impr 10000