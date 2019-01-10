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
python scripts/detail_exp/test_len_docspace_pos.py --data_sets Webscope_C14_Set1 --click_models inf --log_folder log_folder --average_folder outdir/average --output_folder outdir/fullruns/ --n_runs 10 --n_proc 10 --n_impr 5000