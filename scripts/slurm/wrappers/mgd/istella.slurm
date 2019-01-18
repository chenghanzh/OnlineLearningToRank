#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mem=100000
#SBATCH -p largemem
#SBATCH --output=job_output/out
#SBATCH --error=job_output/error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sak2km@virginia.edu
#SBATCH -A hcdm

module load anaconda/5.2.0-py2.7
python scripts/norm_incK.py --data_sets istella --click_models inf --log_folder log_folder --average_folder outdir/average --output_folder outdir/fullruns/ --n_runs 50 --n_proc 25 --n_impr 5000