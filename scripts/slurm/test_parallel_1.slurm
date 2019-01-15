#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=6000
#SBATCH --time=72:00:00
#SBATCH -p parallel
#SBATCH --output=job_output_parallel/out
#SBATCH --error=job_output_parallel/error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sak2km@virginia.edu
#SBATCH -A hcdm
module load anaconda/5.2.0-py2.7
python scripts/detail_exp/test_intp2.py --data_sets MQ2007 --click_models inf per --log_folder log_folder --average_folder outdir/average --output_folder outdir/fullruns/ --n_runs 50 --n_proc 20 --n_impr 5000
