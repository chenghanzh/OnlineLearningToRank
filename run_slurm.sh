#!/bin/bash

sbatch scripts/slurm/baselines/dbgd/hp.slurm
sbatch scripts/slurm/baselines/dbgd/mq07.slurm
sbatch scripts/slurm/baselines/dbgd/mq08.slurm
sbatch scripts/slurm/baselines/dbgd/np.slurm
sbatch scripts/slurm/baselines/dbgd/td.slurm
sbatch scripts/slurm/baselines/dbgd/web10k.slurm
sbatch scripts/slurm/baselines/dbgd/webscope1.slurm

sbatch scripts/slurm/baselines/mgd/hp.slurm
sbatch scripts/slurm/baselines/mgd/mq07.slurm
sbatch scripts/slurm/baselines/mgd/mq08.slurm
sbatch scripts/slurm/baselines/mgd/np.slurm
sbatch scripts/slurm/baselines/mgd/td.slurm
sbatch scripts/slurm/baselines/mgd/web10k.slurm
sbatch scripts/slurm/baselines/mgd/webscope1.slurm


sbatch scripts/slurm/wrappers/dbgd/hp.slurm
sbatch scripts/slurm/wrappers/dbgd/mq07.slurm
sbatch scripts/slurm/wrappers/dbgd/mq08.slurm
sbatch scripts/slurm/wrappers/dbgd/np.slurm
sbatch scripts/slurm/wrappers/dbgd/td.slurm
sbatch scripts/slurm/wrappers/dbgd/web10k.slurm
sbatch scripts/slurm/wrappers/dbgd/webscope1.slurm

sbatch scripts/slurm/wrappers/mgd/hp.slurm
sbatch scripts/slurm/wrappers/mgd/mq07.slurm
sbatch scripts/slurm/wrappers/mgd/mq08.slurm
sbatch scripts/slurm/wrappers/mgd/np.slurm
sbatch scripts/slurm/wrappers/mgd/td.slurm
sbatch scripts/slurm/wrappers/mgd/web10k.slurm
sbatch scripts/slurm/wrappers/mgd/webscope1.slurm