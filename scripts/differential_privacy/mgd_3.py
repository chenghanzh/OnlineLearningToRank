# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.datasimulation import DataSimulation
from utils.argparsers.simulationargparser import SimulationArgumentParser
from algorithms.PDGD.pdgd import PDGD
from algorithms.PDGD.deeppdgd import DeepPDGD
from algorithms.DBGD.tddbgd import TD_DBGD
from algorithms.DBGD.pdbgd import P_DBGD
from algorithms.DBGD.tdmgd import TD_MGD
from algorithms.DBGD.pmgd import P_MGD
from algorithms.DBGD.pmgd_dp import P_MGD_dp
from algorithms.baselines.pairwise import Pairwise
from algorithms.DBGD.neural.pdbgd import Neural_P_DBGD
# python scripts/CIKM2018.py --data_sets web2018 --click_models inf nav per --log_folder log_folder --average_folder outdir/average --output_folder outdir/fullruns/ --n_runs 50 --n_proc 25 --n_impr 5000

description = 'Run script for testing framework.'
parser = SimulationArgumentParser(description=description)

rankers = []

ranker_params = {
  'learning_rate_decay': 0.9999977,
  'noise_method': 3,
  'eta': 0.1,
  }
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'differential_privacy/MGD_3/eta01' 
# rankers.append((run_name, P_MGD_dp, other_args))

ranker_params = {
  'learning_rate_decay': 0.9999977,
  'noise_method': 3,
  'eta': 5,
  }
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'differential_privacy/MGD_3/eta5' 
rankers.append((run_name, P_MGD_dp, other_args))

ranker_params = {
  'learning_rate_decay': 0.9999977,
  'noise_method': 3,
  'eta': 10,
  }
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'differential_privacy/MGD_3/eta10' 
rankers.append((run_name, P_MGD_dp, other_args))

ranker_params = {
  'learning_rate_decay': 0.9999977,
  'noise_method': 3,
  'eta': 1.0,
  }
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'differential_privacy/MGD_3/eta1' 
rankers.append((run_name, P_MGD_dp, other_args))



sim = DataSimulation(sim_args)
sim.run(rankers)