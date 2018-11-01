# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.datasimulation import DataSimulation
from utils.argparsers.simulationargparser import SimulationArgumentParser
from algorithms.PDGD.pdgd import PDGD
from algorithms.PDGD.deeppdgd import DeepPDGD
from algorithms.PDGD.pdgd_wrapper import PDGD_Wrapper
from algorithms.DBGD.tddbgd import TD_DBGD
from algorithms.DBGD.tddbgd_wrapper import TD_DBGD_Wrapper
from algorithms.DBGD.pdbgd import P_DBGD
from algorithms.DBGD.pdbgd_wrapper import P_DBGD_Wrapper
from algorithms.DBGD.tdmgd import TD_MGD
from algorithms.DBGD.tdmgd_wrapper import TD_MGD_Wrapper
from algorithms.DBGD.pmgd import P_MGD
from algorithms.DBGD.pmgd_wrapper import P_MGD_Wrapper
from algorithms.baselines.pairwise import Pairwise
from algorithms.DBGD.neural.pdbgd import Neural_P_DBGD
import pdb
# python scripts/svd_norm.py --data_sets MQ2007 Webscope_C14_Set1 MSLR-WEB10K --click_models inf --log_folder log_folder ---average_folder outdir/average --output_folder outdir/fullruns/ --n_runs 50 --n_proc 20 --n_impr 5000

description = 'Run script for testing framework.'
parser = SimulationArgumentParser(description=description)

rankers = []
#######    SVD, with Normalization     #######
# DBGD MGD
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'svd_norm/svd_norm/P_DBGD_Wrapper' 
rankers.append((run_name, TD_DBGD_Wrapper, other_args))

run_name = 'svd_norm/svd_norm/P_MGD_Wrapper' 
rankers.append((run_name, P_MGD_Wrapper, other_args))

# PDGD
ranker_params = {
  'learning_rate': 0.1,
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False
}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'svd_norm/svd_norm/PDGD_Wrapper' 
rankers.append((run_name, PDGD_Wrapper, other_args))

#######    QR, with Normalization     #######
# DBGD MGD
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': False,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'svd_norm/qr_norm/P_DBGD_Wrapper' 
rankers.append((run_name, TD_DBGD_Wrapper, other_args))

run_name = 'svd_norm/qr_norm/P_MGD_Wrapper' 
rankers.append((run_name, P_MGD_Wrapper, other_args))

# PDGD
ranker_params = {
  'learning_rate': 0.1,
  'learning_rate_decay': 0.9999977,
  'svd': False,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False
}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'svd_norm/qr_norm/PDGD_Wrapper' 
rankers.append((run_name, PDGD_Wrapper, other_args))


#######    QR, with NO Normalization     #######
# DBGD MGD
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': False,
  'project_norm': False,
  'k_initial': 3,
  'k_increase': False}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'svd_norm/qr_no_norm/P_DBGD_Wrapper' 
rankers.append((run_name, TD_DBGD_Wrapper, other_args))

run_name = 'svd_norm/qr_no_norm/P_MGD_Wrapper' 
rankers.append((run_name, P_MGD_Wrapper, other_args))

# PDGD
ranker_params = {
  'learning_rate': 0.1,
  'learning_rate_decay': 0.9999977,
  'svd': False,
  'project_norm': False,
  'k_initial': 3,
  'k_increase': False
}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'svd_norm/qr_no_norm/PDGD_Wrapper' 
rankers.append((run_name, PDGD_Wrapper, other_args))




sim = DataSimulation(sim_args)
sim.run(rankers)