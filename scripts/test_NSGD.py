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

from algorithms.DBGD.tdNSGD import TD_NSGD

import pdb

description = 'Run script for testing framework.'
parser = SimulationArgumentParser(description=description)

rankers = []

# ranker_params = {
#   'learning_rate_decay': 0.9999977}
# sim_args, other_args = parser.parse_all_args(ranker_params)

###		DBGD	###
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': False,
  'k_initial': 3,
  'k_increase': False}
sim_args, other_args = parser.parse_all_args(ranker_params)

# run_name = 'WEB2018/TD_DBGD' 
# rankers.append((run_name, TD_DBGD, other_args))

# run_name = 'WEB2018/TD_DBGD_Wrapper' 
# rankers.append((run_name, TD_DBGD_Wrapper, other_args))

# run_name = 'WEB2018/P-DBGD' 
# rankers.append((run_name, P_DBGD, other_args))

# run_name = 'WEB2018/P_DBGD_Wrapper' 
# rankers.append((run_name, TD_DBGD_Wrapper, other_args))

# run_name = 'CIKM2018/DeepP-DBGD' 
# rankers.append((run_name, Neural_P_DBGD, other_args))

# run_name = 'speedtest/TD-MGD' 
# rankers.append((run_name, TD_MGD, other_args))
# run_name = 'WEB2018/TD_MGD4' 
# rankers.append((run_name, TD_MGD, other_args))

# run_name = 'WEB2018/TD_MGD4_Wrapper' 
# rankers.append((run_name, TD_MGD_Wrapper, other_args))

###		MGD9	###
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'n_candidates': 9,
  # 'svd': True,
  # 'project_norm': False,
  # 'k_initial': 3,
  # 'k_increase': False,
  'GRAD_SIZE':60,
  'EXP_SIZE':25,
  }
sim_args, other_args = parser.parse_all_args(ranker_params)
# run_name = 'WEB2018/TD_MGD9' 
# rankers.append((run_name, TD_MGD, other_args))

# run_name = 'WEB2018/TD_MGD9_Wrapper' 
# rankers.append((run_name, TD_MGD_Wrapper, other_args))

# run_name = 'CIKM2018/P-MGD' 
# rankers.append((run_name, P_MGD, other_args))

# run_name = 'CIKM2018/P_MGD_Wrapper' 
# rankers.append((run_name, P_MGD_Wrapper, other_args))

# ranker_params = {
#   'learning_rate_decay': 0.9999977,
#   'epsilon': 0.8}
# sim_args, other_args = parser.parse_all_args(ranker_params)

# run_name = 'CIKM2018/Pairwise' 
# rankers.append((run_name, Pairwise, other_args))

run_name = 'WEB2018/TD_NSGD' 
rankers.append((run_name, TD_NSGD, other_args))

ranker_params = {
  'learning_rate_decay': 0.9999977,
  'n_candidates': 9,
  # 'svd': True,
  # 'project_norm': False,
  # 'k_initial': 3,
  # 'k_increase': False,
  }
sim_args, other_args = parser.parse_all_args(ranker_params)

# run_name = 'WEB2018/TD_MGD9' 
# rankers.append((run_name, TD_MGD, other_args))

###		PDGD	###
ranker_params = {
  'learning_rate': 0.1,
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': False,
  'k_initial': 3,
  'k_increase': False
}
sim_args, other_args = parser.parse_all_args(ranker_params)

# run_name = 'WEB2018/PDGD' 
# rankers.append((run_name, PDGD, other_args))

# run_name = 'WEB2018/PDGD_Wrapper' 
# rankers.append((run_name, PDGD_Wrapper, other_args))

# run_name = 'CIKM2018/DeepPDGD'
# rankers.append((run_name, DeepPDGD, other_args))

sim = DataSimulation(sim_args)
sim.run(rankers)