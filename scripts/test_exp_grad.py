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
from algorithms.DBGD.p_dbgd_wrapper_exp import P_DBGD_Wrapper_exp
import pdb

description = 'Run script for testing framework.'
parser = SimulationArgumentParser(description=description)

rankers = []

# ranker_params = {
#   'learning_rate_decay': 0.9999977}
# sim_args, other_args = parser.parse_all_args(ranker_params)

###		DBGD	###
# ranker_params = {
#   'learning_rate_decay': 0.9999977,}
# sim_args, other_args = parser.parse_all_args(ranker_params)

ranker_params = {
  'learning_rate_decay': 0.9999977}
sim_args, other_args = parser.parse_all_args(ranker_params)
run_name = 'exp_gradient_sampl/P_DBGD' 
rankers.append((run_name, P_DBGD, other_args))



## TT
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'use_NS':True,          # if not, only use DS
  'use_all_listed':True   # if not, use only clicked
  }
sim_args, other_args = parser.parse_all_args(ranker_params)
run_name = 'exp_gradient_sampl/P_DBGD_exp_TT' 
# rankers.append((run_name, P_DBGD_Wrapper_exp, other_args))


## TF
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'use_NS':True,
  'use_all_listed':False
  }
sim_args, other_args = parser.parse_all_args(ranker_params)
run_name = 'exp_gradient_sampl/P_DBGD_exp_TF' 
# rankers.append((run_name, P_DBGD_Wrapper_exp, other_args))


## FT
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'use_NS':False,
  'use_all_listed':True
  }
sim_args, other_args = parser.parse_all_args(ranker_params)
run_name = 'exp_gradient_sampl/P_DBGD_exp_FT' 
# rankers.append((run_name, P_DBGD_Wrapper_exp, other_args))


## FF
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'use_NS':False,
  'use_all_listed':False
  }
sim_args, other_args = parser.parse_all_args(ranker_params)
run_name = 'exp_gradient_sampl/P_DBGD_exp_FF' 
rankers.append((run_name, P_DBGD_Wrapper_exp, other_args))

sim = DataSimulation(sim_args)
sim.run(rankers)