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


ranker_params = {
  'learning_rate_decay': 0.9999977}
sim_args, other_args = parser.parse_all_args(ranker_params)
run_name = 'exp_gradient_sampl/P_DBGD' 
# rankers.append((run_name, P_DBGD, other_args))


# PDBGD_useNSGD_uniformSampl
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'use_regular_sample':True,
  'use_NDCG':True,
  'use_NS':False,
  'use_all_listed':False
  }
sim_args, other_args = parser.parse_all_args(ranker_params)
# run_name = 'exp_gradient_sampl/PDBGD_useNDCG' 
run_name = 'exp_gradient_sampl/PDBGD_uniformSampl_useNDCG' 
# rankers.append((run_name, P_DBGD_Wrapper_exp, other_args))


# PDBGD_useML_uniformSampl
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'use_regular_sample':True,
  'use_NDCG':False,
  'use_NS':False,
  'use_all_listed':False
  }
sim_args, other_args = parser.parse_all_args(ranker_params)
# run_name = 'exp_gradient_sampl/PDBGD_useNDCG' 
# run_name = 'exp_gradient_sampl/PDBGD_abs(uniformSampl)_useML' 
run_name = 'exp_gradient_sampl/PDBGD_uniformSampl_useML' 
# rankers.append((run_name, P_DBGD_Wrapper_exp, other_args))



## PDBGD_docSampl_useNSGD
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'use_regular_sample':False,
  'use_NDCG':True,
  'use_NS':False,
  'use_all_listed':False
  }
sim_args, other_args = parser.parse_all_args(ranker_params)
# run_name = 'exp_gradient_sampl/PDBGD_useNDCG' 
run_name = 'exp_gradient_sampl/PDBGD_docSampl_useNDCG' 
rankers.append((run_name, P_DBGD_Wrapper_exp, other_args))

## PDBGD_useML_docSampl
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'use_regular_sample':False,
  'use_NDCG':False,
  'use_NS':False,
  'use_all_listed':False
  }
sim_args, other_args = parser.parse_all_args(ranker_params)
# run_name = 'exp_gradient_sampl/PDBGD_useNDCG' 
run_name = 'exp_gradient_sampl/PDBGD_docSampl_useML' 
rankers.append((run_name, P_DBGD_Wrapper_exp, other_args))

## PDBGD_docSampl_NS_useML
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'use_regular_sample':False,
  'use_NDCG':False,
  'use_NS':True,
  'use_all_listed':False
  }
sim_args, other_args = parser.parse_all_args(ranker_params)
# run_name = 'exp_gradient_sampl/PDBGD_useNDCG' 
run_name = 'exp_gradient_sampl/PDBGD_docSampl_NS_useML' 
# rankers.append((run_name, P_DBGD_Wrapper_exp, other_args))


sim = DataSimulation(sim_args)
sim.run(rankers)