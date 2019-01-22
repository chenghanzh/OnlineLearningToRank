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
from algorithms.baselines.pairwise import Pairwise
from algorithms.DBGD.neural.pdbgd import Neural_P_DBGD
from algorithms.DBGD.tdNSGD import TD_NSGD

from algorithms.PDGD.pdgd_wrapper import PDGD_Wrapper
from algorithms.DBGD.tddbgd_wrapper import TD_DBGD_Wrapper
from algorithms.DBGD.pdbgd_wrapper import P_DBGD_Wrapper
from algorithms.DBGD.tdmgd_wrapper import TD_MGD_Wrapper
from algorithms.DBGD.pmgd_wrapper import P_MGD_Wrapper
from algorithms.DBGD.tdNSGD_wrapper import TD_NSGD_Wrapper


# python scripts/CIKM2018.py --data_sets web2018 --click_models inf nav per --log_folder log_folder --average_folder outdir/average --output_folder outdir/fullruns/ --n_runs 50 --n_proc 25 --n_impr 5000

description = 'Run script for testing framework.'
parser = SimulationArgumentParser(description=description)

rankers = []



#######    prev_qeury_len = 5     #######
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  '_lambda': None,
  'prev_qeury_len': 5,
  'viewed': True}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'wrappers/test_hist_viewed/5_P_MGD_Wrapper' 
rankers.append((run_name, P_MGD_Wrapper, other_args))

#######    prev_qeury_len = 10     #######
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  '_lambda': None,
  'prev_qeury_len': 10,
  'viewed': True}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'wrappers/test_hist_viewed/10_P_MGD_Wrapper' 
rankers.append((run_name, P_MGD_Wrapper, other_args))

#######    prev_qeury_len = 20     #######
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  '_lambda': None,
  'prev_qeury_len': 20,
  'viewed': True}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'wrappers/test_hist_viewed/20_P_MGD_Wrapper' 
rankers.append((run_name, P_MGD_Wrapper, other_args))

#######    prev_qeury_len = 30     #######
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  '_lambda': None,
  'prev_qeury_len': 30,
  'viewed': True}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'wrappers/test_hist_viewed/30_P_MGD_Wrapper' 
rankers.append((run_name, P_MGD_Wrapper, other_args))


sim = DataSimulation(sim_args)
sim.run(rankers)