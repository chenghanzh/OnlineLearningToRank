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
from algorithms.DBGD.tdNSGD_wrapper import TD_NSGD_Wrapper
from algorithms.DBGD.tdNSGD import TD_NSGD

import pdb

description = 'Run script for testing framework.'
parser = SimulationArgumentParser(description=description)

rankers = []

#######    interpolation = 0.7,  prev_qeury_len = 20     #######
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  '_lambda': None,
  'lambda_intp': 0.7,
  'prev_qeury_len': 20,
  'viewed': True}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'test_mgd/P_MGD_Wrapper_7_20' 
rankers.append((run_name, P_MGD_Wrapper, other_args))

#######    interpolation = 0.7,  prev_qeury_len = 10     #######
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  '_lambda': None,
  'lambda_intp': 0.7,
  'prev_qeury_len': 10,
  'viewed': True}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'test_mgd/P_MGD_Wrapper_7_10' 
# rankers.append((run_name, P_MGD_Wrapper, other_args))

ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'test_mgd/P_MGD_Wrapper' 
# rankers.append((run_name, P_MGD_Wrapper, other_args))


ranker_params = {
  'learning_rate_decay': 0.9999977,
  'GRAD_SIZE':60,
  'EXP_SIZE':25,
  'TB_QUEUE_SIZE':10,
  'TB_WINDOW_SIZE':50}
# sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'baselines/TD_NSGD_tb' 
# rankers.append((run_name, TD_NSGD, other_args))


ranker_params = {
  'learning_rate_decay': 0.9999977,
  'GRAD_SIZE':60,
  'EXP_SIZE':25}
# sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'baselines/TD_NSGD' 
# rankers.append((run_name, TD_NSGD, other_args))


#######    Normalization and No Increase K     #######
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'GRAD_SIZE':60,
  'EXP_SIZE':25}
# sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'wrappers/norm_NOincK/TD_NSGD_Wrapper' 
# rankers.append((run_name, TD_NSGD_Wrapper, other_args))


#######    Normalization and No Increase K     #######
ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'GRAD_SIZE':60,
  'EXP_SIZE':25,
  'TB_QUEUE_SIZE':10,
  'TB_WINDOW_SIZE':50}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'wrappers/norm_NOincK/TD_NSGD_Wrapper_tb' 
# rankers.append((run_name, TD_NSGD_Wrapper, other_args))


ranker_params = {
  'learning_rate_decay': 0.9999977,
  'svd': True,
  'project_norm': True,
  'k_initial': 3,
  'k_increase': False,
  'GRAD_SIZE':60,
  'EXP_SIZE':25,
  'TB_QUEUE_SIZE':10,
  'TB_WINDOW_SIZE':50,
  'lambda_intp': 0.7,
  'prev_qeury_len': 10,
  'viewed': True}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'wrappers_long/int7_hist10/TD_NSGD_int7_hist10' 
# rankers.append((run_name, TD_NSGD_Wrapper, other_args))


sim = DataSimulation(sim_args)
sim.run(rankers)