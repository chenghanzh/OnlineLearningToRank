# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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

description = 'Run script for testing framework.'
parser = SimulationArgumentParser(description=description)

rankers = []

ranker_params = {
  'learning_rate_decay': 0.9999977}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'CIKM2018_100/TD-DBGD' 
rankers.append((run_name, TD_DBGD, other_args))

run_name = 'CIKM2018_100/P-DBGD' 
rankers.append((run_name, P_DBGD, other_args))

run_name = 'CIKM2018_100/TD-MGD' 
rankers.append((run_name, TD_MGD, other_args))

run_name = 'CIKM2018_100/P-MGD' 
rankers.append((run_name, P_MGD, other_args))


ranker_params = {
  'learning_rate': 0.1,
  'learning_rate_decay': 0.9999977,
}
sim_args, other_args = parser.parse_all_args(ranker_params)

run_name = 'CIKM2018_100/PDGD' 
rankers.append((run_name, PDGD, other_args))

sim = DataSimulation(sim_args)
sim.run(rankers)