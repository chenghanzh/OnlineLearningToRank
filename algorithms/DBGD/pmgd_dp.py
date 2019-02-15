# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from algorithms.DBGD.pdbgd import P_DBGD
from models.linearmodel import LinearModel

# PMGD with differential privacy

# Probabilistic Interleaving Dueling Bandit Gradient Descent
class P_MGD_dp(P_DBGD):

  def __init__(self, n_candidates, noise_method, eta,  *args, **kargs):
    super(P_MGD_dp, self).__init__(*args, **kargs)
    self.n_candidates = n_candidates
    self.model = LinearModel(n_features = self.n_features,
                             learning_rate = self.learning_rate,
                             n_candidates = self.n_candidates)
    self.noise_method = noise_method
    self.eta = eta


  @staticmethod
  def default_parameters():
    parent_parameters = P_DBGD.default_parameters()
    parent_parameters.update({
      'n_candidates': 49,
      })
    return parent_parameters

  def update_to_interaction(self, clicks, stop_index=None, n_impressions=None):
    winners = self.multileaving.winning_rankers(clicks)
    # print(self.noise_method)
    self.model.update_to_mean_winners(winners, noise_method=self.noise_method, eta=self.eta, n_impressions=n_impressions, n_interactions=self.n_interactions)
