# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.linearmodel import LinearModel
from algorithms.DBGD.tddbgd import TD_DBGD

# Dueling Bandit Gradient Descent
class TD_MGD_Wrapper_Global(TD_DBGD):

  def __init__(self, n_candidates, *args, **kargs):
    super(TD_MGD_Wrapper_Global, self).__init__(*args, **kargs)
    self.model = LinearModel(n_features = self.n_features,
                             learning_rate = self.learning_rate,
                             n_candidates = n_candidates)

  @staticmethod
  def default_parameters():
    parent_parameters = TD_DBGD.default_parameters()
    parent_parameters.update({
      'n_candidates': 9
      })
    return parent_parameters

  def update_to_interaction(self, clicks, impressions=None):
    # print("svd: %s, project_norm: %s " %(self.svd,self.project_norm))
    winners = self.multileaving.winning_rankers(clicks)
    self.model.update_to_mean_winners(winners, impressions)