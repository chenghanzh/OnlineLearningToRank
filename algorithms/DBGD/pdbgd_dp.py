# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils.rankings as rnk
from algorithms.DBGD.tddbgd import TD_DBGD
from multileaving.ProbabilisticMultileave import ProbabilisticMultileave

# Probabilistic Interleaving Dueling Bandit Gradient Descent
class P_DBGD_dp(TD_DBGD):

  def __init__(self, PM_n_samples, PM_tau, noise_method, epsilon, noise_interleaving=False, *args, **kargs):
    super(P_DBGD_dp, self).__init__(*args, **kargs)
    self.multileaving = ProbabilisticMultileave(
                             n_samples = PM_n_samples,
                             tau = PM_tau,
                             n_results=self.n_results)
    self.noise_method = noise_method
    self.epsilon = epsilon
    self.noise_interleaving = noise_interleaving


  @staticmethod
  def default_parameters():
    parent_parameters = TD_DBGD.default_parameters()
    parent_parameters.update({
      'learning_rate': 0.01,
      'learning_rate_decay': 1.0,
      'PM_n_samples': 10000,
      'PM_tau': 3.0,
      })
    return parent_parameters

  def _create_train_ranking(self, query_id, query_feat, inverted):
    assert inverted==False
    self.model.sample_candidates()
    scores = self.model.candidate_score(query_feat)
    inverted_rankings = rnk.rank_single_query(scores,
                                              inverted=True,
                                              n_results=None)
    multileaved_list = self.multileaving.make_multileaving(inverted_rankings)
    return multileaved_list

  def update_to_interaction(self, clicks, stop_index=None, n_impressions=None):
    if self.noise_interleaving:
        # Add noise in interleaving by randomizing with P(w'>w) = exp(\epsilon c(w) / 2)
        print(self.multileaving.winning_rankers(clicks))
        sys.exit(0)

      # Add noise in interleaving by randomizing winner by rate of 1/epsilon
      # winners = self.multileaving.winning_rankers(clicks, 1.0/self.epsilon)
    else:
      winners = self.multileaving.winning_rankers(clicks)
    # print(self.noise_method)
    self.model.update_to_mean_winners(winners, noise_method=self.noise_method, epsilon=self.epsilon, n_impressions=n_impressions, n_interactions=self.n_interactions)
