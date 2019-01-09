# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.linearmodel import LinearModel
import utils.rankings as rnk
from algorithms.DBGD.tddbgd import TD_DBGD

# Dueling Bandit Gradient Descent
class TD_DBGD_Wrapper_Global(TD_DBGD):

  def __init__(self, svd, project_norm, k_initial, k_increase, *args, **kargs):
    super(TD_DBGD_Wrapper, self).__init__(*args, **kargs)
    self.svd = svd
    self.project_norm = project_norm
    self.k_initial = k_initial
    self.k_increase = k_increase
    # self.model = LinearModel(n_features = self.n_features,
    #                          learning_rate = self.learning_rate)


  def update_to_interaction(self, clicks, impressions=None):
    # print("svd: %s, project_norm: %s " %(self.svd,self.project_norm))
    winners = self.multileaving.winning_rankers(clicks)
    self.model.update_to_mean_winners(winners, impressions)


  def _create_train_ranking(self, query_id, query_feat, inverted):
    # Save query_id to get access to query_feat when updating
    self.query_id = query_id
    assert inverted == False
    self.model.sample_candidates()
    scores = self.model.candidate_score(query_feat)
    rankings = rnk.rank_single_query(scores, inverted=False, n_results=self.n_results)
    multileaved_list = self.multileaving.make_multileaving(rankings)
    return multileaved_list