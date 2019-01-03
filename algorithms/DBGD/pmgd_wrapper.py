# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from algorithms.DBGD.pdbgd import P_DBGD
import utils.rankings as rnk
from models.linearmodel import LinearModel
import numpy as np


# Probabilistic Interleaving Dueling Bandit Gradient Descent
class P_MGD_Wrapper(P_DBGD):

  def __init__(self, svd, project_norm, k_initial, k_increase, n_candidates, _lambda=None, lambda_intp=None, lambda_intp_dec=None, prev_qeury=None, *args, **kargs):
    super(P_MGD_Wrapper, self).__init__(*args, **kargs)
    self.n_candidates = n_candidates
    self.model = LinearModel(n_features = self.n_features,
                             learning_rate = self.learning_rate,
                             n_candidates = self.n_candidates)
    self.svd = svd
    self.project_norm = project_norm
    self.k_initial = k_initial
    self.k_increase = k_increase
    self._lambda = _lambda
    self.lambda_intp = lambda_intp
    self.lambda_intp_dec = lambda_intp_dec
    self.prev_qeury = prev_qeury

    if prev_qeury:
      self.prev_feat_list = []


  @staticmethod
  def default_parameters():
    parent_parameters = P_DBGD.default_parameters()
    parent_parameters.update({
      'n_candidates': 49,
      })
    return parent_parameters

  def _create_train_ranking(self, query_id, query_feat, inverted):
    # Save query_id to get access to query_feat when updating
    self.query_id = query_id
    assert inverted==False
    self.model.sample_candidates()
    scores = self.model.candidate_score(query_feat)
    inverted_rankings = rnk.rank_single_query(scores,
                                              inverted=True,
                                              n_results=None)
    multileaved_list = self.multileaving.make_multileaving(inverted_rankings)
    return multileaved_list  

  def update_to_interaction(self, clicks):
    if self.lambda_intp_dec:
      self.lambda_intp = 0.9996 ** self.n_interactions # 0.9996^t
    # print("svd: %s, project_norm: %s " %(self.svd,self.project_norm))
    winners = self.multileaving.winning_rankers(clicks)
    ###############################################################
    if True in clicks:
      # For projection
      # keep track of feature vectors of doc list
      viewed_list = []
      # index of last click
      last_click = max(loc for loc, val in enumerate(clicks) if val == True)
      # prevent last_click+k from exceeding interleaved list length
      k_current = self.k_initial
      if self.k_increase:
        # gradually increast k
        k_current += int(self.n_interactions/1000)
      last_doc_index = min(last_click+k_current, len(self._last_ranking)-1)
      # print(last_doc_index)

      query_feat = self.get_query_features(self.query_id,
                                       self._train_features,
                                       self._train_query_ranges)
      for i in range(last_doc_index):
        docid = self._last_ranking[i]
        feature = query_feat[docid]
        viewed_list.append(feature)



      ##### Append feature vectors from previous queries
      if self.prev_qeury:
        if len(self.prev_feat_list) > 0:
          # Append feature vectors from previous queries
          viewed_list = np.append(viewed_list,self.prev_feat_list, axis=0)

        # Add new feature vectors of current query to be used in later iterations
        # if prev_feat_list is not filled up, add current list
        click_list = [loc for loc, val in enumerate(clicks) if val == True]
        for i in click_list:
          docid_c = self._last_ranking[i]
          feature_c = query_feat[docid_c]

           # Remove oldest document feature.
          if len(self.prev_feat_list) >= self.prev_qeury :
            self.prev_feat_list.pop(0)

          self.prev_feat_list.append(feature_c)

      self.model.update_to_mean_winners(winners,viewed_list,self.svd,self.project_norm, _lambda=self._lambda, lambda_intp=self.lambda_intp)
    ###############################################################
    else:
      self.model.update_to_mean_winners(winners)