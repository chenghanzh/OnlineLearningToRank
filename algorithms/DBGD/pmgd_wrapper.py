# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from algorithms.DBGD.pdbgd import P_DBGD
import utils.rankings as rnk
from models.linearmodel import LinearModel
import numpy as np
import math


# Probabilistic Interleaving Dueling Bandit Gradient Descent
class P_MGD_Wrapper(P_DBGD):

  def __init__(self, svd, project_norm, k_initial, k_increase, n_candidates, _lambda=None, lambda_intp=None, lambda_intp_dec=None, prev_qeury_len=None, viewed=False, docspace=[False,0], *args, **kargs):
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
    self.prev_qeury_len = prev_qeury_len
    if prev_qeury_len:
      self.prev_feat_list = []
    self.viewed = viewed
    self.docspace = docspace  # docspace=[False,0]


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

  def update_to_interaction(self, clicks, stop_index=None):
    # print(len(self._last_ranking))
    if self.lambda_intp_dec == 'dec':
      self.lambda_intp = 0.9996 ** self.n_interactions # 0.9996^t
    if self.lambda_intp_dec == 'dec_9999':
      self.lambda_intp = 0.9999 ** self.n_interactions # 0.9998^t
    if self.lambda_intp_dec == 'dec_9998':
      self.lambda_intp = 0.9998 ** self.n_interactions # 0.9998^t
    if self.lambda_intp_dec == 'dec_9994':
      self.lambda_intp = 0.9994 ** self.n_interactions # 0.9994^t
    if self.lambda_intp_dec == 'dec_9992':
      self.lambda_intp = 0.9992 ** self.n_interactions # 0.9994^t
    elif self.lambda_intp_dec == 'inc':
      self.lambda_intp =  1 - math.exp(-0.0006 * self.n_interactions) # 1-e^(-.0006*t)
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
      last_doc_index = min(last_click+k_current, len(self._last_ranking))

      if self.docspace[0] and stop_index is not None: # for test document space length experiment
        # create sub/super set of perfect document space user examined. 
        # user examined documents coming from ccm, where user leaves.
        last_doc_index = stop_index + self.docspace[1] + 1 # 1 added for stopping document, which has been examined.
        last_doc_index = max(last_doc_index,1) # At least 1
        last_doc_index = min(last_doc_index,len(self._last_ranking)) # At most length of current list


      query_feat = self.get_query_features(self.query_id,
                                       self._train_features,
                                       self._train_query_ranges)

      for i in range(last_doc_index):
        docid = self._last_ranking[i]
        feature = query_feat[docid]
        viewed_list.append(feature)
      if self.viewed: # Add examined document, depending on config setting
          add_list = viewed_list



      ##### Append feature vectors from previous queries
      if self.prev_qeury_len:
        if len(self.prev_feat_list) > 0:
          viewed_list = np.append(viewed_list,self.prev_feat_list, axis=0)


        # Add new feature vectors of current query to be used in later iterations
        if self.viewed: # Add examined document, depending on config setting
          for i in add_list:
            if len(self.prev_feat_list) >= self.prev_qeury_len :
              self.prev_feat_list.pop(0)  # Remove oldest document feature.
            # if prev_feat_list is not filled up, add current list
            self.prev_feat_list.append(i)


        else: # Add ONLY from clicked document
          add_list = [loc for loc, val in enumerate(clicks) if val == True]
          for i in add_list:
            docid_c = self._last_ranking[i]
            feature_c = query_feat[docid_c]

            # Remove the oldest document feature.
            if len(self.prev_feat_list) >= self.prev_qeury_len :
              self.prev_feat_list.pop(0)
            self.prev_feat_list.append(feature_c)

      self.model.update_to_mean_winners(winners,viewed_list,self.svd,self.project_norm, _lambda=self._lambda, lambda_intp=self.lambda_intp)
    ###############################################################
    else:
      self.model.update_to_mean_winners(winners)