# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from algorithms.PDGD.pdgd import PDGD
import pdb

# Pairwise Differentiable Gradient Descent
class PDGD_Wrapper(PDGD):

  def __init__(self, svd, project_norm, k_initial, k_increase, _lambda=None, lambda_intp=None, lambda_intp_rate=None, prev_qeury_len=None, viewed=False, *args, **kargs):
    super(PDGD_Wrapper, self).__init__(*args, **kargs)
    self.svd = svd
    self.project_norm = project_norm
    self.k_initial = k_initial
    self.k_increase = k_increase
    self._lambda = _lambda
    
    self.lambda_intp = lambda_intp
    self.lambda_intp_rate = lambda_intp_rate
    self.prev_qeury_len = prev_qeury_len
    if prev_qeury_len:
      self.prev_feat_list = []
    self.viewed = viewed
    # self.learning_rate = learning_rate
    # self.learning_rate_decay = learning_rate_decay
    # self.model = LinearModel(n_features = self.n_features,
    #                          learning_rate = learning_rate,
    #                          learning_rate_decay = learning_rate_decay,
    #                          n_candidates = 1)

  # def __init__(self, hidden_layers, *args, **kargs):
  #   super(DeepPDGD, self).__init__(*args, **kargs)
  #   self.model = NeuralModel(n_features = self.n_features,
  #                            learning_rate = self.learning_rate,
  #                            learning_rate_decay = self.learning_rate_decay,
  #                            hidden_layers = hidden_layers)

  # @staticmethod
  # def default_parameters():
  #   parent_parameters = PDGD_Wrapper.default_parameters()
  #   parent_parameters.update({
  #     'learning_rate': 0.01,
  #     'hidden_layers': [64],
  #     })
  #   return parent_parameters


# update with wrapper
  def _update_to_clicks(self, clicks):
    n_docs = self.ranking.shape[0]
    cur_k = np.minimum(n_docs, self.n_results)

    included = np.ones(cur_k, dtype=np.int32)
    if not clicks[-1]:
      included[1:] = np.cumsum(clicks[::-1])[:0:-1]
    neg_ind = np.where(np.logical_xor(clicks, included))[0]
    pos_ind = np.where(clicks)[0]

    n_pos = pos_ind.shape[0]
    n_neg = neg_ind.shape[0]
    n_pairs = n_pos*n_neg

    if n_pairs == 0:
      return
    pos_r_ind = self.ranking[pos_ind]
    neg_r_ind = self.ranking[neg_ind]

    # print('clicks: %s, pos_ind: %s, neg_ind: %s pos_r_ind: %s'%(clicks,pos_ind,neg_ind,pos_r_ind))
    ###############################################################
    # For projection
    # keep track of feature vectors of doc list
    viewed_list = []
    # index of last click
    last_click = max(pos_ind)
    # prevent last_click+k from exceeding interleaved list length
    k_current = self.k_initial
    if self.k_increase:
      # gradually increast k
      k_current += int(self.n_interactions/1000)
    last_doc_index = min(last_click+k_current, len(self._last_ranking)-1)
    # print(last_doc_index)

    # query_feat = self.get_query_features(self.query_id,
    #                                  self._train_features,
    #                                  self._train_query_ranges)
    # _last_query_feat
    for i in range(last_doc_index):
        docid = self.ranking[i]
        feature = self._last_query_feat[docid]
        viewed_list.append(feature)

    ###############################################################

    pos_scores = self.doc_scores[pos_r_ind]
    neg_scores = self.doc_scores[neg_r_ind]

    log_pair_pos = np.tile(pos_scores, n_neg)
    log_pair_neg = np.repeat(neg_scores, n_pos)

    pair_trans = 18 - np.maximum(log_pair_pos, log_pair_neg)
    exp_pair_pos = np.exp(log_pair_pos + pair_trans)
    exp_pair_neg = np.exp(log_pair_neg + pair_trans)

    pair_denom = (exp_pair_pos + exp_pair_neg)
    pair_w = np.maximum(exp_pair_pos, exp_pair_neg)
    pair_w /= pair_denom
    pair_w /= pair_denom
    pair_w *= np.minimum(exp_pair_pos, exp_pair_neg)

    pair_w *= self._calculate_unbias_weights(pos_ind, neg_ind)

    reshaped = np.reshape(pair_w, (n_neg, n_pos))
    pos_w =  np.sum(reshaped, axis=0)
    neg_w = -np.sum(reshaped, axis=1)

    all_w = np.concatenate([pos_w, neg_w])
    all_ind = np.concatenate([pos_r_ind, neg_r_ind])

    ###############################################################
    # self.project_to_interleaved_doc(total,viewed_list)

    # weighted_docs = self._last_features[doc_ind, :] * doc_weights[:, None]
    # gradient = np.sum(weighted_docs, axis=0)
    # self.weights[:, 0] += self.learning_rate * gradient
    ###############################################################
    self.model.update_to_documents(all_ind,all_w,viewed_list,self.svd,self.project_norm)
