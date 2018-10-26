# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.linearmodel import LinearModel
import utils.rankings as rnk
from algorithms.DBGD.tdmgd import TD_MGD

# Dueling Bandit Gradient Descent
class TD_MGD_Wrapper(TD_MGD):

  def __init__(self, n_candidates, *args, **kargs):
    super(TD_MGD, self).__init__(*args, **kargs)
  #   self.model = LinearModel(n_features = self.n_features,
  #                            learning_rate = self.learning_rate,
  #                            n_candidates = n_candidates)

  # @staticmethod
  # def default_parameters():
  #   parent_parameters = TD_DBGD.default_parameters()
  #   parent_parameters.update({
  #     'n_candidates': 9,
  #     })
  #   return parent_parameters
  def update_to_interaction(self, clicks):
    # n_docs = self.ranking.shape[0]
    # cur_k = np.minimum(n_docs, self.n_results)
    # print(n_results)

    # included = np.ones(cur_k, dtype=np.int32)
    # if not clicks[-1]:
    #   included[1:] = np.cumsum(clicks[::-1])[:0:-1]
    # neg_ind = np.where(np.logical_xor(clicks, included))[0]
    # pos_ind = np.where(clicks)[0]

    # n_pos = pos_ind.shape[0]
    # n_neg = neg_ind.shape[0]
    # n_pairs = n_pos*n_neg

    # if n_pairs == 0:
    #   return

    # pos_r_ind = self.ranking[pos_ind]
    # neg_r_ind = self.ranking[neg_ind]


    winners = self.multileaving.winning_rankers(clicks)
    ###############################################################

    if True in clicks:
      # For projection
      # keep track of feature vectors of doc list
      self.TOP_K = 3
      viewed_list = []
      # index of last click
      last_click = max(loc for loc, val in enumerate(clicks) if val == True)
      # prevent last_click+k from exceeding interleaved list length
      last_doc_index = min(last_click+self.TOP_K, len(self._last_ranking)-1)
      # print(last_doc_index)

      query_feat = self.get_query_features(self.query_id,
                                       self._train_features,
                                       self._train_query_ranges)
      for i in range(last_doc_index):
        docid = self._last_ranking[i]
        feature = query_feat[docid]
        viewed_list.append(feature)
      self.model.update_to_mean_winners(winners,viewed_list)

    ###############################################################
    else:
      self.model.update_to_mean_winners(winners)


  def _create_train_ranking(self, query_id, query_feat, inverted):
    # Save query_id to get access to query_feat when updating
    self.query_id = query_id
    assert inverted == False
    self.model.sample_candidates()
    scores = self.model.candidate_score(query_feat)
    rankings = rnk.rank_single_query(scores, inverted=False, n_results=self.n_results)
    multileaved_list = self.multileaving.make_multileaving(rankings)
    return multileaved_list
