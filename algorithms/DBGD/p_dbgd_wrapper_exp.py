# -*- coding: utf-8 -*-

import sys
import os
# import random.randint
from sympy import Matrix
from scipy.linalg import norm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils.rankings as rnk
import utils.evaluate as evaluate
from algorithms.DBGD.tddbgd import TD_DBGD
from multileaving.ProbabilisticMultileave import ProbabilisticMultileave

# Probabilistic Interleaving Dueling Bandit Gradient Descent
class P_DBGD_Wrapper_exp(TD_DBGD):

  def __init__(self, svd, project_norm, k_initial, k_increase, PM_n_samples, PM_tau, use_NS, use_all_listed, *args, **kargs):
    super(P_DBGD_Wrapper_exp, self).__init__(*args, **kargs)
    self.multileaving = ProbabilisticMultileave(
                             n_samples = PM_n_samples,
                             tau = PM_tau,
                             n_results=self.n_results)
    self.svd = svd
    self.project_norm = project_norm
    self.k_initial = k_initial
    self.k_increase = k_increase
    self.gradient_list = []

    ### Experimental parameters
    # Whether to use g' (DS) only or g'' (NS of sampled)
    self.use_NS = use_NS
    # Whether to sample from relevant only or all listed documents
    self.use_all_listed = use_all_listed


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

  def get_gradient_list_label(self):
    # sample candidates based on label, not clicks
    gradient_list = []
    query_feat = self.get_query_features(self.query_id,
                                     self._train_features,
                                     self._train_query_ranges)
    # print(self.datafold.train_label_vector.shape)
    query_label = self.get_query_label(self.query_id,
                                     self._train_label,
                                     self._train_query_ranges)

    # Add features to the gradient_list if the label > 0
    for i in range(len(query_feat)-1):
      if query_label[i] > 0 : 
        gradient_list.append(query_feat[i])

    return gradient_list

  def sample_gradient(self, gradient_list):
    # Normalize doc_space gradient
    for i in range(0, len(gradient_list)):
      norm = np.linalg.norm(gradient_list[i])
      if norm > 0:
        gradient_list[i] = gradient_list[i]/norm

    if len(gradient_list) == 1:
      gradient = gradient_list[0]
    else:
      gradient = gradient_list[np.random.randint(0, len(gradient_list) - 1)]


    if self.use_NS:
      NS = Matrix(gradient_list).nullspace() #  get null space of gradient matrix
      NS = np.array(NS).astype(np.float64)
      # Normalize NS
      for i in range(0, len(NS)):
        norm = np.linalg.norm(NS[i])
        if norm > 0:
          NS[i] = NS[i]/norm
      # sample gradient NS
      if len(NS) == 1:
        gradient_NS = NS[0]
      else:
        gradient_NS = NS[np.random.randint(0, len(NS) - 1)]

      gradient = gradient + gradient_NS

    # Set the candidate ranker's weight
    self.model.weights[:, 1] = self.model.weights[:, 0] + gradient



  def _create_train_ranking(self, query_id, query_feat, inverted):
    # Save query_id to get access to query_feat when updating
    self.query_id = query_id
    assert inverted==False
    # In original DBGD: 
    # self.model.sample_candidates()
####################################################################################
    # Get graident list
    gradient_list = self.get_gradient_list_label() # Based on label
    # Sample gradients from the list
    if len(gradient_list) > 0 :
      self.sample_gradient(gradient_list)
    else:
      # if no gradient_list, just use current weight
      self.model.weights[:, 1:] = self.model.weights[:, 0, None]
####################################################################################
    scores = self.model.candidate_score(query_feat)
    inverted_rankings = rnk.rank_single_query(scores,
                                              inverted=True,
                                              n_results=None)
    self.descending_rankings = rnk.rank_single_query(scores,
                                              inverted=False,
                                              n_results=None)
    multileaved_list = self.multileaving.make_multileaving(inverted_rankings)
    return multileaved_list

  # Currently 
  # Can be modified for multileaving 
  def update_to_interaction(self, clicks):
    # Same as DBGD, using interleaved result
    # winners = self.multileaving.winning_rankers(clicks)
    # self.model.update_to_mean_winners(winners)

    # NDCG (using label) to determine winner directly.
    query_label = self.get_query_label(self.query_id,
                                     self._train_label,
                                     self._train_query_ranges)
    # for ranked_list in rankings:
    ndcg_list = evaluate.get_single_ndcg_for_rankers(self.descending_rankings,query_label,10)

    winners = []
    if ndcg_list[1] > ndcg_list[0] :
      winners = [1]
    self.model.update_to_mean_winners(winners)




