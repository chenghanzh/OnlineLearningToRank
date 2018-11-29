# -*- coding: utf-8 -*-

import sys
import os
# import random.randint
from sympy import Matrix
from scipy.linalg import norm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils.rankings as rnk
from algorithms.DBGD.tddbgd import TD_DBGD
from multileaving.ProbabilisticMultileave import ProbabilisticMultileave

# Probabilistic Interleaving Dueling Bandit Gradient Descent
class P_DBGD_Wrapper_exp(TD_DBGD):

  def __init__(self, svd, project_norm, k_initial, k_increase, PM_n_samples, PM_tau, use_NS, use_all_listed, *args, **kargs):
    super(P_DBGD_Wrapper_exp, self).__init__(*args, **kargs)
    # self.multileaving = ProbabilisticMultileave(
    #                          n_samples = PM_n_samples,
    #                          tau = PM_tau,
    #                          n_results=self.n_results)
    self.svd = svd
    self.project_norm = project_norm
    self.k_initial = k_initial
    self.k_increase = k_increase

    ### Experimental parameters
    # Whether to use g' (DS) only or g'' (NS of sampled)
    self.use_NS = use_NS

    # Whether to sample from relevant only or all listed documents
    self.use_all_listed = use_all_listed
    # self.model.weights = self.model.weights[:,0].T
    # self.model.weights = self.model.weights[:,0:1]
    # print(self.model.weights[:,0])


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
    # Save query_id to get access to query_feat when updating
    self.query_id = query_id
    assert inverted==False
    # self.model.sample_candidates()
    scores = self.model.candidate_score(query_feat)
    # print(self.rankings)
    rankings = rnk.rank_single_query(scores,
                                    inverted=False,
                                    n_results=self.n_results)
    # print(rankings)
    # print(self.model.weights)

    # verify whether this is correct
    # print("##############################")
    # print(rankings[0])
    # multileaved_list = self.multileaving.make_multileaving(rankings)
    return rankings[0]

    ## Save query_id to get access to query_feat when updating
    # self.query_id = query_id
    # assert inverted==False
    # self.model.sample_candidates()
    # scores = self.model.candidate_score(query_feat)
    # inverted_rankings = rnk.rank_single_query(scores,
    #                                           inverted=True,
    #                                           n_results=None)
    # multileaved_list = self.multileaving.make_multileaving(inverted_rankings)
    # return multileaved_list


  def update_to_interaction(self, clicks):
    # print("svd: %s, project_norm: %s " %(self.svd,self.project_norm))
    # winners = self.multileaving.winning_rankers(clicks)
    ###############################################################
    gradient_list = []
    gradient_list_NS = []

    query_feat = self.get_query_features(self.query_id,
                                     self._train_features,
                                     self._train_query_ranges)
    # if True in clicks:
    for i in range(len(self._last_ranking)-1):
      docid = self._last_ranking[i]
      feature = query_feat[docid]
      # if use all listed, add to gradient list
      # else, add only if clicked.
      if self.use_all_listed or (not self.use_all_listed and clicks[i] == 1):
        gradient_list.append(feature)


    if len(gradient_list) > 0:
      self.update_to_experimental_winners(gradient_list)


  def update_to_experimental_winners(self, gradient_list):
    # assert self.n_models > 1

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


    self.model.weights[:, 0] += self.model.learning_rate * gradient
    self.model.learning_rate *= self.model.learning_rate_decay
