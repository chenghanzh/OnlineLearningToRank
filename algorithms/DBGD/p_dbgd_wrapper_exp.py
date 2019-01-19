# -*- coding: utf-8 -*-

import sys
import os
# import random.randint
from sympy import Matrix
from scipy.linalg import norm
import numpy as np
from numpy import dot

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils.rankings as rnk
import utils.evaluate as evaluate
from algorithms.DBGD.tddbgd import TD_DBGD
from multileaving.ProbabilisticMultileave import ProbabilisticMultileave

# Probabilistic Interleaving Dueling Bandit Gradient Descent
class P_DBGD_Wrapper_exp(TD_DBGD):

  def __init__(self, svd, project_norm, k_initial, k_increase, PM_n_samples, PM_tau, use_regular_sample, use_NDCG, use_NS, use_all_listed, *args, **kargs):
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
    self.use_regular_sample = use_regular_sample
    # Whether to use g' (DS) only or g'' (NS of sampled)
    self.use_NS = use_NS
    # Whether to sample from relevant only or all listed documents
    self.use_all_listed = use_all_listed
    self.use_NDCG = use_NDCG
    self.win_count=0
    self.loss_count=0


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

  def get_gradient_list_label(self,query_feat,query_label):
    self.selected_docnum = []
    # sample candidates based on label, not clicks
    gradient_list = []

    # Add features to the gradient_list if the label > 0
    for i in range(len(query_feat)):
      if query_label[i] > 0 :
        gradient_list.append(query_feat[i])
        self.selected_docnum.append(i)
    return gradient_list

  def sample_gradient(self, gradient_list):
    if (self.loss_count + self.win_count > 0) and self.n_interactions%500 == 0:
      print("###### Win Rate at %s: " %self.n_interactions)
      # print(self.win_count/float(self.loss_count + self.win_count))
      print(self.win_count/500.0)


    # Normalize doc_space gradient
    for i in range(0, len(gradient_list)):
      norm = np.linalg.norm(gradient_list[i])
      if norm > 0:
        gradient_list[i] = gradient_list[i]/norm

    if len(gradient_list) == 1:
      gradient = gradient_list[0]
    else:
      rand_num = np.random.randint(0, len(gradient_list) - 1)
      gradient = gradient_list[rand_num]
      # print("DocID selected: %s"%self.selected_docnum[rand_num])


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

    # ##### Sample with Basis (add gausian weights)
    # weight = np.random.normal(0, 1, len(gradient))
    # gradient = weight.dot(gradient)
    # gradient /= np.linalg.norm(gradient)

    # Set the candidate ranker's weight
    self.model.weights[:, 1] = self.model.weights[:, 0] + (gradient)




  def _create_train_ranking(self, query_id, query_feat, inverted):
    assert inverted==False
    query_label = self.get_query_label(query_id,
                                     self._train_label,
                                     self._train_query_ranges)

    # print("query_feat: %s"%query_feat)
    # if query_id == 10:
    #   print("query_feat: %s"%query_feat)
    #   print("query_label: %s"%query_label)
    # print(query_label)
    # Get graident list
####################################################################################
    # In original DBGD: 
    # if self.n_interactions == 1000:
    #   print("##### Current weight")
    #   print(np.transpose(self.model.weights[:,0]))
    # temp = np.square(query_feat)
    # print(np.sum(temp, axis=0)) # sum each column: 41
    # print(np.sum(temp, axis=1)) # sum each row: 40, 50
    # print(np.linalg.norm(temp ))
    # print(temp.shape)
    # print("feat: %s" %(query_feat))
    # print("feat_sq: %s" %(temp))
    # print(sum(temp).shape)
    # print("sum(feat): %s" %(sum(temp)))
    # print("sum(feat[0]): %s" %(sum(temp, axis=0)))
    # print("##################")
    if self.use_regular_sample:
      self.model.sample_candidates()
      # self.model.weights[:, 1] = abs(self.model.weights[:, 1])
      # print(self.model.weights)
    else:
      gradient_list = self.get_gradient_list_label(query_feat, query_label) # Based on label
      # Sample gradients from the list
      if len(gradient_list) > 0 :
        self.sample_gradient(gradient_list)
      else:
        # if no gradient_list, just use current weight
        self.model.weights[:, 1:] = self.model.weights[:, 0, None]
####################################################################################
    

    scores = self.model.candidate_score(query_feat)
    # Inverted ranking used for Prob. Interleaving
    self.inverted_rankings = rnk.rank_single_query(scores,
                                              inverted=True,
                                              n_results=None)

    # Ranking used for NDCG evaluation
    self.descending_rankings = rnk.rank_single_query(scores,
                                              inverted=False,
                                              n_results=None)

    multileaved_list = self.multileaving.make_multileaving(self.inverted_rankings)

    return multileaved_list

  # Can be modified for multileaving 
  def update_to_interaction(self, clicks, stop_index=None):

    ##############
    # Dubugging purpose
    if self.n_interactions == 500:
      weight_optimal_dbgd = [0.02832735, 0.01285759, -0.05165583, 0.03138522, -0.03074056, 0.01276765,
      0.03291753, -0.03352945, -0.00616694, -0.05204681,  0.01169881,  0.02033736,
      0.02435693,  0.00357046, -0.01496411, -0.04720677,  0.02733036, -0.01115817,
      0.04149337, -0.03569844,  0.00154001,  0.00856143,  0.02862681, -0.00299965,
      -0.05985323, -0.06145907, -0.03056593,  0.03600999,  0.01145822, -0.03381266,
      -0.01912709, -0.02858585, -0.00074354, -0.03987856, -0.0166531,  -0.02060764,
      0.01061898, -0.01129424, -0.04443411, -0.01374962, -0.01553134]

      self.model.weights[:, 0] = weight_optimal_dbgd
      self.win_count = 0
      self.loss_count = 0

    # Option #2, directly use NDCG of two lists to determine winner
    if(self.use_NDCG):
      query_label = self.get_query_label(self._last_query_id,
                                       self._train_label,
                                       self._train_query_ranges)
      # for ranked_list in rankings:
      ndcg_list_inv = evaluate.get_single_ndcg_for_rankers(self.inverted_rankings,query_label,10)
      ndcg_list = evaluate.get_single_ndcg_for_rankers(self.descending_rankings,query_label,10)
      winners = []

      if ndcg_list[1] > ndcg_list[0] :
        winners = [1]
        self.win_count += 1
      elif ndcg_list[1] < ndcg_list[0]:
        self.loss_count += 1
      # print(winners)
    # Option #1, Use Multileaving
    # Same as DBGD, using interleaved result
    else:
      winners = self.multileaving.winning_rankers(clicks)
      # if self.n_interactions %50 == 0:
      #   print(winners)
      #   print("########################################################################################################")
    # if winners == [1]:
    #   print("NEW WINNER!!!!!")
    self.model.update_to_mean_winners(winners) # , apply_LR=False




