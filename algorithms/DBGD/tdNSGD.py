# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.linearmodel import LinearModel
from algorithms.DBGD.tddbgd import TD_DBGD
import numpy as np
from sys import maxint
import copy
from scipy.spatial.distance import cosine
import utils.rankings as rnk
# Dueling Bandit Gradient Descent
class TD_NSGD(TD_DBGD):

  def __init__(self, n_candidates, GRAD_SIZE, EXP_SIZE, *args, **kargs):
    super(TD_NSGD, self).__init__(*args, **kargs)
    self.model = LinearModel(n_features = self.n_features,
                             learning_rate = self.learning_rate,
                             n_candidates = n_candidates)
    self.GRAD_SIZE = GRAD_SIZE
    self.EXP_SIZE = EXP_SIZE
    self.sample_basis = True
    self.clicklist = np.empty([self.GRAD_SIZE,1], dtype=int) #click array
    self.grad = np.zeros([self.GRAD_SIZE,self.n_features], dtype=float)
    self.gradCol = 0
  @staticmethod
  def default_parameters():
    parent_parameters = TD_DBGD.default_parameters()
    parent_parameters.update({
      'n_candidates': 9,
      })
    return parent_parameters

  def update_to_interaction(self, clicks):
    winners, ranker_clicks = self.multileaving.winning_rankers_with_clicks(clicks)
    # print (ranker_clicks)
    self.model.update_to_mean_winners(winners)

    cl_sorted = sorted(ranker_clicks) # in ascending order
    for i in range(1, len(ranker_clicks)):
        # only save subset of rankers (worst 4 ouf of 9 rankers)
        # add if current cl is smaller than or equal to maximum form the set of candidates
        if ranker_clicks[i] <= cl_sorted[3] and ranker_clicks[i]<ranker_clicks[0]:
            # print ('update')
            self.clicklist[self.gradCol] = ranker_clicks[i] -ranker_clicks[0]
            self.grad[self.gradCol] = self.model.gs[i-1]
            self.gradCol = (self.gradCol + 1) % self.GRAD_SIZE # update to reflect next column to be updaed



  def _create_train_ranking(self, query_id, query_feat, inverted):
    assert inverted == False
    #  Get the worst gradients by click
    nums = []
    dif = self.GRAD_SIZE - self.EXP_SIZE
    for i in range(0, dif):
        max = -maxint-1
        n = 0
        # Choose
        for j in range(0, self.GRAD_SIZE):
            if self.clicklist[j] > max and j not in nums:
                max = self.clicklist[j] #  The better cl value to be excluded
                n = j # index of it
        nums.append(n)

    #  create subset of gradient matrix
    grad_temp = np.empty([self.EXP_SIZE, self.n_features], dtype=float)
    c = 0
    for i in range(0,self.GRAD_SIZE):
        if i not in nums:
            # The wrost 'EXP_SIZE' gradients from grad[] added to gr_temp
            grad_temp[c] = copy.deepcopy(self.grad[i])
            c = c + 1

    self.model.sample_candidates_null_space(grad_temp, query_feat, self.sample_basis)
    scores = self.model.candidate_score(query_feat)
    rankings = rnk.rank_single_query(scores, inverted=False, n_results=self.n_results)
    multileaved_list = self.multileaving.make_multileaving(rankings)
    return multileaved_list
