# -*- coding: utf-8 -*-

import time
import numpy as np
from evaluate import get_idcg_list, evaluate, evaluate_ranking
from clicks import *

from numpy.linalg import norm

def cosine(u, v):
  return np.dot(u, v) / (norm(u) * norm(v))

class SingleSimulation(object):

  def __init__(self, sim_args, output_queue, click_model, datafold):
    self.train_only = sim_args.train_only
    self.n_impressions = sim_args.n_impressions

    self.n_results = sim_args.n_results
    self.click_model = click_model
    self.switch_click_model = sim_args.switch_click_model
    self.click_models = [get_click_models(['nav','bin'])[0], get_click_models(['inf','bin'])[0]]

    self.datafold = datafold
    if not self.train_only:
      self.test_idcg_vector = get_idcg_list(self.datafold.test_label_vector,
                                            self.datafold.test_doclist_ranges,
                                            self.n_results, spread=True)
    self.train_idcg_vector = get_idcg_list(self.datafold.train_label_vector,
                                           self.datafold.train_doclist_ranges,
                                           self.n_results)

    self.run_details = {
            'data folder': str(self.datafold.data_path),
            'held-out data': str(self.datafold.heldout_tag),
            'click model': self.click_model.get_name(),
          }
    self.output_queue = output_queue

    self.print_frequency = sim_args.print_freq
    self.print_all_train = sim_args.all_train
    self.print_logscale = sim_args.print_logscale
    if self.print_logscale:
      self.print_scale = self.print_frequency
      self.print_next_scale = self.print_scale
      self.print_frequency = 1

    self.last_print = 0
    self.next_print = 0
    self.online_score = 0.0
    self.cur_online_discount = 1.0
    self.online_discount = 0.9995

    self.eval_gradient = False
    # Load w^* for simulation data
    if sim_args.w_star_path is not None:
        self.eval_gradient = True
        with open(sim_args.w_star_path, 'r') as f:
          self.w_star = np.loadtxt(f)
          print (self.w_star)

  def timestep_evaluate(self, results, iteration, ranker, ranking_i,
                        train_ranking, ranking_labels):

    test_print = (not self.train_only
                  and (iteration == self.last_print
                       or iteration == self.next_print
                       or iteration == self.n_impressions))

    if test_print:
      cur_results = self.evaluate_ranker(iteration,
                                         ranker,
                                         ranking_i,
                                         train_ranking,
                                         ranking_labels)
      self.online_score += cur_results['display']*self.cur_online_discount
      cur_results['cumulative-display'] = self.online_score
      results.append(cur_results)
    else:
      cur_results = self.evaluate_ranker_train_only(iteration,
                                                    ranker,
                                                    ranking_i,
                                                    train_ranking,
                                                    ranking_labels)
      self.online_score += cur_results['display']*self.cur_online_discount
      if self.print_all_train:
        cur_results['cumulative-display'] = self.online_score
        results.append(cur_results)

    self.cur_online_discount *= self.online_discount

    if iteration >= self.next_print:
      if self.print_logscale and iteration >= self.print_next_scale:
          self.print_next_scale *= self.print_scale
          self.print_frequency *= self.print_scale

      self.last_print = self.next_print
      self.next_print = self.next_print + self.print_frequency


  def evaluate_ranker(self, iteration, ranker,
                      ranking_i, train_ranking,
                      ranking_labels):

    test_rankings = ranker.get_test_rankings(
                    self.datafold.test_feature_matrix,
                    self.datafold.test_doclist_ranges,
                    inverted=True)
    test_ndcg = evaluate(
                  test_rankings,
                  self.datafold.test_label_vector,
                  self.test_idcg_vector,
                  self.datafold.test_doclist_ranges.shape[0] - 1,
                  self.n_results)

    train_ndcg = evaluate_ranking(
            train_ranking,
            ranking_labels,
            self.train_idcg_vector[ranking_i],
            self.n_results)

    results = {
      'iteration': iteration,
      'heldout': np.mean(test_ndcg),
      'display': np.mean(train_ndcg),
    }

    for name, value in ranker.get_messages().items():
      results[name] = value

    return results

  def evaluate_ranker_train_only(self, iteration, ranker,
                                 ranking_i, train_ranking,
                                 ranking_labels):

    train_ndcg = evaluate_ranking(
            train_ranking,
            ranking_labels,
            self.train_idcg_vector[ranking_i],
            self.n_results)

    results = {
      'iteration': iteration,
      'display': np.mean(train_ndcg),
    }

    for name, value in ranker.get_messages().items():
      results[name] = value

    return results

  def evaluate_gradient(self, results, iteration, ranker,):
    results[-1]['cosine_w'] = cosine(ranker.model.weights[:, 0].T, self.w_star)
    results[-1]['l2_w'] = norm(ranker.model.weights[:, 0].T - self.w_star)
    if ranker.model.g_t is not None:      
      results[-1]['cosine_g'] = cosine(ranker.model.g_t, self.w_star-ranker.model.weights[:, 0].T)
      results[-1]['cosine_g_diff'] = cosine(ranker.model.g_t-ranker.model.u_t, self.w_star-ranker.model.weights[:, 0].T)
    else:
      results[-1]['cosine_g_diff'] = 0 #or np.nan?
      results[-1]['cosine_g'] = 0 #or np.nan?
      
      # print (results[-1]['cosine'])


  def sample_and_rank(self, ranker, impressions=None):
    if impressions == None:
      ranking_i = np.random.choice(self.datafold.n_train_queries())
    else:
      ranking_i = impressions % self.datafold.n_train_queries()
    # print (impressions)
    
    train_ranking = ranker.get_train_query_ranking(ranking_i)

    assert train_ranking.shape[0] <= self.n_results, 'Shape is %s' % (train_ranking.shape,)
    assert len(train_ranking.shape) == 1, 'Shape is %s' % (train_ranking.shape,)

    return ranking_i, train_ranking


  def run(self, ranker, output_key):
    starttime = time.time()

    ranker.setup(train_features = self.datafold.train_feature_matrix,
                 train_query_ranges = self.datafold.train_doclist_ranges)

    run_results = []
    impressions = 0
    for impressions in range(self.n_impressions):
      # ranking_i, train_ranking = self.sample_and_rank(ranker)
      ranking_i, train_ranking = self.sample_and_rank(ranker, impressions)
      ranking_labels = self.datafold.train_query_labels(ranking_i)
      if self.switch_click_model:
        clicks = self.click_models[int(impressions/2500)].generate_clicks(train_ranking, ranking_labels)
      else:
        clicks = self.click_model.generate_clicks(train_ranking, ranking_labels)
      self.timestep_evaluate(run_results, impressions, ranker,
                             ranking_i, train_ranking, ranking_labels)

      ranker.process_clicks(clicks, impressions)
      if self.eval_gradient:
        self.evaluate_gradient(run_results, impressions, ranker)

    # evaluate after final iteration
    ranking_i, train_ranking = self.sample_and_rank(ranker)
    ranking_labels =  self.datafold.train_query_labels(ranking_i)
    impressions += 1
    self.timestep_evaluate(run_results, impressions, ranker,
                           ranking_i, train_ranking, ranking_labels)

    ranker.clean()

    self.run_details['runtime'] = time.time() - starttime

    output = {'run_details': self.run_details,
              'run_results': run_results}

    self.output_queue.put((output_key, output))