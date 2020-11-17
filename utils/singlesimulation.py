# -*- coding: utf-8 -*-

import time
import numpy as np
from evaluate import *
from clicks import *
import scipy.stats as stats # For kendall tau
import matplotlib.pyplot as plt
import math


class SingleSimulation(object):

  def __init__(self, sim_args, output_queue, click_model, datafold):
    self.train_only = sim_args.train_only
    self.n_impressions = sim_args.n_impressions

    self.n_results = sim_args.n_results
    self.click_model = click_model
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

  def sample_and_rank(self, ranker):
    ranking_i = np.random.choice(self.datafold.n_train_queries())
    train_ranking = ranker.get_train_query_ranking(ranking_i)

    assert train_ranking.shape[0] <= self.n_results, 'Shape is %s' % (train_ranking.shape,)
    assert len(train_ranking.shape) == 1, 'Shape is %s' % (train_ranking.shape,)

    return ranking_i, train_ranking


  def run(self, ranker, output_key, attacker_output_key):
    starttime = time.time()

    ranker.setup(train_features = self.datafold.train_feature_matrix,
                 train_query_ranges = self.datafold.train_doclist_ranges)

    run_results = []
    attacker_results = []
    impressions = 0

    for impressions in range(self.n_impressions):
      ranking_i, train_ranking = self.sample_and_rank(ranker)
      freq = {}
      for i in range(0, 10):
        for r in train_ranking:
          freq[r] = freq.get(r, 0) + 1
        train_ranking = ranker.get_train_query_ranking(ranking_i)

      ranking_labels = self.datafold.train_query_labels(ranking_i)
      train_feat = ranker.get_query_features(ranking_i,
                                       self.datafold.train_feature_matrix,
                                       self.datafold.train_doclist_ranges)

      attacker_scores, attacker_ranking = get_attack_scores_and_ranking(train_feat)

      # clicks = self.click_model.generate_clicks(train_ranking, ranking_labels)
      clicks = self.click_model.generate_clicks(train_ranking, attacker_scores, attacker_ranking, freq)

      self.test_evaluation(attacker_results, impressions, ranker, clicks, self.n_results)

      self.timestep_evaluate(run_results, impressions, ranker,
                             ranking_i, train_ranking, ranking_labels)

      ranker.process_clicks(clicks)


    # # evaluate after final iteration
    # x = [i for i in range(self.n_impressions-1)]
    # # plt.plot(x, taus)
    # ndcgs = ndcgs[1:]
    # fig_ndcg, ax_ndcg = plt.subplots()
    # ax_ndcg.plot(x, ndcgs)
    # ax_ndcg.set_xlabel("Impressions")
    # ax_ndcg.set_ylabel(self.datafold.name + ": Fold " + str(self.datafold.fold_num+1) + " NDCGs")

    # plt.show()

    ranking_i, train_ranking = self.sample_and_rank(ranker)
    ranking_labels =  self.datafold.train_query_labels(ranking_i)
    impressions += 1
    self.timestep_evaluate(run_results, impressions, ranker,
                           ranking_i, train_ranking, ranking_labels)

    ranker.clean()

    self.run_details['runtime'] = time.time() - starttime

    output = {'run_details': self.run_details,
              'run_results': run_results}

    attacker_output = {'run_details': self.run_details,
                       'run_results': attacker_results}

    self.output_queue.put((output_key, output))
    self.output_queue.put((attacker_output_key, attacker_output))


  def test_evaluation(self, attacker_results, iteration, ranker, clicks, n_results):
      test_r = ranker.get_test_rankings(self.datafold.test_feature_matrix, self.datafold.test_doclist_ranges, inverted=True)

      ndcg_attack, ndcg_label, tau, tau_sum = 0, 0, 0, 0
      for test_query in range(self.datafold.test_doclist_ranges.shape[0]-1):
        start_doc = self.datafold.test_doclist_ranges[test_query]
        end_doc = self.datafold.test_doclist_ranges[test_query+1]
        test_labels = self.datafold.test_query_labels(test_query)

        test_features = self.datafold.test_feature_matrix[start_doc:end_doc, :]

        attacker_scores, attacker_ranking = get_attack_scores_and_ranking(test_features)

        assert len(attacker_ranking) == test_r[start_doc:end_doc].shape[0]

        ndcg_attack += get_ndcg_with_ranking(test_r[start_doc:end_doc], attacker_ranking, n_results)
        ndcg_label += get_ndcg_with_labels(test_r[start_doc:end_doc], test_labels, n_results)

        inverted_model_ranking = [0 for i in range(len(model_ranking))]      #  [2,3,4,1,0]  => [5, 4, 0, 1, 3]
        for i in range(len(model_ranking)):
            if model_ranking[i] < len(inverted_model_ranking):
                inverted_model_ranking[model_ranking[i]] = i
        tau, _ = stats.kendalltau(attacker_ranking, r)
        tau_sum += tau

      num_clicks = np.count_nonzero(clicks)

      cur_results = {
        'iteration': iteration,
        'NDCG_attack': ndcg_attack/test_query,
        'NDCG_label': ndcg_label/test_query,
        'Kendall\'s Tau': tau_sum/test_query,
        'Click Number': num_clicks,
      }

      for name, value in ranker.get_messages().items():
        cur_results[name] = value
      attacker_results.append(cur_results)


def get_attack_scores_and_ranking(features):
    attacker_weights_file = open("Weights.txt","r")
    attacker_weights_lines = attacker_weights_file.read().split(',')
    attacker_weights_lines = [float(i) for i in attacker_weights_lines]
    attacker_weights = np.expand_dims(np.asarray(attacker_weights_lines), axis=1)

    attacker_scores = np.dot(features, attacker_weights)
    attacker_scores = attacker_scores/np.linalg.norm(attacker_scores)
    attacker_score_document_pair = [(attacker_scores[i], i) for i in range(len(attacker_scores))]
    attacker_score_document_pair = sorted(attacker_score_document_pair, key = lambda x: (-1*x[0], x[1]))
    attacker_ranking = list(map(lambda x: x[1], attacker_score_document_pair))
    return attacker_scores, attacker_ranking
