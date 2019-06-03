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

    self.wstar = np.array([0.090921813,-0.11485128,0.044372544,-0.087539188,
0.009004562,-0.072301656,-0.0688373,-0.06482818,0.096968994,-0.046205021,-0.081590528,-0.062295628,
-0.096978417,0.051877838,0.049993948,0.105052165,-0.115473766,
0.093215119,0.034444575,0.098947574,-0.106032557,0.007000408,-0.122599149,-0.163043936,-0.132420002,
0.008172192,-0.025648746,0.024661343,-0.060795502,0.049398335,-0.043957122,-0.073410604,0.01488797,
-0.04552856,-0.011800254,-0.016397403,-0.040835005,-0.025397195,-0.04686599,-0.13255839,0.02296312,0.00791,-0.105128702,0.120163274,
0.116768347,0.036304662])




    self.wstar2 = np.array([0.05487592855960689,-0.16177660823499504,-0.1326441616464022,-0.025244180904836714,0.0608957589996712,
0.03565139323472977,-0.08807320147752762,-0.052409082651138306,-0.017523562535643578,0.04421379417181015,
-0.11383271436743246, -0.05965518118458506, -0.1642389763390179, -0.01770502037805504, -0.0020050693877653896,
0.01625400976524229, -0.012986437299060776, 0.001044217962890701, -0.05650360734995826, -0.09585559186965918,
-0.11594187169150313, -0.01945667628924064, -0.09069662303376838, -0.10249463908900396, -0.19089999340880087,
-0.16589926061760352, -0.2057328359134015, -0.09014762644324936, -0.08652332692572763, -0.17825847630106761,
-0.1028191429653127, -0.17696874180478117, 0.043575013356484345, 0.012777915201885627, -0.061118497356137044,
 0.031164346680439256, -0.0014622776294397632, -0.14564148008167005, -0.11557535752333076, -0.15106161690020906,
 -0.05217807212105205, -0.0842433198861431, -0.05336505571525466, -0.0555042398092651, -7.019967011790595E-4,
 0.0540026950094062])
    # # For Ordered queries experiment
    # self.ordered_queries = None
    # if sim_args.ordered_queries:
    #   self.ordered_queries = [line.strip() for line in open(sim_args.ordered_queries)]


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

# version for ordered queries
  # def sample_and_rank(self, ranker, impressions=None):
  #   if self.ordered_queries and impressions:
  #     assert self.n_impressions < self.datafold.n_train_queries()
  #     ranking_i = int(self.ordered_queries[impressions])
  #   else:
  #     ranking_i = np.random.choice(self.datafold.n_train_queries())
  #   train_ranking = ranker.get_train_query_ranking(ranking_i)

  #   assert train_ranking.shape[0] <= self.n_results, 'Shape is %s' % (train_ranking.shape,)
  #   assert len(train_ranking.shape) == 1, 'Shape is %s' % (train_ranking.shape,)

  #   return ranking_i, train_ranking

  def sample_and_rank(self, ranker):
    ranking_i = np.random.choice(self.datafold.n_train_queries())
    train_ranking = ranker.get_train_query_ranking(ranking_i)

    assert train_ranking.shape[0] <= self.n_results, 'Shape is %s' % (train_ranking.shape,)
    assert len(train_ranking.shape) == 1, 'Shape is %s' % (train_ranking.shape,)
    return ranking_i, train_ranking


  # Record gradient info
  def record_gradient(self, results, iteration, ranker,):
    if ranker.model.g_t is not None: 
      results[-1]['u_t'] = ranker.model.u_t.tolist()
      results[-1]['g_t'] = ranker.model.g_t.tolist()
    else:
      results[-1]['u_t'] = np.zeros(len(ranker.model.weights[:, 0].T)).tolist()
      results[-1]['g_t'] = np.zeros(len(ranker.model.weights[:, 0].T)).tolist()
    results[-1]['w_t'] = ranker.model.weights[:, 0].T.tolist()
    conine_wstar = cosine(self.wstar2, ranker.model.weights[:, 0].T)
    results[-1]['cosine_w'] = conine_wstar
    if iteration%100 ==0:
      print iteration, conine_wstar
    results[-1]['noise_norm'] = ranker.model.noise_norm
    results[-1]['noise_norm_cum'] = ranker.model.noise_norm_cum


  def run(self, ranker, output_key):
    starttime = time.time()

    ranker.setup(train_features = self.datafold.train_feature_matrix,
                 train_query_ranges = self.datafold.train_doclist_ranges)
    # Added for ranker to access label directly
    ranker._train_label = self.datafold.train_label_vector

    run_results = []
    impressions = 0
    for impressions in range(self.n_impressions):
      ranking_i, train_ranking = self.sample_and_rank(ranker) #, impressions
      ranking_labels = self.datafold.train_query_labels(ranking_i)
      # stop_index temporarily added by sak2km
      clicks, stop_index = self.click_model.generate_clicks(train_ranking, ranking_labels)
      self.timestep_evaluate(run_results, impressions, ranker,
                             ranking_i, train_ranking, ranking_labels)

      ranker.process_clicks(clicks, stop_index, self.n_impressions)

      # Added temporarily to record gradient info
      if ranker.model.noise_norm is not None: 
        self.record_gradient(run_results, impressions, ranker)

    # evaluate after final iteration
    ranking_i, train_ranking = self.sample_and_rank(ranker) # , impressions
    ranking_labels =  self.datafold.train_query_labels(ranking_i)
    impressions += 1
    self.timestep_evaluate(run_results, impressions, ranker,
                           ranking_i, train_ranking, ranking_labels)

    ranker.clean()

    self.run_details['runtime'] = time.time() - starttime

    output = {'run_details': self.run_details,
              'run_results': run_results}

    self.output_queue.put((output_key, output))