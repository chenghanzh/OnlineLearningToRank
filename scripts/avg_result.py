import sys
import sys
import numpy as np
import os
import traceback
import json

import pdb
def create_folders(filename):
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

def click_model_name(full_name):
  return str(full_name[:full_name.rfind('_')])

def average_results(output_path):
  with open(output_path, 'r') as f:
    sim_args = json.loads(f.readline())
    first_run = json.loads(f.readline())
    run_details = first_run['run_details']

    cur_click_model = click_model_name(
      run_details['click model'])
    runtimes = {
        cur_click_model: [float(run_details['runtime'])],
      }

    all_ind = {}
    first_val = {}
    for event in first_run['run_results']:
      iteration = event['iteration']
      for name, val in event.items():
        if name == 'iteration':
          continue
        if name not in all_ind:
          all_ind[name] = []
          first_val[name] = []
        all_ind[name].append(iteration)
        first_val[name].append(val)

    all_val = {}
    for name in all_ind:
      all_ind[name] = np.array(all_ind[name],
                               dtype=np.int32)
      all_val[name] = {
          cur_click_model: [np.array(first_val[name],
                                     dtype=float)]
        }

    for line in f:
      events = json.loads(line)

      run_details = events['run_details']
      cur_click_model = click_model_name(
        run_details['click model'])
      if cur_click_model not in runtimes:
        runtimes[cur_click_model] = []

      runtimes[cur_click_model].append(
        float(run_details['runtime']))

      cur_i = {}
      cur_val = {}
      for name, val in all_ind.items():
        cur_i[name] = 0
        if '_t' in name: # Add array of vectors: u_t, g_t, w_t for each iteration
          cur_val[name] = np.zeros((len(val),len(events['run_results'][0][name])))
        else:
          cur_val[name] = np.zeros(val.shape)
        if cur_click_model not in all_val[name]:
          all_val[name][cur_click_model] = []
        all_val[name][cur_click_model].append(cur_val[name])

      for event in events['run_results']:
        iteration = event['iteration']
        for name, val in event.items():
          if name != 'iteration':
            c_i = cur_i[name]
            assert all_ind[name][c_i] == iteration
            cur_val[name][c_i] = val
            cur_i[name] += 1

      for name, val in all_ind.items():
        if name != 'iteration':
          assert cur_i[name] == val.shape[0]

  average_runtimes = {}
  for click_model, values in runtimes.items():
    average_runtimes[click_model] = np.mean(values).tolist()

  results = {}
  for name, cur_ind in all_ind.items():
    cur_results = {
        'indices': cur_ind.tolist()
      }
    results[name] = cur_results
    for click_model, lists in all_val[name].items():
      stacked = np.stack(lists)
      cm_mean = np.mean(stacked, axis=0)
      cm_std = np.std(stacked, axis=0)
      cur_results[click_model] = {
          'mean': cm_mean.tolist(),
          'std': cm_std.tolist(),          
        }

  output = {
    'simulation_arguments': sim_args,
    'runtimes': average_runtimes,
    'results': results
  }

  return output

def main():
  # For history
  # directory = "../average_rivanna_1_27_full_hist_1ds/"
  directory = "../average_rivanna_1_27_full_intp_1ds/"
  # for i in [5,10,20,30,50,70,100]:
  for i in [7,9]:
    name = str(i) + "_MGD_DSP.out"
    full_output_path = directory + name
    print "opening %s" % full_output_path
    output = average_results(full_output_path)
    avg_output_path = directory + "test_intp_1ds/" + name

  # For Interpolation

    create_folders(avg_output_path)
    with open(avg_output_path, 'w') as w:
      w.write(json.dumps(output))

if __name__ == '__main__':
    main()