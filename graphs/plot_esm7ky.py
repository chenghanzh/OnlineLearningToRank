# Authored by Eric McCord-Snook
# October 19, 2019

import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import datetime
import pdb

def plot(label, x, y, color=None, ax=None):
    ax = ax if ax is not None else plt.gca()
    base_line, = ax.plot(x, y, label=label)

def main():
    # Parser setup
    parser = argparse.ArgumentParser(description="parse args for graphs")
    parser.add_argument('plot_name', type=str, help='Name to save plots under.')
    parser.add_argument('output_files', type=str, help='Output files to be parsed.', nargs='+')
    args = parser.parse_args()

    # Parsing files
    plot_title = args.plot_name
    plot_data = []
    for output_file in args.output_files:
        file_data = np.zeros((5,10000))
        with open(output_file) as f:
            line_no = 0
            for line in f:
                if line_no == 0:
                    line_no += 1
                    continue
                line_obj = json.loads(line)
                click_model = line_obj["run_details"]["click model"][:3]
                if click_model != "inf": # generating graphs for inf, nav, per, separately for convenience
                    continue
                run_results = line_obj["run_results"]
                for it in range(0, len(run_results)-1):
                    it_obj = run_results[it]
                    file_data[line_no-1][it] = it_obj["cumulative-display"]
                line_no += 1
        avg_data = np.average(file_data, axis=0)
        plot_data.append({"filename": output_file[output_file.rfind('/')+1:][:-4], "x": np.arange(10000), "y": avg_data})

    # Plotting data
    plt.title(plot_title)
    plt.xlabel("Iteration")
    plt.ylabel("Online NDCG")
    plt.xlim((0, 10000))
    for file_data in plot_data:
        plot(file_data["filename"], file_data["x"], file_data["y"])
    plt.legend(loc='upper left')
    plt.savefig('graphs/output/differential_privacy/int/' + plot_title + '.png')
    plt.clf()


####################

    # for dataset in dataset_arr:
    #     for ccm in ccm_arr:
    #         plt.title(dataset + "-" + ccm)
    #         plt.xlabel("Iteration")
    #         plt.ylabel("NDCG")
    #         for alg in algorithms:
    #             summary_path = 'config/esm7ky/' + alg + '/v000/output/' + ccm + '/' + dataset + '/outdir/summary.txt'
    #             data = read_csv(summary_path)
    #             plot_with_error(alg, data[:,0], data[:,1], data[:,2])
    #         plt.xlim((-5,len(data)-1))
    #         plt.legend(loc='lower right')
    #         plt.savefig('plots/' + experiment_name + '/' + dataset + '-' + ccm + '.png')
    #         plt.clf()

if __name__ == '__main__':
    main()
