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
    file_no = 0
    for output_file in args.output_files:
        file_data = np.zeros((5,5000))
        # Each file contains 15 lines (3 cm, 5 folds)
        # need to average 5 folds and plot dbgd/mgd with/without noise together
        with open(output_file) as f:
            line_no = 0
            for line in f:
                if line_no == 0:
                    line_no += 1
                    continue
                line_obj = json.loads(line)
                plot_title = plot_title + '_' + line_obj["run_details"]["click model"][:3]
                run_results = line_obj["run_results"]
                for it in range(0, len(run_results)-1):
                    it_obj = run_results[it]
                    # file_data[line_no-1][it] = it_obj["noise_norm"]
                    file_data[line_no-1][it] = it_obj["cumulative-display"]
                line_no += 1
        avg_data = np.average(file_data, axis=0)
        plot_data.append({"filename": output_file[output_file.rfind('/')+1:][:-4], "x": np.arange(5000), "y": avg_data})
        file_no += 1

    # Plotting data
    plt.title("MQ2007 per")
    plt.xlabel("Iteration")
    # plt.ylabel("Cumulative Noise L2 Norm")
    plt.ylabel("Online NDCG")
    plt.xlim((-5, 5005))
    for file_data in plot_data:
        plot(file_data["filename"], file_data["x"], file_data["y"])
    plt.legend(loc='upper left')
    # plt.savefig('graphs/output/differential_privacy/noise_norm/MQ2007Hybrid.png')
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
