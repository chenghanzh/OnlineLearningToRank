# Authored by Eric McCord-Snook
# January 23, 2019
import sys
import numpy as np
import pdb

def main():
    args = sys.argv
    print("Reading input file", args[1], "...")
    lines = [line.strip() for line in open(args[1])]
    print("Calculating variances...")
    qids_and_variances = []
    current_qid = None
    query_count = 0
    feature_matrix = []
    for line in lines:
        tokens = line.split(" ")
        if tokens[1] == current_qid or current_qid == None:
            current_qid = tokens[1]
            feature_vector = np.zeros(701)
            for token in tokens[2:]: # 48 added for MQ. depends on dataset format
                f_id, f_val = token.split(":")
                feature_vector[int(f_id)] = float(f_val)
            feature_matrix.append(feature_vector)
        else:
            net_var_for_query = np.sum(np.var(np.mat(feature_matrix), axis=0))
            qids_and_variances.append((query_count, net_var_for_query))
            # qids_and_variances.append((int(current_qid.split(":")[1]), net_var_for_query))
            print("Finished", current_qid)
            current_qid = tokens[1]
            feature_matrix = []
            query_count += 1
    qids_and_variances.sort(key=lambda tup: tup[1])

    with open("Webscope_Set2_qid_inc_var.txt", "w") as outfile:
        for tup in qids_and_variances:
            outfile.write(str(tup[0]) + "\n")

if __name__ == '__main__':
    main()
