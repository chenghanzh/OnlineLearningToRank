import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import csv

# with open("cand_cum.csv", 'rU') as data:
# with open("intp.csv", 'rU') as data:
with open("hist/k_off.csv", 'rU') as data:

    csvreader = csv.reader(data)
    k=4
    cand, per, inf, nav = zip(*csvreader)
    cand = [0,3,5,7]

    cand = [int(cand[i]) for i in range(k)]
    per = [float(per[i]) for i in range(k)]
    nav = [float(nav[i]) for i in range(k)]
    inf = [float(inf[i]) for i in range(k)]

plt.plot(cand, per, 'D-',linewidth= 1, label='per', markevery=1)
plt.plot(cand, nav,  '^-',linewidth = 1, label='nav', markevery=1)
plt.plot(cand, inf, 'o-',linewidth= 1, label='inf', markevery=1)

# plt.xlabel('Interpolation parameter', fontsize = 14, fontweight='bold')
# plt.xlabel('Historical Feature Vector Queuesize', fontsize = 14, fontweight='bold')plt.xlabel('Num of Doc. Below Last Click', fontsize = 14, fontweight='bold')
plt.xlabel('Num of Doc. Below Last Click', fontsize = 14, fontweight='bold')

plt.ylabel('Cumulative NDCG', fontsize = 14, fontweight='bold')

plt.legend(ncol = 2, prop={'size':14})

axes = plt.gca()
axes.set_xlim([-.5, 8]) # For intp
# axes.set_xlim([-10, 120]) # For hist
axes.set_ylim([.3,.43])
# axes.set_ylim([65,72])

matplotlib.rcParams.update({'font.size': 20})

plt.show()
