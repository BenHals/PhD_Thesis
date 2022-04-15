import numpy as np
import pandas as pd
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
import skmultiflow.core
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import glob
sys.path.append('..\GenerateDatastreamFiles\DriftStream\SyntheticWoodsmoke')
sys.path.append('..\DatasetGen\DriftStream\SyntheticWoodsmoke')
from windSimStream import WindSimGenerator
from windSimStream import PollutionSource
import pickle
import math
import matplotlib.animation as animation
from matplotlib.patches import Circle
from collections import deque


name = f"ff_res_*U5.pickle"

fns = glob.glob(name)
print(fns)

fig, axs = plt.subplots(2, 2, sharey = 'row', dpi=120)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
ax2 = axs[1, 0].twinx()
ax3 = axs[1, 1].twinx()
secondary_axs = (ax2, ax3)
ax2.get_shared_y_axes().join(ax2, ax3)
# plt.rc('xtick',labelsize=5)
# plt.rc('ytick',labelsize=5)

for ci,fn in enumerate(fns):
    with open(fn, 'rb') as f:
        result_obj = pickle.load(f)
    results = result_obj['r']
    reused = result_obj['p']
    retrained = result_obj['l']
    z = result_obj['z']

    name_st = fn[len(name.split('*')[0]):-1 * len(name.split('*')[1])]
    print(name_st)
    # print(results)
    print(f"{name_st} num errors reuse: {results[-1][4]}")
    print(f"{name_st} num errors retrain: {results[-1][3]}")
    # axs[1, ci].plot([x[0] for x in results], [x[1] for x in results], label = "retrain", lw = 1, color = 'tab:blue', ls = 'dashed')
    axs[1, ci].plot([x[0] for x in results], [x[1] for x in results], label = "Retrain Acc", lw = 1, color = 'tab:blue')
    axs[1, ci].plot([x[0] for x in results], [x[2] for x in results], label = 'Reuse Acc', lw = 1, color = 'tab:blue', marker = 'x', markevery=0.1, markersize=4)
    axs[1, ci].fill_between([x[0] for x in results], [x[1] for x in results], [x[2] for x in results], alpha = 0.3, color = 'tab:green', label = "Advantage")
    # secondary_axs[ci].plot([x[0] for x in results], [x[3] for x in results], label = "retrain", lw = 1, color = 'tab:red', ls = 'dashed')
    secondary_axs[ci].plot([x[0] for x in results], [x[3] for x in results], label = "Retrain #Err", lw = 1, color = 'tab:red', ls = 'dashed')
    secondary_axs[ci].plot([x[0] for x in results], [x[4] for x in results], label = 'Reuse #Err', lw = 1, color = 'tab:red', ls = 'dashed', marker = 'x', markevery=0.1, markersize=4)
    secondary_axs[ci].tick_params(axis='y', labelcolor='tab:red', labelsize = 7)
    secondary_axs[ci].tick_params(axis='x', labelsize = 7)
    axs[1, ci].tick_params(axis='y', labelcolor='tab:blue', labelsize=7)
    axs[1, ci].tick_params(axis='x', labelsize=7)
    # secondary_axs[ci].fill_between([x[0] for x in results], [x[3] for x in results], [x[4] for x in results], alpha = 0.3, color = 'green', label = "Advantage")
    
    axs[1, ci].set_xlabel("Observation")
    if ci == 0:
        axs[1, ci].set_ylabel("Recent Accuracy", color = 'tab:blue')
        secondary_axs[ci].set_yticks([])
    if ci == 1:
        # axs[1, ci].legend(loc = 'center right')
        h1, l1 = axs[1, ci].get_legend_handles_labels()
        h2, l2 = secondary_axs[ci].get_legend_handles_labels()
        axs[1, ci].legend(h1+h2, l1+l2, loc='center right', prop={'size': 8})
        secondary_axs[ci].set_ylabel('# Errors', color='tab:red')
        # axs[1, 1].set_yticks([])
    axs[0, ci].set_title(name_st)
    axs[0, ci].imshow(z, cmap='gray', vmin = 0, vmax = 255)
    axs[0, ci].set_axis_off()
plt.savefig(f"ff.png", dpi=200, bbox_inches='tight')
plt.savefig(f"ffH.png", dpi=500, bbox_inches='tight')
plt.savefig(f"ff.pdf", bbox_inches='tight')
plt.show()