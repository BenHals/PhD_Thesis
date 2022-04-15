#%%
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

sns.set_context('talk')

#%%
# experiment_name =  "select_mem_management_l"
# experiment_name =  "select_mem_management"
experiment_name =  "select_mem_l2"
base_path = pathlib.Path(__file__).absolute().parents[1] / 'output' / experiment_name
results_files = list(base_path.rglob('results*'))
results = []
for rp in results_files:
    result = json.load(rp.open('r'))
    results.append(result)

df = pd.DataFrame(results)

#%%
# data_names = ['cmc', 'Arabic', 'RTREESAMPLE_HARD', 'WINDSIM', 'AQSex', 'AQTemp', 'STAGGERS']
data_names = ['RTREE', 'RTREESAMPLE_HARD', 'WINDSIM', 'RBFMed']
classifier = 'ficsum'
# classifier = 'cc'
# classifier = 'advantage'
# classifier = 'airstream'
# data_names = ['cmc', 'Arabic']
fig, axs = plt.subplots(1, len(data_names), figsize=(len(data_names) * 10, 10), sharey=False)
if len(data_names) == 1:
    axs = [axs]
for i, dn in enumerate(data_names):
    dn_df = df[df['data_name'] == dn]
    dn_df = dn_df[dn_df['classifier'] == classifier]
    ax = axs[i]
    sns.lineplot(data=dn_df, x='repository_max', y='kappa', hue='valuation_policy', hue_order=['rA', 'LRU'], ax=ax, ci=None)
    ax.set_title(dn)
plt.show()

# %%
df['classifier'].unique()