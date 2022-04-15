#%%
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from collections import Counter
import json
from scipy.stats import mode

sns.set_context('talk')

base_path = pathlib.Path(__file__).absolute().parents[0] / 'datasets'
print(pathlib.Path(__file__))

# data_name = "covtype"
data_name = "Rangiora_test2"

dataset_path = base_path / f"{data_name}.csv"
full_context_path = base_path / f"{data_name}_full_context.csv"

#%%
data_has_mask_column = True
data_df = pd.read_csv(dataset_path, header=None)
if data_has_mask_column:
    data_df = data_df.drop(data_df.columns[-2], axis=1)
full_context_df = pd.read_csv(full_context_path)
print(data_df.shape[0], full_context_df.shape[0])

full_df = pd.merge(data_df, full_context_df, left_index=True, right_index=True)
full_df.head()

#%%
full_context_df.columns

#%%
# context_columns = ["WS_4"]
# context_columns = ["WS_4", "WD_4"]
# context_columns = ["WS_4", "WD_4"]
context_columns = ["day"]
c_col = full_df[context_columns[0]]
# n_factors = 3
n_factors = 2

if len(c_col.unique()) > n_factors:
    c_col = pd.qcut(c_col, n_factors, labels=False, duplicates='drop')
    print(c_col)
context_column = c_col.astype(str)
for col in context_columns[1:]:
    c_col = full_df[col]
    if len(c_col.unique()) > n_factors:
        c_col = pd.qcut(c_col, n_factors, labels=False, duplicates='drop')
    context_column = context_column + '-' + c_col.astype(str)
print(context_column)
context_df = full_df.copy()
context_df['context'] = context_column
context_df = context_df[[*context_df.columns[:-2], context_df.columns[-1], context_df.columns[-2]]]
context_df = context_df.drop(full_context_df.columns, axis=1)
context_df.head()

unique_contexts = list(context_column.unique())
context_map = {k:unique_contexts.index(k) for k in unique_contexts}
context_df['context'] = context_df['context'].replace(context_map)
use_median = True
if use_median:
    # context_df['context'] = context_df['context'].rolling(50).median().fillna(method='bfill').astype('int')
    context_df['context'] = context_df['context'].rolling(50).apply(lambda x: mode(x)[0]).fillna(method='bfill').astype('int')

context_df.head()


#%%
def ts_to_ranges(ts, start=0):
    ranges = []
    prev_val = None
    ts_vals = ts.values if hasattr(ts, 'values') else ts
    for i, v in enumerate(ts_vals, start=start):
        if np.isnan(v):
            continue
        if prev_val != v:
            if prev_val != None:
                ranges[-1][1] = i-1
            ranges.append([i, None, int(v)])
        prev_val = v
    ranges[-1][1] = ts.shape[0] + start if hasattr(ts, 'shape') else len(ts) + start
    # print(ranges)
    return ranges
gt_model_ranges = ts_to_ranges(context_df['context'].values)
gt_colors = plt.cm.tab10.colors
gt_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
unique_gt_names = []
for _,_,gt_name in gt_model_ranges:
    if gt_name not in unique_gt_names:
        unique_gt_names.append(gt_name)

fig, ax_b = plt.subplots(figsize=(50, 5))
for rs, re, v in gt_model_ranges:
    # if str(v) in labels:
        # color = handles[labels.index(str(v))].get_color()
    color = gt_colors[unique_gt_names.index(v)]
    context_label = gt_labels[unique_gt_names.index(v)]
    # alpha_val = 1 if context_label in focus else 0.25
    alpha_val = 1
    y_val = -0.8
    ax_b.hlines(y = y_val, xmin = rs, xmax = re, colors = [color], alpha=alpha_val)
    mid_point = (rs+re)/2
    gt_label = gt_labels[unique_gt_names.index(v)]
    ax_b.annotate(gt_label,
    xy=(mid_point, y_val-0.1),
    xycoords='data',
    bbox={'boxstyle':'circle',
        'fc':'white',
        'ec': color,
        'alpha':alpha_val},
    fontsize='small',
    ha='center',
    va='center', 
    alpha=alpha_val
    )
ax_b.annotate("Ground Truth Contexts: ",
xy=(0, y_val),
xycoords='data',
ha='right',
va='center'
)
plt.savefig('test.pdf')
# %%
no_context_df = context_df.drop(['context'], axis=1)
dataset_name = f"{data_name}-{'_'.join(context_columns)}"
save_dir = base_path / dataset_name
save_dir.mkdir(parents=True, exist_ok=True)
total = 0
for c_name in unique_contexts:
    one_context_df = no_context_df[context_df['context'] == context_map[c_name]]
    print(one_context_df.head())
    save_path = save_dir / f"{dataset_name}_{c_name}.csv"
    one_context_df.to_csv(save_path)
    total += one_context_df.shape[0]

print(total)
# %%
no_context_df = context_df.drop(['context'], axis=1)
dataset_name = f"{data_name}-{'_'.join(context_columns)}"
save_dir = base_path / f"{dataset_name}-nonordered"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / f"{dataset_name}_0.csv"
no_context_df.to_csv(save_path)
context_df = context_df['context']
save_path = save_dir / f"context.csv"
context_df.to_csv(save_path, index=False)


# %%
full_df.head()
# %%
sns.lineplot(full_df, x=full_df.index, y='elevation')