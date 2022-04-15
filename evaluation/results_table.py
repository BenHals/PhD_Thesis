#%%
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns

from eval_utils import get_classifier_matched_results
# get_ipython().run_line_magic('matplotlib', 'ipympl')


def ts_to_ranges(ts):
    ranges = []
    prev_val = None
    for i, v in enumerate(ts.values):
        if np.isnan(v):
            continue
        if prev_val != v:
            if prev_val != None:
                ranges[-1][1] = i-1
            ranges.append([i, None, int(v)])
        prev_val = v
    ranges[-1][1] = ts.shape[0]
    print(ranges)
    return ranges

def plot_probabilities(prob_column, save_file=False, run_individual=False, smoothing=True):
    probs = df[prob_column].str.split(';', expand=True)
    active_model = df['active_model']
    gt_model = df['ground_truth_concept']
    active_model_ranges = ts_to_ranges(active_model)
    gt_model_ranges = ts_to_ranges(gt_model)
    del_cols = []
    val_nas = {}
    for c in probs.columns:

        col_id = probs[c].str.rsplit(':').str[0].astype('float').unique()[-1]
        print(probs[c].str.rsplit(':').str[0].astype('float').unique())
        probs[c] = probs[c].str.rsplit(':').str[-1].astype('float')
        probs[f"v{col_id}"] = probs[c]
        val_nas[c] = probs[c].isna().sum()
        # if probs[c].isna().sum() > probs.shape[0] * 0.5:
        #     del_cols.append(f"v{col_id}")
        del_cols.append(c)
    print(val_nas)
    probs = probs.drop(del_cols, axis=1)
    del_cols = []
    for c in probs.columns:
        probs[int(float(c[1:]))] = probs[c]
        del_cols.append(c)
    probs = probs.drop(del_cols, axis=1)
    probs['ts'] = probs.index
    m_df = pd.melt(probs.iloc[::10, :], id_vars='ts')
    m_df['variable'] = m_df['variable'].astype('category')
    m_df['value'] = m_df['value'].replace({-1:0})
    # m_df['value'] = m_df['value'].rolling(window=int(probs.shape[0]*0.001)).mean()
    if smoothing:
        m_df['value'] = m_df['value'].rolling(window=7).mean()

    sns.lineplot(data=m_df[m_df['variable'] != '-1'], x='ts', y='value', hue='variable')
    if save_file:
        plt.savefig(f"explore_probabilities/{experiment_name}-{data_name}.pdf")
    
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # colorsplt.rcParams["axes.prop_cycle"].by_key()["color"]
    print(labels)
    for rs, re, v in active_model_ranges:
        if str(v) in labels:
            color = handles[labels.index(str(v))].get_color()
            plt.hlines(y = -0.05, xmin = rs, xmax = re, colors = [color])
        else:
            plt.hlines(y = -0.01, xmin = rs, xmax = re, colors=["red"])
    for rs, re, v in gt_model_ranges:
        if str(v) in labels:
            color = handles[labels.index(str(v))].get_color()
            plt.hlines(y = -0.1, xmin = rs, xmax = re, colors = [color])
        else:
            plt.hlines(y = -0.1, xmin = rs, xmax = re, colors=["red"])
    plt.xlim((0, probs.shape[0]))
    plt.title(f"{data_name}-{prob_column}")
    plt.legend().remove()
    plt.show()
    if not run_individual:
        return
    for c in m_df['variable'].unique():
        sns.lineplot(data=m_df[m_df['variable'] == c], x='ts', y='value', hue='variable')
        if save_file:
            plt.savefig(f"explore_probabilities/{experiment_name}-{data_name}-{c}.pdf")
        plt.show()

# %%

experiment_name = "cndpm_comparison"

base_path = pathlib.Path(__file__).absolute().parents[1] / 'output' / experiment_name
# base_path = pathlib.Path("//TOMSK/shared_results") / experiment_name
# base_path = pathlib.Path(r"C:\Users\Ben\Documents\shared_results") / experiment_name

experiment_parameters = ["data_name", "classifier"]
experiment_IVs = ['overall_accuracy', 'GT_mean_f1', 'kappa', 'kappa_m', 'kappa_t', "overall_time"]
data_name = ''


# # seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# # seeds = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# # seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
# # seeds = [1, 2, 3, 4, 5]
# # seeds = [6, 7, 8, 9, 10]
# # seeds = [11, 12, 13, 14, 15]
# # seeds = [16, 17, 18, 19, 20]
# # seeds = [2]

restrictions = []
# restrictions = [

#     ("merge_threshold", [0.9])
# ]
selected, groups, grouped = get_classifier_matched_results(base_path, experiment_name, parameters_of_interest=experiment_parameters, IVs_of_interest=experiment_IVs, seeds=seeds, restrictions=restrictions)
pd.set_option('display.max_rows', None)
grouped = grouped.reset_index()

grouped

# %%
# *****************************************
# Table with H result type and V classifier
# *****************************************
grouped.to_latex(f"{experiment_name}_table.tex", index=False, float_format="{:0.2%}".format)
# %%
grouped[grouped['data_name'].isin(['AQSex', 'AQTemp', 'Arabic', 'cmc', 'RTREESAMPLE_HARD', 'STAGGERS'])].to_latex(f"{data_name}_table.tex", index=False, float_format="{:0.2%}".format)

# %%
def str_stdev(x):
    return f"{np.mean(x):.2%} ({np.std(x):.2%})"

g = groups.aggregate(str_stdev)[experiment_IVs]
g['size'] = groups.size()
# %%
g.reset_index().to_latex(f"{experiment_name}_table.tex", index=False, float_format="{:0.2%}".format)



#%%
plt.rcParams["font.family"] = "Times New Roman"
data_name_map = {
    "AQSex": "AQSex",
    "AQTemp": "AQTemp",
    "RTREESAMPLE_HARD": "TREE-H",
    "Arabic": "Arabic",
    "cmc": "CMC",
    "RTREESAMPLE_Diff": "TREE",
    "qg": "QG",
    "RBFMed": "RBF",
    "UCI-Wine": "Wine",
    "STAGGERS": "STAGGER",
    "WINDSIM": "WINDSIM"
}

# classifier_name_map = {
#     "CC": "PhDCode",
#     "CC_basicprior": "$S_p$",
#     "CC_MAP": "$S_{MAP}$",
#     "CC_nomerge": "$S_{m}$",
# }
# classifier_name_map = {
#     "CC": "PhDCode",
#     "CC_basicprior": "$S_p$",
#     "CC_MAP": "$S_{MAP}$",
#     "CC_nomerge": "$S_{m}$",
#     "arf": "ARF",
#     "ficsum": "FICSUM",
#     "lower_bound": "lower",
#     "upper_bound": "upper",
#     "dwm": "DWM",
#     "cpf": "CPF",
#     "rcd": "RCD",
#     "dynse": "DYNSE",
# }
classifier_name_map = {
    "cc": "SELeCT",
    "cc_cndpm_base": "$S_{nn}$",
    "cndpm": "CN-DPM",
    "arf": "ARF",
    "ficsum": "FICSUM",
    "lower_bound": "lower",
    "upper_bound": "upper",
    "upper_bound_cndpm_base": "upper$_{nn}$",
    "dwm": "DWM",
    "cpf": "CPF",
    "rcd": "RCD",
    "dynse": "DYNSE",
}

column_name_map = {
    "GT_mean_f1": "C-F1",
    "drift_width": "Width",
    "kappa": "$\kappa$",
    "noise": "Noise",
    "TMnoise": "Transition Noise",
    "overall_time": "Runtime (s)"
}



# parameter = "Width"
# parameter_values = [0, 100, 500, 2500]
# parameter = "Noise"
# parameter = "Transition Noise"
parameter_values = [0, 0.05, 0.1, 0.25]
parameter = None
performance = ["$\kappa$", "C-F1"]
# performance = ["$\kappa$", "C-F1", "Runtime (m)"]
# data_sets = ['AQSex', 'AQTemp', 'TREE-H']
data_sets = ['AQSex', 'AQTemp', 'Arabic', 'CMC', 'TREE-H', 'STAGGER', "WINDSIM"]

def mean_stdev(X):
    return f"{np.mean(X):.2f}({np.std(X):.2f})"

def map_list(X):
    return [x if x not in column_name_map else column_name_map[x] for x in X]

t = selected.copy()
# t = t[t['noise'] == 0.0]
t['Runtime (m)'] = t['overall_time'] / 60
t.columns = map_list(t.columns)
t['data_name'] = t['data_name'].map(data_name_map)
t= t.dropna()
t['classifier'] = t['classifier'].map(classifier_name_map)
# t['data_name']

t = t[t['data_name'].isin(data_sets)]
# t = t[t['classifier'].isin(['PhDCode', "$S_p$", "$S_{MAP}$", "$S_{m}$", 'FICSUM', 'lower', 'upper', 'DWM', 'CPF', 'RCD', 'DYNSE'])]
if parameter:
    t = t[t[parameter].isin(parameter_values)]
    tg = t.groupby(["classifier", parameter, "data_name"]).agg(mean_stdev)[performance].unstack('data_name')
else:
    tg = t.groupby(["classifier", "data_name"]).agg(mean_stdev)[performance].unstack('data_name')
# t.columns = pd.MultiIndex.from_tuples([(d, p) for d in data_sets for p in performance])
tg.columns = tg.columns.swaplevel(0, 1)
tg = tg.reindex(data_sets, axis=1, level=0)
# tg = tg.transpose()[['FICSUM', 'RCD',  'CPF', 'DWM', 'DYNSE','PhDCode', 'lower', 'upper', "$S_p$", "$S_{MAP}$", "$S_{m}$" ]]
# tg = tg.transpose()[['PhDCode', "$S_p$", "$S_{MAP}$", "$S_{m}$" ]]
tg = tg.transpose()[['SELeCT', "$S_{nn}$", "CN-DPM", "upper", "upper$_{nn}$" ]]
tg = tg.swaplevel()
tg = tg.sort_index()
# tg.to_latex(f"{parameter}_datatable.txt", escape=False, sparsify=True, multirow=True)
tg.to_latex(f"{experiment_name}-{parameter}_datatable_ablation.txt", escape=False, sparsify=True, multirow=True)
tg
# %%
selected['data_name']
# %%
noise_df = selected.copy()
# %%
width_df = selected.copy()

#%%
tmnoise_df = selected.copy()

#%%
tmnoise_df['param'] = 'Transition Noise'
tmnoise_df['param_value'] = tmnoise_df['TMnoise']
width_df['param'] = 'Drift Width'
width_df['param_value'] = width_df['drift_width']
noise_df['param'] = 'Noise'
noise_df['param_value'] = noise_df['noise']

#%%
width_df_s = width_df[width_df['drift_width'].isin([0, 500, 2500])]
noise_df_s = noise_df[noise_df['noise'].isin([0, 0.10, 0.25])]
tmnoise_df_s = tmnoise_df[tmnoise_df['TMnoise'].isin([0, 0.10, 0.25])]
full_df = pd.concat([width_df_s, noise_df_s, tmnoise_df_s])
full_df

#%%
def mean_stdev(X):
    return f"{np.mean(X):.2f}({np.std(X):.2f})"
# g = full_df.groupby(['data_name', 'classifier', 'param', 'param_value']).mean()[['kappa', 'GT_mean_f1']]
g = full_df.groupby(['data_name', 'classifier', 'param', 'param_value']).agg(mean_stdev)[['kappa', 'GT_mean_f1']]
# full_df.groupby(['data_name', 'classifier', 'param', 'param_value']).mean().columns
#%%
gt = g.unstack(['param', 'param_value'])
# gt.columns = gt.columns.swaplevel(0, 2)
# gt.columns = gt.columns.swaplevel(0, 1)
gt = gt.stack(0)
#%%
gt.index = gt.index.swaplevel(0, 2)
gt.index = gt.index.swaplevel(1, 2)
# gt.index = gt.index.sortlevel(0)
#%%
gt = gt.sort_index(0, ascending=False)
# %%
gt = gt.loc[( slice(None), ['AQSex', 'AQTemp', 'RTREESAMPLE_HARD'], ['CC', 'ficsum', 'lower_bound', 'upper_bound']), :]
# %%
gt.to_latex('param_table.txt', sparsify=True, multirow=True)
# %%

table_path = pathlib.Path('param_table.txt')
table_str = table_path.open().read()
table_str = table_str.replace(r'RTREESAMPLE\_HARD', 'TREE')
table_str = table_str.replace(r'upper\_bound', 'Upper')
table_str = table_str.replace(r'lower\_bound', 'Lower')
table_str = table_str.replace(r'GT\_mean\_f1', 'C-F1')
table_path.open('w').write(table_str)

# %%
