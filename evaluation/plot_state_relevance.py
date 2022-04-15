#%%
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
plt.rcParams["font.family"] = "Times New Roman"
sns.set_context('talk')

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
    # print(ranges)
    return ranges

def plot_probabilities(df, prob_column, focus, experiment_name="default", data_name="default", save_file=False, save_name="", run_individual=False, smoothing=True, plot_state_relevance=True, plot_accuracy=True):
    
    active_model = df['active_model']
    merge_model = df['merge_model'] if 'merge_model' in df.columns else None
    repair_model = df['repair_model'] if 'repair_model' in df.columns else None
    gt_model = df['ground_truth_concept']
    active_model_ranges = ts_to_ranges(active_model)
    merge_model_ranges = ts_to_ranges(merge_model) if merge_model is not None else None
    repair_model_ranges = ts_to_ranges(repair_model) if repair_model is not None else None
    gt_model_ranges = ts_to_ranges(gt_model)
    unique_gt_names = []
    for _,_,gt_name in gt_model_ranges:
        if gt_name not in unique_gt_names:
            unique_gt_names.append(gt_name)

    del_cols = []
    val_nas = {}

    gt_colors = plt.cm.tab10.colors
    gt_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    dms = df['deletions'].dropna().values
    deleted_models = []
    for dm in dms:
        try:
            deleted_models.append(int(float(dm)))
        except:
            ids = dm.split(';')
            # print(ids)
            for id in ids:
                if len(id) > 0:
                    deleted_models.append(int(float(id)))
    if plot_state_relevance:                
        probs = df[prob_column].str.split(';', expand=True)
        for c in probs.columns:
            unique_ids = probs[c].str.rsplit(':').str[0].astype('float').unique()
            col_ids = probs[c].str.rsplit(':').str[0].astype('float')
            prob_vals = probs[c].str.rsplit(':').str[-1].astype('float')
            for u_id in unique_ids:
                if np.isnan(u_id):
                    continue
                indexes = col_ids == u_id
                if f"v{u_id}" not in probs.columns:
                    probs[f"v{u_id}"] = np.nan
                probs.loc[indexes, f"v{u_id}"] = prob_vals.loc[indexes]
                if u_id in deleted_models:
                    del_cols.append(f"v{u_id}")
            probs[c] = probs[c].str.rsplit(':').str[-1].astype('float')
            val_nas[c] = probs[c].isna().sum()
            
            del_cols.append(c)
        # print(val_nas)
        probs = probs.drop(del_cols, axis=1)
        del_cols = []
        for c in probs.columns:
            probs[int(float(c[1:]))] = probs[c]
            del_cols.append(c)
        probs = probs.drop(del_cols, axis=1)
        probs['ts'] = probs.index
        # print(probs)
        m_df = pd.melt(probs.iloc[::10, :], id_vars='ts')
        m_df['variable'] = m_df['variable'].astype('category')
        m_df['value'] = m_df['value'].replace({-1:0})
        if smoothing:
            m_df['value'] = m_df['value'].rolling(window=7).mean()
    # print(m_df)
    fig, (ax, ax_b) = plt.subplots(nrows=2, figsize=(20, 10), sharex=True, gridspec_kw={'height_ratios':[0.7, 0.3]})

    active_state_CF1 = {}
    active_state_color = {}
    merge_state_CF1 = {}
    merge_state_color = {}
    repair_state_CF1 = {}
    repair_state_color = {}
    gt_timesteps = {}
    for gt in unique_gt_names:
        gt_ts = set(df[df['ground_truth_concept'] == gt]['example'].values)
        gt_timesteps[gt] = gt_ts

    for state_id in df['active_model'].unique():
        state_ts = set(df[df['active_model'] == state_id]['example'].values)
        if len(state_ts) == 0:
            continue
        state_CF1_per_gt = []
        for gt in unique_gt_names:
            gt_ts = gt_timesteps[gt]
            recall = len(state_ts.intersection(gt_ts)) / len(gt_ts)
            precision = len(state_ts.intersection(gt_ts)) / len(state_ts)
            CF1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
            state_CF1_per_gt.append((CF1, gt))
        max_CF1, best_gt = max(state_CF1_per_gt, key=lambda x: x[0])
        active_state_CF1[state_id] = (max_CF1, best_gt)
        color = gt_colors[best_gt]
        active_state_color[state_id] = color
        if plot_state_relevance:
            context_match = active_state_CF1[state_id][1]
            context_label = gt_labels[unique_gt_names.index(context_match)]
            alpha_val = 1 if context_label in focus else 0.25
            ax.plot(m_df[m_df['variable'] == state_id]['ts'].values, m_df[m_df['variable'] == state_id]['value'].values, c=color, label=str(state_id), alpha=alpha_val)    
    for state_id in df['merge_model'].unique():
        state_ts = set(df[df['merge_model'] == state_id]['example'].values)
        if len(state_ts) == 0:
            continue
        state_CF1_per_gt = []
        for gt in unique_gt_names:
            gt_ts = gt_timesteps[gt]
            recall = len(state_ts.intersection(gt_ts)) / len(gt_ts)
            precision = len(state_ts.intersection(gt_ts)) / len(state_ts)
            CF1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
            state_CF1_per_gt.append((CF1, gt))
        max_CF1, best_gt = max(state_CF1_per_gt, key=lambda x: x[0])
        merge_state_CF1[state_id] = (max_CF1, best_gt)
        color = gt_colors[best_gt]
        merge_state_color[state_id] = color
    for state_id in df['repair_model'].unique():
        state_ts = set(df[df['repair_model'] == state_id]['example'].values)
        if len(state_ts) == 0:
            continue
        state_CF1_per_gt = []
        for gt in unique_gt_names:
            gt_ts = gt_timesteps[gt]
            recall = len(state_ts.intersection(gt_ts)) / len(gt_ts)
            precision = len(state_ts.intersection(gt_ts)) / len(state_ts)
            CF1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
            state_CF1_per_gt.append((CF1, gt))
        max_CF1, best_gt = max(state_CF1_per_gt, key=lambda x: x[0])
        repair_state_CF1[state_id] = (max_CF1, best_gt)
        color = gt_colors[best_gt]
        repair_state_color[state_id] = color

    # sns.lineplot(data=m_df[m_df['variable'] != '-1'], x='ts', y='value', hue='variable', ax=ax, linewidth=0.5)
    # sns.scatterplot(data=m_df[m_df['variable'] != '-1'], x='ts', y='value', hue='variable', ax=ax, linewidth=0.5)
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    # print(labels)
    for rs, re, v in active_model_ranges:
        y_val = -0.2
        op = active_state_CF1[v][0]
        context_match = active_state_CF1[v][1]
        context_label = gt_labels[unique_gt_names.index(context_match)]
        alpha_val = 1 if context_label in focus else 0.25
        # if str(v) in labels:
            # color = handles[labels.index(str(v))].get_color()
        color = active_state_color[v]
        ax_b.hlines(y = y_val, xmin = rs, xmax = re, colors = [color], lw=10*op, alpha=alpha_val)
        # else:
        #     ax_b.hlines(y = y_val + 0.005, xmin = rs, xmax = re, colors=["red"], alpha=alpha_val)
        mid_point = (rs+re)/2
        ax_b.annotate(str(v),
        xy=(mid_point, y_val+0.05),
        xycoords='data',
        bbox={'boxstyle':'circle',
            'fc':'white',
            'ec': color,
            'alpha':alpha_val},
        fontsize='xx-small',
        ha='center',
        va='bottom',
        alpha=alpha_val
        )
    ax_b.annotate("Active States: ",
    xy=(0, y_val),
    xycoords='data',
    ha='right',
    va='center'
    )
    if merge_model_ranges is not None:
        for rs, re, v in merge_model_ranges:
            y_val = -0.5
            # if str(v) in labels:
                # color = handles[labels.index(str(v))].get_color()
            op = merge_state_CF1[v][0]
            context_match = active_state_CF1[v][1]
            context_label = gt_labels[unique_gt_names.index(context_match)]
            alpha_val = 1 if context_label in focus else 0.25
            color = merge_state_color[v]
            ax_b.hlines(y = y_val, xmin = rs, xmax = re, colors = [color], lw=10*op, alpha=alpha_val)
            # else:
            #     ax_b.hlines(y = y_val + 0.005, xmin = rs, xmax = re, colors=["red"])
            mid_point = (rs+re)/2
            ax_b.annotate(str(v),
            xy=(mid_point, y_val+0.05),
            xycoords='data',
            bbox={'boxstyle':'circle',
                'fc':'white',
                'ec': color,
                'alpha':alpha_val},
            fontsize='xx-small',
            ha='center',
            va='bottom', 
            alpha=alpha_val
            )
        ax_b.annotate("With Merging: ",
        xy=(0, y_val),
        xycoords='data',
        ha='right',
        va='center'
        )
    if repair_model_ranges is not None:
        for rs, re, v in repair_model_ranges:
            y_val = -0.8
            # if str(v) in labels:
                # color = handles[labels.index(str(v))].get_color()
            op = repair_state_CF1[v][0]
            context_match = active_state_CF1[v][1]
            context_label = gt_labels[unique_gt_names.index(context_match)]
            alpha_val = 1 if context_label in focus else 0.25
            color = repair_state_color[v]
            ax_b.hlines(y = y_val, xmin = rs, xmax = re, colors = [color], lw=10*op, alpha=alpha_val)
            # else:
            #     ax_b.hlines(y = y_val + 0.005, xmin = rs, xmax = re, colors=["red"])
            mid_point = (rs+re)/2
            ax_b.annotate(str(v),
            xy=(mid_point, y_val+0.05),
            xycoords='data',
            bbox={'boxstyle':'circle',
                'fc':'white',
                'ec': color,
                'alpha':alpha_val},
            fontsize='xx-small',
            ha='center',
            va='bottom', 
            alpha=alpha_val
            )
        ax_b.annotate("With Repair: ",
        xy=(0, y_val),
        xycoords='data',
        ha='right',
        va='center'
        )
    for rs, re, v in gt_model_ranges:
        # if str(v) in labels:
            # color = handles[labels.index(str(v))].get_color()
        color = gt_colors[unique_gt_names.index(v)]
        context_label = gt_labels[unique_gt_names.index(v)]
        alpha_val = 1 if context_label in focus else 0.25
        y_val = -1.1
        ax_b.hlines(y = y_val, xmin = rs, xmax = re, colors = [color], alpha=alpha_val)
        mid_point = (rs+re)/2
        gt_label = gt_labels[unique_gt_names.index(v)]
        ax_b.annotate(gt_label,
        xy=(mid_point, y_val),
        xycoords='data',
        bbox={'boxstyle':'circle',
            'fc':'white',
            'ec': color,
            'alpha':alpha_val},
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

    plt.xlim((0, df.shape[0]))
    # ax.set_title(f"{prob_column.replace('_', ' ')}")
    ax.get_legend().remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.xaxis.set_tick_params(which='both', labelbottom=True)
    # ax_b.axis('off')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.spines['bottom'].set_visible(False)
    ax_b.spines['left'].set_visible(False)
    ax_b.get_yaxis().set_ticks([])
    ax_b.set_xlabel("Number of Observations")
    if plot_state_relevance:
        ax.set_ylabel("State Relevance")
    else:
        ax.set_ylabel("Accuracy")
    ax_b.set_ylim((-1.2, -0.1))
    ax.set_ylim((0, 1))

    if plot_accuracy:
        ax_acc = ax.twinx()
        # selected = (df['ground_truth_concept'].isin(unique_gt_names))
        focus_as_index = map(lambda x: unique_gt_names[gt_labels.index(x)], focus)
        selected = (df['ground_truth_concept'].isin(focus_as_index))
        selected_example_num = df[selected]['example']
        rolling_acc = df[selected]['is_correct'].rolling(500).mean()
        sum_acc = df[selected]['is_correct'].cumsum()
        count = selected.cumsum()
        avg_acc = sum_acc / count[selected]
        # ax_acc.plot(df['example'].values, df['overall_accuracy'].values, c="red")
        ax_acc.plot(selected_example_num.values, rolling_acc.values, c="red")
        ax_acc.plot(selected_example_num.values, avg_acc.values, c="red", ls='--')
        ax_acc.set_ylabel('Accuracy', color="red")
        ax_acc.tick_params(axis='y', labelcolor="red")
        ax_acc.set_ylim((0, 1))

    if save_file:
        print(pathlib.Path.cwd())
        plt.savefig(f"chap8/{save_name}-{focus}-{plot_state_relevance}-{plot_accuracy}.pdf", facecolor='white', transparent=False, bbox_inches="tight")
    else:
        plt.show()
    if not run_individual:
        return
    for c in m_df['variable'].unique():
        df_variable = m_df[m_df['variable'] == c]

        sns.scatterplot(data=df_variable[df_variable['value'].diff() != 0], x='ts', y='value', hue='variable')
        # sns.lineplot(data=m_df[m_df['variable'] == c], x='ts', y='value', hue='variable')
        if save_file:
            plt.savefig(f"explore_probabilities/{experiment_name}-{data_name}-{c}.pdf")
        plt.show()

#%%
# experiment_name =  "t_advantage_SR"
experiment_name =  "t_reuse_benefit"
# data_name = 'Arabic'
# data_name = 'Arabic_advantage'
data_name = 'RTREESAMPLE_Diff'
base_path = pathlib.Path(__file__).absolute().parents[1] / 'output' / experiment_name
csv_paths = list(base_path.rglob(f"run_*.csv"))
csv_paths = [x for x in csv_paths if data_name in str(x)]
try:
    print(csv_paths[0])
    path_to_csv = csv_paths[0]
except:
    print("CSV NOT FOUND")

#%%
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit_o2\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_-9223184426572261465_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_-2882783775406466135_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_123658262966771440_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_2833846104345857229_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_-4046084020633530406_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit_o3\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_9165536785917687234_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_-5742271418931331568_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_5118823566521207150_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_1984275667536504455_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_7562653212300316124_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_-1515304942937934001_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_reuse_benefit\PhDCode-master-053ef69\RTREESAMPLE_Diff\2\run_8389899307811935008_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\STAGGERS_o\1\run_-2743991349254952116_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\STAGGERS\1\run_-2681837717704871834_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\STAGGERS\1\run_-7436924144071111115_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\STAGGERS\1\run_-7436924144071111115_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_-597732427835952520_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_-2710109566518561066_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\STAGGERS\1\run_-2777931881635741546_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_-7562075339071447807_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_2600276598985989598_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_-1468404031580643766_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_l\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_5931501551461697316_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_-3465869012347640324_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_-303966771824027272_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_238976074090315767_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_-8671668373053841484_0.csv"
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_4301749188625423942_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_-1444852100942215552_0.csv"
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\cmc_testbed\PhDCode-master-fc177e7\Arabic\1\run_4470463408804018824_0.csv"
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-fc177e7\Arabic\1\run_-5183163055732709053_0.csv"
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-fc177e7\Arabic\1\run_-422488834876466698_0.csv"
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-fc177e7\Arabic\1\run_-4312260045481824178_0.csv"
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-fc177e7\Arabic\1\run_-3704333371991248595_0.csv"
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-fc177e7\STAGGERS\1\run_-7863623006466259731_0.csv"
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-fc177e7\Arabic\1\run_5114187662324734754_0.csv"
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_-5134628783786329387_0.csv"
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_-1738608345554613005_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-fc177e7\RTREESAMPLE_Diff\1\run_3525975781498042960_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-fc177e7\Arabic\1\run_5114187662324734754_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_2\PhDCode-master-d965ab7\Arabic\1\run_9205705404765103408_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_4\PhDCode-master-d965ab7\Arabic\1\run_7465268297433467859_0.csv"
# STAGGERS cndpm - no prior, 500 sleep with batch 1 to correspond to an online setting
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_5\PhDCode-master-d965ab7\STAGGERS\1\run_-2952965605661407641_0.csv"
# STAGGERS select_cndpm_base
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_5\PhDCode-master-d965ab7\STAGGERS\1\run_-1744360918285954307_0.csv"
# STAGGERS upper_bound_cndpm_base
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_5\PhDCode-master-d965ab7\STAGGERS\1\run_-8584165160872375685_0.csv"
# STAGGERS upper_bound
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_5\PhDCode-master-d965ab7\STAGGERS\1\run_-1728499882426024604_0.csv"
# STAGGERS select standard
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_5\PhDCode-master-d965ab7\STAGGERS\1\run_1805466391890044670_0.csv"
# STAGGERS cndpm_prior
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_5\PhDCode-master-d965ab7\STAGGERS\1\run_-1326503117426440158_0.csv"
# Longer STAGGERS select cndpm_base
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_6\PhDCode-master-d965ab7\STAGGERS\1\run_1701120817830558487_0.csv"
# Longer STAGGERS upper_bound cndpm_base
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_6\PhDCode-master-d965ab7\STAGGERS\1\run_165747606305211501_0.csv"
# Longer STAGGERS select standard
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_6\PhDCode-master-d965ab7\STAGGERS\1\run_6073351949822369145_0.csv"
# Longer STAGGERS cndpm prior
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_6\PhDCode-master-d965ab7\STAGGERS\1\run_2715123469914942515_0.csv"
# Longer STAGGERS cndpm noprior
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_6\PhDCode-master-d965ab7\STAGGERS\1\run_3810838760907244186_0.csv"
# Longer STAGGERS select cndpm_base_2
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_6\PhDCode-master-2286687\STAGGERS\1\run_1699273559368212720_0.csv"
# Longer STAGGERS select cndpm_base_3
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_6\PhDCode-master-2286687\STAGGERS\1\run_-3457831068152199832_0.csv"
# Longer STAGGERS select cndpm_base_4
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_6\PhDCode-master-2286687\STAGGERS\1\run_32683348221569676_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_7\PhDCode-master-2286687\STAGGERS\1\run_-4269620887424175919_0.csv"

# path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_6\PhDCode-master-2286687\RTREESAMPLE_Diff\1\run_-3740405981645354036_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_7\PhDCode-master-caa89ba\cmc\1\run_8147842185737755302_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_7\PhDCode-master-caa89ba\Arabic\1\run_-8455550762636120818_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_7\PhDCode-master-caa89ba\Arabic\1\run_8592448840674116932_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_7\PhDCode-master-caa89ba\Arabic\1\run_-1971995732898857692_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_7\PhDCode-master-1a0aa45\Arabic\1\run_-610579844836038864_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_7\PhDCode-master-1a0aa45\Arabic\1\run_3245286464534252987_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_7\PhDCode-master-1a0aa45\Arabic\1\run_-8117146838571308218_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_7\PhDCode-master-1a0aa45\RTREESAMPLE_Diff\1\run_6859150165838344416_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\AQTemp\1\run_-6813455094144740549_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\test_cndpm_7\PhDCode-master-1a0aa45\AQTemp\1\run_-7292013161664249569_0.csv"

# Short STAGGERS - cc
path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\STAGGERS\2\run_-8849049942210103776_0.csv"
# Short STAGGERS - cc_cndpm_base
path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\STAGGERS\2\run_-5336569775636633752_0.csv"
# Short STAGGERS - cndpm
path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\STAGGERS\2\run_-2700465354989151589_0.csv"
# Short STAGGERS - upper_bound
path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\STAGGERS\2\run_120687971538166270_0.csv"
# Short STAGGERS - upper_bound_cndpm_base
path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\STAGGERS\2\run_-6346157602699552952_0.csv"
# Short Arabic - cc
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_-4833921452513628429_0.csv"
# Short Arabic - cc_cndpm_base
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_7015076232704290810_0.csv"
# Short Arabic - cndpm
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_2789321365962527231_0.csv"
# Short Arabic - cndpm_noprio
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_6528714596228183659_0.csv"
# Short Arabic - upper_bound
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_1077546576240376245_0.csv"
# Short Arabic - upper_bound_cndpm_base
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_-3060671018603268805_0.csv"
# Long Arabic - cc
path_to_csv = r"S:\PhD\Packages\PhDCode\output\longer_cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_-1910855383261278879_0.csv"
# Long Arabic - cc_cndpm_base
path_to_csv = r"S:\PhD\Packages\PhDCode\output\longer_cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_-4939464340714169513_0.csv"
# Long Arabic - cndpm
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\longer_cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_-5112098789995308632_0.csv"
# Long Arabic - cndpm_noprio
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\longer_cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_1809438585178761101_0.csv"
# Long Arabic - upper_bound
path_to_csv = r"S:\PhD\Packages\PhDCode\output\longer_cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_-120397201642799558_0.csv"
# Long Arabic - upper_bound_cndpm_base
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\longer_cndpm_comparison\PhDCode-master-1a0aa45\Arabic\2\run_-3060671018603268805_0.csv"
#%%
df = pd.read_csv(path_to_csv)
df.head()
# focus=['A', 'B', 'C']
# focus=['A', 'B']
focus=['A', 'B', 'C', "D", "E", "F"]
plot_state_relevance = False
plot_accuracy = True
# save_name = "SELeCT"
# save_name = "advantage"
# save_name = "CNDPM-Prior"
# save_name = "CNDPM-Select-Test-RTREE-CNDPM_Prior"
# save_name = "CNDPM-Select-Test-RTREE-UpperMLP"
# save_name = "CNDPM-Select-Test-RTREE-CNDPM_NoPrior"
# save_name = "CNDPM-Select-Test-RTREE-SelectMLP"4
# save_name = "STAGGERS_upperboundcndpm"
# save_name = "STAGGERS_selectcndpm"
# save_name = "STAGGERS_upperbound"
# save_name = "STAGGERS_selectstandard"
save_name = "STAGGERS_l_select_cndpm_base"
save_name = "STAGGERS_l_select_standard_base"
save_name = "STAGGERS_l_ub_cndpm_base"
save_name = "STAGGERS_l_cndpm_prior"
save_name = "STAGGERS_l_cndpm_noprior"
save_name = "STAGGERS_l_select_cndpm_batch_base"
save_name = "cndpm-comparison-STAGGERS-s-cc"
save_name = "cndpm-comparison-STAGGERS-s-cc_cndpm_base"
save_name = "cndpm-comparison-STAGGERS-s-cndpm"
save_name = "cndpm-comparison-STAGGERS-s-upper_bound"
save_name = "cndpm-comparison-STAGGERS-s-upper_bound_cndpm_base"
# save_name = "cndpm-comparison-Arabic-s-cc"
# save_name = "cndpm-comparison-Arabic-s-cc_cndpm_base"
# save_name = "cndpm-comparison-Arabic-s-cndpm"
# save_name = "cndpm-comparison-Arabic-s-cndpm_noprior"
# save_name = "cndpm-comparison-Arabic-s-upper_bound"
# save_name = "cndpm-comparison-Arabic-s-upper_bound_cndpm_base"
# save_name = "cndpm-comparison-Arabic-l-cc"
save_name = "cndpm-comparison-Arabic-l-cc_cndpm_base"
# save_name = "cndpm-comparison-Arabic-l-cndpm"
# save_name = "cndpm-comparison-Arabic-l-cndpm_noprior"
save_name = "cndpm-comparison-Arabic-l-upper_bound"
# save_name = "cndpm-comparison-Arabic-l-upper_bound_cndpm_base"
# save_file = True
save_file = True
# plot_probabilities(df, 'state_relevance', focus=focus, plot_state_relevance=plot_state_relevance, plot_accuracy=plot_accuracy, save_name=save_name, save_file=True)
plot_probabilities(df, 'state_relevance', focus=focus, plot_state_relevance=plot_state_relevance, plot_accuracy=plot_accuracy, save_name=save_name, save_file=save_file)
# plt.savefig(f"{save_name}-{focus}-{plot_state_relevance}-{plot_accuracy}.pdf")
# plot_probabilities(df, 'concept_posteriors')