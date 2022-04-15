#%%
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from get_LL_results import BasicNB
sns.set_context('talk')








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

def plot_probabilities(df, prob_column, focus, context, CI_train_len, experiment_name="default", data_name="default", save_file=False, save_name="", run_individual=False, smoothing=True, plot_state_relevance=True, plot_inference=True):
    
    active_model = df['active_model']
    merge_model = df['merge_model'] if 'merge_model' in df.columns else active_model
    repair_model = df['repair_model'] if 'repair_model' in df.columns else merge_model
    gt_model = df['ground_truth_concept']
    active_model_ranges = ts_to_ranges(active_model)
    merge_model_ranges = ts_to_ranges(merge_model) if merge_model is not None else active_model_ranges
    repair_model_ranges = ts_to_ranges(repair_model) if repair_model is not None else merge_model_ranges
    gt_model_ranges = ts_to_ranges(gt_model)
    unique_gt_names = []
    for _,_,gt_name in gt_model_ranges:
        if gt_name not in unique_gt_names:
            unique_gt_names.append(gt_name)

    del_cols = []
    val_nas = {}

    gt_colors = plt.cm.tab10.colors
    gt_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    gt_name_color_map = lambda n: gt_colors[unique_gt_names.index(n)]

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
    if plot_state_relevance:
        fig, (ax, ax_b) = plt.subplots(nrows=2, figsize=(20, 10), sharex=True, gridspec_kw={'height_ratios':[0.7, 0.3]})
    else:
        fig, (ax, ax_b) = plt.subplots(nrows=2, figsize=(20, 5), sharex=True, gridspec_kw={'height_ratios':[0.0, 1.0]})

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
        color = gt_name_color_map(best_gt)
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
        color = gt_name_color_map(best_gt)
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
            print(state_id, gt, recall, precision, CF1)
            state_CF1_per_gt.append((CF1, gt))
        max_CF1, best_gt = max(state_CF1_per_gt, key=lambda x: x[0])
        repair_state_CF1[state_id] = (max_CF1, best_gt)
        color = gt_name_color_map(best_gt)
        repair_state_color[state_id] = color

    # sns.lineplot(data=m_df[m_df['variable'] != '-1'], x='ts', y='value', hue='variable', ax=ax, linewidth=0.5)
    # sns.scatterplot(data=m_df[m_df['variable'] != '-1'], x='ts', y='value', hue='variable', ax=ax, linewidth=0.5)
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    if repair_model_ranges is not None:
        for rs, re, v in repair_model_ranges:
            y_val = -0.2
            # if str(v) in labels:
                # color = handles[labels.index(str(v))].get_color()
            op = repair_state_CF1[v][0]
            context_match = repair_state_CF1[v][1]
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
        ax_b.annotate("Final Active State: ",
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

    # Add box around trained samples
    ax_b.add_patch(mpatches.Rectangle((0, -1.0), CI_train_len, 1.0, facecolor="none", ec='black', lw=2, ls='--'))
    ax_b.annotate("Training\nInference",
    xy=(0, -0.3),
    xycoords='data',
    ha='left',
    va='top'
    )
    CI_inference_classifier = BasicNB()
    y_train = context[df['example'] < CI_train_len]
    y_test = context[df['example'] >= CI_train_len]
    X_train = repair_model[df['example'] < CI_train_len].values
    X_test = repair_model[df['example'] >= CI_train_len].values
    CI_inference_classifier.fit(X_train, y_train)
    predicted_context = CI_inference_classifier.predict(X_test)
    predicted_context_ranges = ts_to_ranges(predicted_context, CI_train_len)
    accuracy = CI_inference_classifier.get_accuracy(predicted_context, y_test)
    print(f"Accuracy of Context-Inference was: {accuracy}")
    for rs, re, v in predicted_context_ranges:
        # if str(v) in labels:
            # color = handles[labels.index(str(v))].get_color()
        color = gt_colors[unique_gt_names.index(v)] if v != -1 else "red"
        context_label = gt_labels[unique_gt_names.index(v)] if v != -1 else "N"
        alpha_val = 1 if context_label in focus else 0.25
        y_val = -0.6
        ax_b.hlines(y = y_val, xmin = rs, xmax = re, colors = [color], alpha=alpha_val)
        mid_point = (rs+re)/2
        gt_label = context_label
        ax_b.annotate(gt_label,
        xy=(mid_point, y_val+0.1),
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
        ax_b.annotate("",
        xy=(mid_point, y_val+0.15),
        xytext=(mid_point, -0.2),
        xycoords='data',
        arrowprops={'arrowstyle':"->",
            'fc':'white',
            'ec': color,
            'alpha':alpha_val},
        ha='center',
        va='center', 
        alpha=alpha_val
        )
    ax_b.annotate("Inferred Contexts: ",
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
    ax_b.set_ylim((-1.05, 0.1))

    ax.set_ylabel("State Relevance")
    ax.set_ylim((0, 1))

    if plot_state_relevance is False:
        ax.axis('off')


    if save_file:
        print(pathlib.Path.cwd())
        plt.savefig(f"chap8/CI2-{save_name}-{focus}-{plot_state_relevance}-{plot_inference}.pdf", facecolor='white', transparent=False, bbox_inches="tight")
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
path_to_csv = r"S:\PhD\Packages\PhDCode\output\expDefault\PhDCode-master-c1561b1\covtype-Elevation\1\run_-1961503583357536946_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\expDefault\PhDCode-master-c1561b1\covtype-Elevation\1\run_1180432800481590779_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\expDefault\PhDCode-master-e25e8bc\covtype-Elevation\1\run_4056981222759274133_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\expDefault\PhDCode-master-e25e8bc\covtype-Elevation\1\run_4056981222759274133_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_repo_max\PhDCode-master-e25e8bc\covtype-Elevation\1\run_-2623410241817811415_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\t_repo_max\PhDCode-master-e25e8bc\covtype-Elevation\1\run_-6400736979592349735_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\poker-lastcard\PhDCode-master-ed2331a\poker-LastCard\1\run_8609898001333774231_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\poker-lastcard\PhDCode-master-ed2331a\poker-LastCard\1\run_7915568782811667470_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test-WD_1\PhDCode-master-e4bf741\Rangiora_test-WD_1\1\run_3747409970584674253_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test-WD_1-nonordered\PhDCode-master-e4bf741\Rangiora_test-WD_1-nonordered\1\run_-6706989350516942898_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test-WS_3-nonordered\PhDCode-master-e4bf741\Rangiora_test-WS_3-nonordered\1\run_2903518356179213113_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-WD_4-nonordered\PhDCode-master-e4bf741\Rangiora_test2-WD_4-nonordered\1\run_8990305668550363372_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-WD_4-nonordered\PhDCode-master-e4bf741\Rangiora_test2-WD_4-nonordered\1\run_-5622544089234882332_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-WD_4-nonordered\PhDCode-master-e4bf741\Rangiora_test2-WD_4-nonordered\1\run_6197770537485040561_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-WD_4-mode-nonordered\PhDCode-master-e4bf741\Rangiora_test2-WD_4-nonordered\1\run_6277088422146538107_0.csv"
# path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-WD_4-mode-nonordered\PhDCode-master-e4bf741\Rangiora_test2-WD_4-nonordered\1\run_-6911827744921969632_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-WS_4-mode-nonordered\PhDCode-master-e4bf741\Rangiora_test2-WS_4-nonordered\1\run_-3234120369925642197_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-WS_4-mode-nonordered\PhDCode-master-e4bf741\Rangiora_test2-WS_4-nonordered\1\run_1325016000452234511_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\expDefault\PhDCode-master-f408325\covtype-Elevation\1\run_4087607915628819025_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-WD_4-mode-nonordered\PhDCode-master-e4bf741\Rangiora_test2-WD_4-nonordered\1\run_-8873945120011557865_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-WD_4-mode-nonordered\PhDCode-master-e4bf741\Rangiora_test2-WD_4-nonordered\1\run_4332532440149766574_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\covtype-Slope\PhDCode-master-e4bf741\covtype-Slope\1\run_-679326763800858052_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-WD_4-mode-nonordered\PhDCode-master-e4bf741\Rangiora_test2-WD_4-nonordered\1\run_2684705200569657800_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-day-mode-nonordered\PhDCode-master-e4bf741\Rangiora_test2-day-nonordered\1\run_-8829295893927709625_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\Rangiora_test2-WS_4-mode-nonordered\PhDCode-master-e4bf741\Rangiora_test2-WS_4-nonordered\1\run_-2783198173949615223_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\covtype-Elevation_max\PhDCode-master-e4bf741\covtype-Elevation\1\run_6674090941872145902_0.csv"
path_to_csv = r"S:\PhD\Packages\PhDCode\output\poker-lastcard\PhDCode-master-e4bf741\poker-LastCard\1\run_3265884973849094726_0.csv"


#%%
df = pd.read_csv(path_to_csv)
df.head()
focus=['A', 'B', 'C', 'D', 'E', 'F']
# focus=['C']
plot_state_relevance = False
plot_inference = True
# save_name = "SELeCT"
# save_name = "SELeCT-covtype-Hist_mm"
# save_name = "SELeCT-covtype-slope"
# save_name = "SELeCT-poker"
save_name = "SELeCT-poker-hist"
# save_name = "SELeCT-rangiora_WD4_S5_2"
# save_name = "SELeCT-hist-rangiora_WD4_S5_2"
# save_name = "SELeCT-hist-rangiora_WS4_S5_2"
# save_name = "SELeCT-rangiora_day_S5_2"
# save_name = "FICSUM-rangiora_WD4_S5_2"
# save_name = "SELeCT-rangiora_WS4_S5"
# save_name = "AiRStream-rangiora_WD4_S5"
# save_name = "AiRStream-rangiora_WS4_S5"
# save_name = "advantage-covtype"
# save_name = "advantage-poker"
context = df['ground_truth_concept'].fillna(-1).values
# plot_probabilities(df, 'state_relevance', focus=focus, plot_state_relevance=plot_state_relevance, plot_inference=plot_inference, save_name=save_name, save_file=True)
# plot_probabilities(df, 'state_relevance', focus=focus, context=context, CI_train_len=18000, plot_state_relevance=plot_state_relevance, plot_inference=plot_inference, save_name=save_name, save_file=False)
plot_probabilities(df, 'state_relevance', focus=focus, context=context, CI_train_len=int(df.shape[0] * 0.1), plot_state_relevance=plot_state_relevance, plot_inference=plot_inference, save_name=save_name, save_file=True)
# plot_probabilities(df, 'state_relevance', focus=focus, context=context, CI_train_len=int(df.shape[0] * 0.33), plot_state_relevance=plot_state_relevance, plot_inference=plot_inference, save_name=save_name, save_file=True)
# plt.savefig(f"{save_name}-{focus}-{plot_state_relevance}-{plot_accuracy}.pdf")
# plot_probabilities(df, 'concept_posteriors')

# %%
