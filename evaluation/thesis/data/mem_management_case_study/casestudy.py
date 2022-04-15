import sys, os
import glob
import argparse
import random
import pickle
import results_df

import numpy as np
import pandas as pd
import io
import pathlib

import numpy as np
import pandas as pd
import os, time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from graphviz import Digraph
from collections import Counter
sns.set()
sns.set_context("paper")
sns.set_style("ticks")
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory",
    help="tdata generator for stream", default="datastreams")
args = vars(ap.parse_args())

same_cl_comparisons = {}

directory = args['directory']
info_names = ['noise', 'nc', 'hd', 'ed', 'ha', 'ea', 'hp', 'epa', 'st', 'gradual', 'drift_window']
rename_strategies = {'rA': "#E", "auc": "AAC", "score": "EP", "acc": "Acc", "age":"FIFO", "LRU":"LRU", "div":"DP"}
filenames = list(pathlib.Path(directory).glob('system*.csv'))
drift_fn = f'{directory}{os.sep}drift_info.csv'
drift_data = pd.read_csv(drift_fn)
drift_data.columns = ['example', 'ground_truth_concept', 'drift_occured']
print(drift_data.tail())
last_col = drift_data.iloc[-1]
print(last_col)


print(last_col)
drift_data = drift_data.append({"example": last_col['example'] + 1, "ground_truth_concept": last_col['ground_truth_concept'], "drift_occured": last_col['drift_occured']}, ignore_index= True)
print(drift_data.tail())
print(filenames)
for csv_filename in filenames:
    csv_filename = csv_filename.resolve()
    print(csv_filename)
    parent_dirs = csv_filename.parts
    print(parent_dirs)
    can_make_table = False
    experiment_info = None
    if len(parent_dirs) > 2:
        info_dir = parent_dirs[-2]
        info = tuple(info_dir.split('_'))
        info = info[:len(info_names)]
        print(info)
        drift_window = 0
        if len(parent_dirs) > 3:
            drift_dir = parent_dirs[-3]
            print(drift_dir)
            if drift_dir[-1] == 'w':
                try:
                    drift_window = int(drift_dir[:-1])
                except:
                    print("cant_convert to drift window")
            else:
                print("no w")
        info = tuple([x for x in info] + [drift_window])
        if len(info) == len(info_names):
            can_make_table = True

            experiment_info = dict(zip(info_names, info))
    if experiment_info is None:
        info = (-1, -1, -1, -1, -1, -1, -1, -1, parent_dirs[-1], -1, -1)
        experiment_info = dict(zip(info_names, info))
        can_make_table = True
    print(experiment_info)
    drift_fn = f'{csv_filename.parents[0]}{os.sep}drift_info.csv'
    if os.path.exists(drift_fn):
        drift_data = pd.read_csv(drift_fn)
        drift_data.columns = ['example', 'ground_truth_concept', 'drift_occured']
        print(drift_data.tail())
        last_col = drift_data.iloc[-1]
        print(last_col)
        
        
        print(last_col)
        drift_data = drift_data.append({"example": last_col['example'] + 1, "ground_truth_concept": last_col['ground_truth_concept'], "drift_occured": last_col['drift_occured']}, ignore_index= True)
        print(drift_data.tail())
    else:
        drift_data = None
    filename_str = str(csv_filename.parts[-1])
    print(f"filename_str: {filename_str}")
    dash_split = filename_str.replace('--', '-').split('-')
    print(dash_split)

    run_name = dash_split[0]
    run_noise = 0
    cl = 'def'
    mm = 'def'
    sensitivity = 'def'
    window = 'def'
    sys_learner = 'def'
    poisson = "def"
    optimal_drift = False
    similarity = 'def'
    merge = 'def'
    time = -1
    memory = -1
    merge_similarity = 0.9

    if run_name == 'system':
        run_noise = dash_split[1]
        cl = dash_split[2].split('.')[0]
        if 'ARF' in filename_str:
            sys_learner = 'ARF'
        if 'HAT' in filename_str:
            sys_learner = 'HAT'
        if 'HATN' in filename_str:
            sys_learner = 'HATN'
        if 'HN' in filename_str:
            sys_learner = 'HN'

        if filename_str[-5].isnumeric() and filename_str[-6] == '-':
            print(filename_str)
            print("not a final csv")
            continue
        if len(dash_split) > 3:
            mm = dash_split[3].split('.')[0]
        else:
            mm = 'def'
        if len(dash_split) > 4:
            sensitivity = dash_split[4]
            if 'e' in sensitivity:
                sensitivity = dash_split[4] + dash_split[5]
                if len(dash_split) > 6:
                    window = dash_split[6]
                else:
                    window = 'def'
            else:
                if len(dash_split) > 5:
                    window = dash_split[5]
                else:
                    window = 'def'
        else:
            sensitivity = 'def'
        if len(dash_split) > 8:
            if len(str(dash_split[8].split('.')[0])) < 3:
                poisson = str(dash_split[8].split('.')[0])
        if len(dash_split) > 10:
            optimal_drift = dash_split[10] == 'True'
        if len(dash_split) > 11:
            similarity = dash_split[11]
        if len(dash_split) > 12:
            merge = dash_split[12]
        if len(dash_split) > 13:
            merge_similarity = '.'.join(dash_split[13].split('.')[:-1])
        # if len(dash_split) > 13:
        #     merge = dash_split[13]

    # elif run_name == 'rcd':
    else:
        if filename_str[-5].isnumeric() and filename_str[-6] == '-':
            print(filename_str)
            print("not a final csv")
            continue
        cl = dash_split[1].split('.')[0]
        if 'py' in filename_str:
            sys_learner = 'py'
        if 'pyn' in filename_str:
            sys_learner = 'pyn'
        if len(dash_split) > 4:
            run_noise = dash_split[4]
    # else:
    #     cl = 0



    rep = 0
    extended_names = info_names + ['ml', 'cl', 'mem_manage', 'rep', 'sens', 'window', 'sys_learner', 'poisson', 'od', 'sm', 'merge', 'run_noise', 'merge_similarity']
    extended_info = tuple(list(info) + [run_name, cl, mm, rep, sensitivity, window, sys_learner, poisson, optimal_drift, similarity, merge, run_noise, merge_similarity])
    print(extended_info)

    if cl not in same_cl_comparisons:
        same_cl_comparisons[cl] = []
    same_cl_comparisons[cl].append((csv_filename, mm))

print(same_cl_comparisons)

for cl_i, cl in enumerate(same_cl_comparisons.keys()):
    comparisons = same_cl_comparisons[cl]
    plt.figure(figsize=(20,5))
    line_colors = ["red", "green"]
    for ci, comparison in enumerate(comparisons):
        data = pd.read_csv(comparison[0])
        if 'ground_truth_concept' not in data.columns and not drift_data is None:
            if 'ex' in data.columns:
                data.rename(columns={'ex': 'example'}, inplace=True)
            data = data.merge(drift_data, on = 'example', how = 'left')

            plotting_states = True
        print(data.head())
        state_id = 0

        sys_concept_position = 0.03 + 0.1 * (ci+1)

        highlight_concepts = [0]
        # highlight_concepts = [0, 2]
        # highlight_concepts = [0, 1, 2, 3]
        all_highlight = highlight_concepts == [0, 1, 2, 3]
        show_ranges = []
        in_concept = None
        for i, row in data.iterrows():
            if int(row['ground_truth_concept']) in highlight_concepts:
                if in_concept is None:
                    show_ranges.append([i, None])
                    in_concept = int(row['ground_truth_concept'])
            if int(row['ground_truth_concept']) != in_concept:
                if in_concept is not None:
                    show_ranges[-1][1] = i-1
                    in_concept = None
        if in_concept is not None:
            show_ranges[-1][1] = data.shape[0]

        if 'overall_accuracy' in data.columns:
            data['overall_accuracy'] = data['overall_accuracy']/100
            data['sliding_window_accuracy'] = data['sliding_window_accuracy']/100
            lag_row = pd.Series()
            start_row = pd.Series()
            for i, row in data.iterrows():
                if 'ground_truth_concept' in start_row:
                    if row['ground_truth_concept'] != start_row['ground_truth_concept']:
                        # plt.plot([start_row['example'], row['example']], [0.03, 0.03], 
                        #     color = sns.color_palette()[int(start_row['ground_truth_concept']) % len(sns.color_palette())])
                        selected = all_highlight or any([abs(int(start_row['example']) - x[0]) < 1000 for x in show_ranges]) or any([abs(int(row['example']) - x[1]) < 1000 for x in show_ranges])
                        plt.plot(data.iloc[int(start_row['example']):int(row['example'])]['example'], data.iloc[int(start_row['example']):int(row['example'])]['overall_accuracy'], linewidth = 1, color = line_colors[ci],  dashes= (5, 2), alpha = 1 if selected else 0.1, label = "_nolegend_")
                        plt.plot(data.iloc[int(start_row['example']):int(row['example'])]['example'], data.iloc[int(start_row['example']):int(row['example'])]['sliding_window_accuracy'], linewidth = 1, color = line_colors[ci], alpha = 1 if selected else 0.1, label = "_nolegend_")
                            
                        lag_row = start_row
                        start_row = row
                else:
                    lag_row = start_row
                    start_row = row
            last_example_estimate = data['example'].iloc[-1] + (data['example'].iloc[-1] - data['example'].iloc[-2])
            selected = all_highlight or any([abs(int(start_row['example']) - x[0]) < 1000 for x in show_ranges]) or any([abs(int(last_example_estimate) - x[1]) < 1000 for x in show_ranges])
            plt.plot(data.iloc[int(start_row['example']):int(last_example_estimate)]['example'], data.iloc[int(start_row['example']):int(last_example_estimate)]['overall_accuracy'], linewidth = 1, color = line_colors[ci],  dashes= (5, 2), label = f"{rename_strategies[comparison[1]]} Overall Accuracy", alpha = 1 if selected else 0.1)
            plt.plot(data.iloc[int(start_row['example']):int(last_example_estimate)]['example'], data.iloc[int(start_row['example']):int(last_example_estimate)]['sliding_window_accuracy'], linewidth = 1, color = line_colors[ci], label = f"{rename_strategies[comparison[1]]} Sliding Window Accuracy", alpha = 1 if selected else 0.1)
            # plt.plot([start_row['example'], last_example_estimate], [0.03, 0.03], 
            #                 color = sns.color_palette()[int(start_row['ground_truth_concept']) % len(sns.color_palette())])  


            # plt.plot(data['example'], data['overall_accuracy'], linewidth = 1, color = line_colors[ci], marker = 'o' if ci > 0 else 'x', markevery = 0.05,  dashes= (5, 2), label = f"{comparison[1]} Overall Accuracy")
            # plt.plot(data['example'], data['sliding_window_accuracy'], linewidth = 1, color = line_colors[ci], marker = 'o' if ci > 0 else 'x', markevery = 0.05, label = f"{comparison[1]} Sliding Window Accuracy")
            # sns.lineplot(x='example', y='overall_accuracy',
            #         linewidth=1, data=data,
            #         color = line_colors[ci],
            #         marker= 'o' if ci > 0 else None,
            #         dashes= "-",
            #         legend=None)
            # sns.lineplot(x='example', y='sliding_window_accuracy',
            #         linewidth=1, data=data,
            #         color = line_colors[ci],
            #         markers= True if ci > 0 else False,
            #         legend=None)
        # if 'change_detected' in data.columns:
        #     # Plot the change detections in green
        #     for i, row in data[data['change_detected'] != 0].iterrows():
        #         plt.plot([row['example'], row['example']], [sys_concept_position - 0.02, sys_concept_position + 0.02], color = "green")

        if 'ground_truth_concept' in data.columns:
            lag_row = pd.Series()
            start_row = pd.Series()
            for i, row in data.iterrows():
                if 'ground_truth_concept' in start_row:
                    if row['ground_truth_concept'] != start_row['ground_truth_concept']:
                        selected = all_highlight or any([abs(int(start_row['example']) - x[0]) < 1000 for x in show_ranges]) or any([abs(int(row['example']) - x[1]) < 1000 for x in show_ranges])
                        plt.plot([start_row['example'], row['example']], [0.03, 0.03], 
                            color = sns.color_palette()[int(start_row['ground_truth_concept']) % len(sns.color_palette())], alpha = 1 if selected else 0.1, linewidth = 2)
                            
                        lag_row = start_row
                        start_row = row
                else:
                    lag_row = start_row
                    start_row = row
            last_example_estimate = data['example'].iloc[-1] + (data['example'].iloc[-1] - data['example'].iloc[-2])
            selected = all_highlight or any([abs(int(start_row['example']) - x[0]) < 1000 for x in show_ranges]) or any([abs(int(last_example_estimate) - x[1]) < 1000 for x in show_ranges])
            plt.plot([start_row['example'], last_example_estimate], [0.03, 0.03], 
                            color = sns.color_palette()[int(start_row['ground_truth_concept']) % len(sns.color_palette())], alpha = 1 if selected else 0.1, linewidth = 2)
            plt.text(last_example_estimate  * 1.005, 0.03, f"Contexts")
        if 'ground_truth_concept' in data.columns:
            lag_row = pd.Series()
            start_row = pd.Series()
            for i, row in data.iterrows():
                if 'ground_truth_concept' in start_row:
                    if row['ground_truth_concept'] != start_row['ground_truth_concept']:
                        x = (row['example'] - start_row['example']) / 2 + start_row['example']
                        selected = all_highlight or any([abs(int(start_row['example']) - x[0]) < 1000 for x in show_ranges]) or any([abs(int(row['example']) - x[1]) < 1000 for x in show_ranges])
                        plt.plot(x, 0.03, 'o', color='white', markersize=12, alpha = 1 if selected else 0.1)
                        plt.plot(x, 0.03, 'k', marker=f"${list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[int(start_row['ground_truth_concept'])]}$", color=sns.color_palette()[int(start_row['ground_truth_concept']) % len(sns.color_palette())], markersize=9, alpha = 1 if selected else 0.1)
                            
                        lag_row = start_row
                        start_row = row
                else:
                    lag_row = start_row
                    start_row = row
            last_example_estimate = data['example'].iloc[-1] + (data['example'].iloc[-1] - data['example'].iloc[-2]) 
            x = (last_example_estimate - start_row['example']) / 2 + start_row['example']
            selected = all_highlight or any([abs(int(start_row['example']) - x[0]) < 1000 for x in show_ranges]) or any([abs(int(last_example_estimate) - x[1]) < 1000 for x in show_ranges])
            plt.plot(x, 0.03, 'o', color='white', markersize=12, alpha = 1 if selected else 0.1)
            plt.plot(x, 0.03, 'k', marker=f"${list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[int(start_row['ground_truth_concept'])]}$", color=sns.color_palette()[int(start_row['ground_truth_concept']) % len(sns.color_palette())], markersize=9, alpha = 1 if selected else 0.1)   


        if 'model_update' in data.columns:
            # Plot model updates (hoeffding tree splits)
            for row in data[data['model_update'] != 0]:
                plt.plot([row['example'], row['example']], [0.5, 0.7])

        if 'drift_occured' in data.columns:
            # Plot model updates (hoeffding tree splits)
            for i,row in data[data['drift_occured'] != 0].iterrows():
                selected = all_highlight or (int(row['example']) in show_ranges and int(row['example']) + 500 in show_ranges)
                plt.plot([row['example'], row['example']], [0.03, 0.07], color="Black", alpha = 1 if selected else 0.1)  

        if 'system_concept' in data.columns:
            lag_row = pd.Series()
            start_row = pd.Series()
            for i, row in data.iterrows():
                if 'system_concept' in start_row:
                    if row['system_concept'] != start_row['system_concept']:
                        selected = all_highlight or any([abs(int(start_row['example']) - x[0]) < 1000 for x in show_ranges]) or any([abs(int(row['example']) - x[1]) < 1000 for x in show_ranges])
                        plt.plot([start_row['example'], row['example']], [sys_concept_position, sys_concept_position], 
                            color = sns.color_palette()[int(start_row['system_concept']) % len(sns.color_palette())], alpha = 1 if selected else 0.1, linewidth = 2)
                        
                        
                        lag_row = start_row
                        start_row = row
                else:
                    lag_row = start_row
                    start_row = row
            last_example_estimate = data['example'].iloc[-1] + (data['example'].iloc[-1] - data['example'].iloc[-2])
            selected = all_highlight or any([abs(int(start_row['example']) - x[0]) < 1000 for x in show_ranges]) or any([abs(int(last_example_estimate) - x[1]) < 1000 for x in show_ranges])
            plt.plot([start_row['example'], last_example_estimate], [sys_concept_position, sys_concept_position], 
                            color = sns.color_palette()[int(start_row['system_concept']) % len(sns.color_palette())], alpha = 1 if selected else 0.1, linewidth = 2)  
        if 'system_concept' in data.columns:
            lag_row = pd.Series()
            start_row = pd.Series()
            for i, row in data.iterrows():
                if 'system_concept' in start_row:
                    if row['system_concept'] != start_row['system_concept']:
                        x = (row['example'] - start_row['example']) / 2 + start_row['example']
                        selected = all_highlight or any([abs(int(start_row['example']) - x[0]) < 1000 for x in show_ranges]) or any([abs(int(row['example']) - x[1]) < 1000 for x in show_ranges])
                        plt.plot(x, sys_concept_position, 'o', color='white', markersize=12, alpha = 1 if selected else 0.1)
                        plt.plot(x, sys_concept_position, 'k', marker=f"${int(start_row['system_concept'])}$", color=sns.color_palette()[int(start_row['system_concept']) % len(sns.color_palette())],
                     markersize=9, alpha = 1 if selected else 0.1)
                        plt.plot([row['example'], row['example']], [sys_concept_position - 0.02, sys_concept_position + 0.02], color = "black", linewidth = 0.5, alpha = 1 if selected else 0.1)
                        lag_row = start_row
                        start_row = row
                else:
                    lag_row = start_row
                    start_row = row
            last_example_estimate = data['example'].iloc[-1] + (data['example'].iloc[-1] - data['example'].iloc[-2])
            x = (last_example_estimate - start_row['example']) / 2 + start_row['example']
            selected = all_highlight or any([abs(int(start_row['example']) - x[0]) < 1000 for x in show_ranges]) or any([abs(int(last_example_estimate) - x[1]) < 1000 for x in show_ranges])
            plt.plot(x, sys_concept_position, 'o', color='white', markersize=12, alpha = 1 if selected else 0.1)
            plt.plot(x, sys_concept_position, 'k', marker=f"${int(start_row['system_concept'])}$", color=sns.color_palette()[int(start_row['system_concept']) % len(sns.color_palette())], markersize=9, alpha = 1 if selected else 0.1)
            plt.text(last_example_estimate * 1.005, sys_concept_position, f"{rename_strategies[comparison[1]]}")

    # plt.legend(bbox_to_anchor=(1, 1.005), loc=2, borderaxespad=0.)
    leg = plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, frameon=False)
    for l in leg.get_lines():
        l.set_alpha(1)
    plt.xlabel("Examples")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"{directory}{os.sep}{run_name}-{cl_i}-{'_'.join([str(c) for c in highlight_concepts])}.pdf")
    plt.clf()