#%%
import pathlib

import pandas as pd

#%%
base_data_location = pathlib.Path(rf"G:\My Drive\UniMine\Uni\PhD\SysMemPaper\CasestudyTest")
base_data_location.exists()


#%%



same_cl_comparisons = {}

info_names = ['noise', 'nc', 'hd', 'ed', 'ha', 'ea', 'hp', 'epa', 'st', 'gradual', 'drift_window']
rename_strategies = {'rA': "#E", "auc": "AAC", "score": "EP", "acc": "Acc", "age":"FIFO", "LRU":"LRU", "div":"DP"}
filenames = list(base_data_location.glob('system*.csv'))
drift_fn = base_data_location / f'drift_info.csv'
drift_data = pd.read_csv(drift_fn)
drift_data.columns = ['example', 'ground_truth_concept', 'drift_occured']
print(drift_data.tail())
last_col = drift_data.iloc[-1]
print(last_col)


print(last_col)
drift_data = drift_data.append({"example": last_col['example'] + 1, "ground_truth_concept": last_col['ground_truth_concept'], "drift_occured": last_col['drift_occured']}, ignore_index= True)
print(drift_data.tail())
print(filenames)

#%%
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
