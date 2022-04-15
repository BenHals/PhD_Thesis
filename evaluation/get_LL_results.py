#%%
import argparse
import json
import os
import pathlib
import pandas as pd
import numpy as np
import tqdm
from collections import Counter

class BasicNB:
    """
    Basic Naive Bayes classifier for 1D X and y.
    If an X value is seen which is not in training set, mark as a failed prediction.
    """
    def __init__(self):
        self.p_X = {}
        self.p_y = {}
        self.p_Xy = {}
    
    def get_key(self, x_val, y_val):
        return str(int(x_val)) +':' + str(int(y_val))

    def fit(self, X_train, y_train):
        count_X = Counter()
        count_y = Counter()
        count_Xy= Counter()

        for x_val, y_val in zip(X_train, y_train):
            count_X[x_val] += 1
            count_y[y_val] += 1
            count_Xy[self.get_key(x_val, y_val)] += 1
        
        total_count = X_train.shape[0]

        for k, v in count_X.items():
            self.p_X[k] = v / total_count
        for k, v in count_y.items():
            self.p_y[k] = v / total_count
        for k, v in count_Xy.items():
            x_val, y_val = map(lambda x: int(float(x)), k.split(':'))
            total_count_for_y = count_y[y_val]
            self.p_Xy[k] = v / total_count_for_y
        
    def predict(self, X_test):
        predictions = []
        for x_val in X_test:
            class_predictions = {}
            for y_val in self.p_y:
                key = self.get_key(x_val, y_val)
                if key not in self.p_Xy:
                    class_predictions[y_val] = -1
                    continue
                class_predictions[y_val] = self.p_Xy[key] * self.p_y[y_val]
            
            prediction, prob = max(class_predictions.items(), key = lambda x: x[1])
            if prob == -1:
                prediction = -1
            predictions.append(prediction)
        return predictions
    
    def get_accuracy(self, predictions, y_test):
        right = 0
        wrong = 0
        for p, y in zip(predictions, y_test):
            right += p == y
            wrong += p != y
        
        return right / (right + wrong)

def mean_stdev_size(X):
    return f"{np.mean(X):.2f}|{np.std(X):.2f}<{len(X)}>"

def get_accuracy_increase(df : pd.DataFrame, n_obs_first_check:int, n_obs_second_check:int) -> float:
    """ Get the increase in accuracy between the timestep given by n_obs_first_check and n_obs_second_check.
    The DataFrame df should include an 'example' column with the number of examples seen at each row, and a overall_accuracy column with the overall_accuracy up to that row.
    """

    first_check_accuracy = df[df['example'] == n_obs_first_check].iloc[0]['overall_accuracy']
    second_check_accuracy = df[df['example'] == n_obs_second_check].iloc[0]['overall_accuracy']
    return second_check_accuracy - first_check_accuracy

def get_bayesian_CI_accuracy(df : pd.DataFrame, n_obs_training_period:int) -> float:
    """ Get the accuracy of a Naive Bayes classifier trained to take the active state and predict ground truth context on the training period over the remaining stream.
    The DataFrame df should include an 'example' column with the number of examples seen at each row, and an 'active_model' column with the active state at each observation, 
    and a ground_truth_concept column with the ground truth context. Optionally, merge_model and repair_model columns can be included for repaired active states.
    """

    active_model = df['active_model']
    merge_model = df['merge_model'] if 'merge_model' in df.columns else active_model
    repair_model = df['repair_model'] if 'repair_model' in df.columns else merge_model
    gt_model = df['ground_truth_concept'].fillna(-1).values
    CI_inference_classifier = BasicNB()
    y_train = gt_model[df['example'] < n_obs_training_period]
    y_test = gt_model[df['example'] >= n_obs_training_period]
    X_train = repair_model[df['example'] < n_obs_training_period].values
    X_test = repair_model[df['example'] >= n_obs_training_period].values
    CI_inference_classifier.fit(X_train, y_train)
    predicted_context = CI_inference_classifier.predict(X_test)
    accuracy = CI_inference_classifier.get_accuracy(predicted_context, y_test)

    return accuracy

def get_LL_results(experiment_name):
    file_path = pathlib.Path(__file__).resolve()
    main_dir = file_path.parents[1]

    # Collect results
    output = main_dir / 'output' / experiment_name
    results_files = list(output.rglob('results*'))
    results = []
    for rp in tqdm.tqdm(results_files):
        result = json.load(rp.open('r'))
        run_log_path = rp.parent / f"{rp.stem.split('results_')[1]}.csv"
        run_df = pd.read_csv(run_log_path)
        accuracy_increase = get_accuracy_increase(run_df, int(run_df.shape[0] * 0.05), int(run_df.shape[0] * 0.9))
        result['accuracy_increase'] = accuracy_increase
        bayesian_CI_accuracy = get_bayesian_CI_accuracy(run_df, int(run_df.shape[0] * 0.1))
        result['bayesian_CI_accuracy'] = bayesian_CI_accuracy
        results.append(result)
    
    df = pd.DataFrame(results)
    df = df[(df['classifier'] != 'cc') | (df['buffer_ratio'] == 0.025)]
    classification_performance_metric = 'kappa'
    adaption_performance_metric = 'GT_mean_f1' # measure for C-F1
    print("**************TEST RESULTS**************")
    pd.set_option('display.max_rows', 500)
    print(df.groupby(['data_name', 'classifier', "fs_method"])[[classification_performance_metric, adaption_performance_metric, 'accuracy_increase', 'bayesian_CI_accuracy', 'driftdetect_50_kappa']].aggregate(mean_stdev_size))
    return df

#%%
if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--experimentname', default="expDefault", type=str)
    args = my_parser.parse_args()

    df = get_LL_results(args.experimentname)
    # file_path = pathlib.Path(__file__).resolve()
    # main_dir = file_path.parents[1]

    # # Collect results
    # output = main_dir / 'output' / args.experimentname
    # results_files = list(output.rglob('results*'))
    # results = []
    # for rp in tqdm.tqdm(results_files):
    #     result = json.load(rp.open('r'))
    #     run_log_path = rp.parent / f"{rp.stem.split('results_')[1]}.csv"
    #     run_df = pd.read_csv(run_log_path)
    #     accuracy_increase = get_accuracy_increase(run_df, int(run_df.shape[0] * 0.05), int(run_df.shape[0] * 0.9))
    #     result['accuracy_increase'] = accuracy_increase
    #     bayesian_CI_accuracy = get_bayesian_CI_accuracy(run_df, int(run_df.shape[0] * 0.1))
    #     result['bayesian_CI_accuracy'] = bayesian_CI_accuracy
    #     results.append(result)
    
    # df = pd.DataFrame(results)
    # df = df[df['buffer_ratio'] == 0.05]
    # classification_performance_metric = 'kappa'
    # adaption_performance_metric = 'GT_mean_f1' # measure for C-F1
    # print("**************TEST RESULTS**************")
    # pd.set_option('display.max_rows', 500)
    # print(df.groupby(['data_name', 'classifier', "fs_method"])[[classification_performance_metric, adaption_performance_metric, 'accuracy_increase', 'bayesian_CI_accuracy', 'driftdetect_50_kappa']].aggregate(mean_stdev_size))


#%%
df = get_LL_results('t_histogram_buffer_ratio')

#%%
classification_performance_metric = 'kappa'
adaption_performance_metric = 'GT_mean_f1' # measure for C-F1
tab_df = df.groupby(['data_name', 'classifier', "fs_method"])[[classification_performance_metric, adaption_performance_metric, 'bayesian_CI_accuracy']].aggregate(mean_stdev_size)
tab_df = tab_df.unstack('data_name')
tab_df = tab_df.transpose()
with open('./chap8/long_run_results_0025.txt', 'w+') as f:
    f.write(tab_df.to_latex())