import argparse
import json
import os
import pathlib
import pandas as pd
import numpy as np

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--experimentname', default="expDefault", type=str)
    args = my_parser.parse_args()

    file_path = pathlib.Path(__file__).resolve()
    main_dir = file_path.parents[1]

    # Collect results
    output = main_dir / 'output' / args.experimentname
    results_files = list(output.rglob('results*'))
    results = []
    for rp in results_files:
        result = json.load(rp.open('r'))
        results.append(result)
    
    df = pd.DataFrame(results)
    
    classification_performance_metric = 'kappa'
    adaption_performance_metric = 'GT_mean_f1' # measure for C-F1
    print("**************TEST RESULTS**************")
    def mean_stdev_size(X):
        return f"{np.mean(X):.4f}|{np.std(X):.2f}<{len(X)}>"
    pd.set_option('display.max_rows', 500)
    print(df.groupby(['data_name', 'classifier', 'repository_max', 'cndpm_use_prior', "valuation_policy", "fs_method"])[['overall_accuracy', classification_performance_metric, adaption_performance_metric, 'driftdetect_50_accuracy']].aggregate(mean_stdev_size))

    def mean_stdev(X):
        return f"{np.mean(X):.2f}|{np.std(X):.2f}"
    classifier_str_map = {
        'ccTrue': 'SELeCT',
        'cc_cndpm_baseTrue': '$S_{cn}$',
        'upper_boundTrue': 'UB',
        'upper_bound_cndpm_baseTrue': 'UB$_{cn}$',
        'cndpmTrue': 'CNDPM$_{p}$',
        'cndpmFalse': 'CNDPM$_{np}$',
    }
    df_table = df
    df_table['classifier_str'] = (df_table['classifier'] + df_table['cndpm_use_prior'].astype(str)).map(classifier_str_map)
    df_table = df_table.melt(id_vars=['data_name', 'classifier_str'], value_vars = [classification_performance_metric, adaption_performance_metric])
    df_table = df_table.groupby(['variable', 'data_name', 'classifier_str']).aggregate(mean_stdev)
    df_table = df_table.unstack('classifier_str')
    df_table.to_latex(f"{args.experimentname}-{[classification_performance_metric, adaption_performance_metric]}-table.txt", escape=False, multirow=True)
    # print(df_table.index)
    # print(df.groupby(['data_name', 'classifier', 'cndpm_use_prior']).aggregate(mean_stdev_size)[[classification_performance_metric, adaption_performance_metric]])
    # print(df_table.unstack('classifier_str'))




