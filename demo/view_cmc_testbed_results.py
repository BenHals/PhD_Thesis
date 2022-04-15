import argparse
import json
import os
import pathlib
import pandas as pd

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--dataset', default="cmc", type=str)
    my_parser.add_argument('--classifier', default="cc", type=str)
    my_parser.add_argument('--cpus', default=1, type=int)
    my_parser.add_argument('--conceptdifficulty', default=0, type=int)
    my_parser.add_argument('--concept_max', default=6, type=int)
    my_parser.add_argument('--repository_cap', default=-1, type=int)
    my_parser.add_argument('--repeats', default=3, type=int)
    my_parser.add_argument('--TMforward', default=1, type=int)
    my_parser.add_argument('--TMnoise', default=0.0, type=float)
    my_parser.add_argument('--window_size', default=100, type=int)
    my_parser.add_argument('--run_minimal', action='store_true')
    my_parser.add_argument('--discritize_stream', action='store_true')
    args = my_parser.parse_args()

    file_path = pathlib.Path(__file__).resolve()
    main_dir = file_path.parents[1]

    # Collect results
    output = main_dir / 'output' / 'cmc_testbed'
    results_files = list(output.rglob('results*'))
    results = []
    for rp in results_files:
        result = json.load(rp.open('r'))
        results.append(result)
    
    df = pd.DataFrame(results)
    
    classification_performance_metric = 'kappa'
    adaption_performance_metric = 'GT_mean_f1' # measure for C-F1
    print("**************TEST RESULTS**************")
    print(df.groupby(['data_name', 'classifier', 'repository_cap', 'discritize_stream'])[[classification_performance_metric, adaption_performance_metric, 'driftdetect_50_accuracy']].mean())



