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
    my_parser.add_argument('--repeats', default=3, type=int)
    my_parser.add_argument('--TMforward', default=1, type=int)
    my_parser.add_argument('--TMnoise', default=0.0, type=float)
    my_parser.add_argument('--window_size', default=100, type=int)
    my_parser.add_argument('--concept_length', default=5000, type=int)
    my_parser.add_argument('--repository_max', default=-1, type=int)
    my_parser.add_argument('--valuation_policy', default='rA', type=str)
    my_parser.add_argument('--run_minimal', action='store_true')
    my_parser.add_argument('--discritize_stream', action='store_true')
    args = my_parser.parse_args()

    file_path = pathlib.Path(__file__).resolve()
    main_dir = file_path.parents[1]


    # Run experiment
    seeds = ' '.join(map(str, (range(1, 46) if not args.run_minimal else range(1, 5))))
    # Options are:
    # Dataset -> command
    # AQS -> 'AQSex'
    # AQT -> 'AQTemp'
    # AD -> 'Arabic'
    # CMC -> 'cmc'
    # STGR -> 'STAGGERS'
    # TREE -> 'RTREESAMPLE_HARD' 
    # WIND -> 'WINDSIM' 
    dataset = args.dataset
    classifier = args.classifier
    print(args.conceptdifficulty)
    cmd_str = f'python "{str((main_dir / "run_experiment.py").absolute())}" --forcegitcheck --seeds {seeds} --seedaction list --datalocation "{str((main_dir / "RawData").absolute())}" --datasets {dataset} --classifier {classifier} --experimentname cmc_testbed {"--single" if args.cpus <= 1 else "--cpu " + str(args.cpus)} --outputlocation "{str((main_dir / "output").absolute())}" --loglocation "{str((main_dir / "experimentlog").absolute())}" --conceptdifficulty {args.conceptdifficulty} --concept_max {args.concept_max} --repeats {args.repeats} --TMforward {args.TMforward} --TMnoise {args.TMnoise} --window_size {args.window_size} {"--discritize_stream" if args.discritize_stream else ""} --repository_max {args.repository_max} --valuation_policy {args.valuation_policy} --concept_length {args.concept_length}'
    print(cmd_str)
    os.system(f'{cmd_str}')

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
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df.groupby(['data_name', 'classifier', 'repository_max', 'discritize_stream', "valuation_policy"])[['overall_accuracy', classification_performance_metric, adaption_performance_metric, 'driftdetect_50_accuracy']].mean())



