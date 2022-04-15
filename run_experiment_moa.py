import argparse
import importlib
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
from re import L
import sys
import pathlib
import subprocess
import time
import warnings
from collections import Counter
from logging.handlers import RotatingFileHandler
from multiprocessing import RLock, freeze_support, Lock
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psutil
import tqdm
from pyinstrument import Profiler

from PhDCode.Data.load_data import (AbruptDriftStream,
                                               get_inorder_concept_ranges,
                                               load_real_concepts,
                                               load_synthetic_concepts)

from PhDCode.utils.eval_utils import (
    actualsize,
    NpEncoder,
    make_stream,
    extract_nominal_attributes,
    aquire_lock,
    get_package_status,
    get_drift_point_accuracy,
    get_driftdetect_point_accuracy,
    get_performance,
    get_recall_precision,
    get_discrimination_results,
    plot_feature_weights,
    get_unique_stream_names,
    get_ground_truth_concept_idx,
    dump_results,
    check_dataset_args,
)

warnings.filterwarnings('ignore')

def process_option(params):
    option, lock, queue = params
    if lock:
        tqdm.tqdm.set_lock(lock)
    np.seterr(all='ignore')
    warnings.filterwarnings('ignore')
    logging.info(pathlib.Path(sys.executable).as_posix().split('/')[-3])
    mp_process = mp.current_process()
    mp_id = 1
    try:
        mp_id = None
        if queue:
            mp_id = queue.get(timeout=10)
        if mp_id is None:
            mp_id = int(mp_process.name.split('-')[1]) % option['cpu']
    except:
        pass

    if mp_id is None:
        mp_id = 1

    proc = psutil.Process(os.getpid())
    if option['setaffinity']:
        # set cpu affinity for process
        possible_cpus = proc.cpu_affinity()
        possible_cpus = possible_cpus[:min(len(possible_cpus), option['pcpu'])]
        pcpu = possible_cpus[mp_id % len(possible_cpus)]
        try:
            proc.cpu_affinity([pcpu])
        except Exception as e:
            pass

    profiler = None
    if option['pyinstrument']:
        profiler = Profiler()
        profiler.start()
    nominal_attributes = None
    if option['discritize_stream']:
        nominal_attributes = extract_nominal_attributes(option)
    stream, stream_concepts, length, classes = make_stream(option, cat_features_idx=nominal_attributes)

    if stream is None:
        return None
    UID = hash(tuple(f"{k}{str(v)}" for k, v in option.items()))
    window_size = option['window_size']

    # use an observation_gap of -1 as auto, take 1000 observations across the stream
    if option['observation_gap'] == -1:
        option['observation_gap'] = math.floor(length / 1000)
    
    # Find ground truth active concept
    stream_names = [c[3] for c in stream_concepts]
    unique_stream_names = get_unique_stream_names(stream_concepts)
    ground_truth_concept_init = get_ground_truth_concept_idx(0, stream_concepts, unique_stream_names)


    # print(type(classifier))
    output_path = option['base_output_path'] / option['experiment_name'] / \
        option['package_status'] / option['data_name'] / str(option['seed'])
    output_path.mkdir(parents=True, exist_ok=True)
    run_index = 0
    run_name = f"run_{UID}_{run_index}"
    log_dump_path = output_path / f"{run_name}.csv"
    arff_dump_path = output_path / f"{run_name}_arff.arff"
    moa_bat_path = output_path / f"{run_name}_bat.{'bat' if option['moa_type'] == 'bat' else 'sh'}"
    options_dump_path = output_path / f"{run_name}_options.txt"
    options_dump_path_partial = output_path / f"partial_{run_name}_options.txt"
    results_dump_path = output_path / f"results_{run_name}.txt"

    # Look for existing file using this name.
    # This will happen for all other option types, so
    # look for other runs with the same options.
    json_options = json.loads(json.dumps(option, cls=NpEncoder))

    def compare_options(A, B):
        for k in A:
            if k in ['log_name', 'package_status', 'base_output_path', 'cpu', 'pcpu']:
                continue
            if k not in A or k not in B:
                continue
            if A[k] != B[k]:
                return False
        return True
    other_runs = output_path.glob('*_options.txt')
    for other_run_path in other_runs:
        if 'partial' in other_run_path.stem:
            continue
        else:
            with other_run_path.open() as f:
                existing_options = json.load(f)
            if compare_options(existing_options, json_options):
                if option['pyinstrument']:
                    profiler.stop()
                return other_run_path

    while options_dump_path.exists() or options_dump_path_partial.exists():
        run_index += 1
        run_name = f"runother_{UID}_{run_index}"
        log_dump_path = output_path / f"{run_name}.csv"
        arff_dump_path = output_path / f"{run_name}_arff.arff"
        moa_bat_path = output_path / f"{run_name}_bat.{'bat' if option['moa_type'] == 'bat' else 'sh'}"
        options_dump_path = output_path / f"{run_name}_options.txt"
        options_dump_path_partial = output_path / \
            f"partial_{run_name}_options.txt"
        results_dump_path = output_path / f"results_{run_name}.txt"

    partial_log_size = 2500
    partial_logs = []
    partial_log_index = 0

    with options_dump_path_partial.open('w+') as f:
        json.dump(option, f, cls=NpEncoder)

    right = 0
    wrong = 0
    stream_names = [c[3] for c in stream_concepts]
    unique_stream_names = get_unique_stream_names(stream_concepts)

    monitoring_data = []
    monitoring_header = ('example', 'y', 'p', 'is_correct', 'right_sum', 'wrong_sum', 'overall_accuracy', 'active_model', 'ground_truth_concept', 'drift_occured', 'change_detected', 'model_evolution',
                         'active_state_active_similarity', 'active_state_buffered_similarity', 'all_state_buffered_similarity', 'all_state_active_similarity', 'feature_weights', 'concept_likelihoods', 'concept_priors', 'concept_priors_1h', 'concept_priors_2h', 'concept_posteriors', "adwin_likelihood_estimate", "adwin_posterior_estimate", "adwin_likelihood_estimate_background", "adwin_posterior_estimate_background", "state_relevance", "merges", 'deletions', 'cpu_time', 'ram_use', 'fingerprint_ram')
    feature_columns = []
    label_column = []
    for i in tqdm.tqdm(range(option['length'])):
        X, y = stream.next_sample()
        for xi, x in enumerate(X[0]):
            if xi >= len(feature_columns):
                feature_columns.append([])
            feature_columns[xi].append(x)
        label_column.append(y[0])
    start_time = time.process_time()


    # noise_rng = np.random.RandomState(option['seed'])
    noise_rng = np.random.default_rng(option['seed'])
    
    with arff_dump_path.open('w+') as f:
        print(f"@relation myRel", file=f)
        for xi in range(len(feature_columns)):
            print(f"@attribute x{xi} numeric", file=f)
        unique_labels = np.unique(label_column)
        unique_labels.sort()
                
        print(f"@attribute Class {{{','.join([str(x) for x in unique_labels])}}}", file=f)

        print(f"@data", file=f)
        row_items = zip(*feature_columns, label_column)
        for row in row_items:
            if option['noise'] > 0:
                noise_roll = noise_rng.rand()
                if noise_roll < option['noise']:
                    y = np.array([noise_rng.choice(classes)])
                    row = list(row)
                    row[-1] = y
            print(f"{','.join([str(x) for x in row])}", file=f)
    
    MOA_EVALUATORS = {
        'preq': 'EvaluatePrequential',
        'int':  'EvaluateInterleavedTestThenTrain',
    }

    MOA_LEARNERS = {
        'rcd': 'meta.RCD -l trees.HoeffdingTree -d EDDM',
        'arf': 'meta.AdaptiveRandomForest',
        'obag': 'meta.OzaBagAdwin',
        'ht': 'trees.HoeffdingTree',
        # 'ecpf': 'meta.ECPF -l trees.HoeffdingTree -d ADWINChangeDetector',
        'ecpf': 'meta.ECPF -l trees.HoeffdingTree',
        'cpf': 'meta.CPF -l trees.HoeffdingTree'
    }

    def get_learner_string(learner, concept_limit):
        learner_string = MOA_LEARNERS[learner]
        print(learner)
        if learner == 'rcd':
            if concept_limit != 0:
                concept_limit = max(0, concept_limit)
            concept_string = f"-c {concept_limit}" if concept_limit != None else f""
        elif learner == 'arf':
            if concept_limit != 0:
                concept_limit = max(1, concept_limit)
            concept_string = f"-s {concept_limit}" if concept_limit != None else f""
        else:
            concept_string = ""
        print(f'{learner_string} {concept_string}')
        return f'({learner_string} {concept_string})'

    def make_moa_command(stream_string, learner, concept_limit, evaluator, length, report_length, result_path, is_bat = True):
        return f'java -cp {"%" if is_bat else "$"}1\moa.jar -javaagent:{"%" if is_bat else "$"}1\sizeofag-1.0.4.jar moa.DoTask "{MOA_EVALUATORS[evaluator]} -l {get_learner_string(learner, concept_limit)} -s {stream_string} -e (BenPerformanceEvaluator -w {report_length})-i {length} -f {1}" > "{str(result_path)}"'

    def get_moa_stream_from_filename(arff_file_path):
        return f"(ArffFileStream -f ({str(arff_file_path)}))"

    def save_moa_bat(moa_command, filename, is_bat = True):

        print(f"{moa_command}\n")
        with open(filename, 'w') as f:
            if not is_bat:
                f.write(f"#!/bin/sh\n")
            f.write(f"{moa_command}\n")

    num_examples = len(label_column)
    classifier_type_is_moa = option['classifier'] not in ['dynse']
    if classifier_type_is_moa:
        stream_string = get_moa_stream_from_filename(arff_dump_path)
        moa_string = make_moa_command(
            stream_string,
            option['classifier'],
            option['mem_management'],
            'int',
            num_examples,
            75,
            log_dump_path,
            is_bat= option['moa_type'] == 'bat'
        )
        save_moa_bat(moa_string, str(moa_bat_path), option['moa_type'] == 'bat')
    elif option['classifier'] == 'dynse':
        is_bat= option['moa_type'] == 'bat'
        dynse_command = rf"java -javaagent:{'%' if is_bat else '$'}1\dynse\target\sizeofag-1.0.0.jar -jar {'%' if is_bat else '$'}1\dynse\target\dynse-0.2-jar-with-dependencies.jar -t -f {arff_dump_path} {log_dump_path}"
        save_moa_bat(dynse_command, str(moa_bat_path), option['moa_type'] == 'bat')
    else:
        raise ValueError(f"Classifier type {option['classifier']} not handled")

    moa_location = str(option['moa_path'])
    command = f'{str(moa_bat_path)} "{moa_location}"'
    print(command)
    if not option['moa_type'] == 'bat':
        subprocess.run(['chmod' ,'+x', str(moa_bat_path)])
        subprocess.run([str(moa_bat_path), moa_location])
    else:
        subprocess.run(command)


    df = pd.read_csv(log_dump_path)
    if df.shape[0] < 10:
        raise ValueError("Error in java process, need to delete bad output")

    ground_truth_concept = []
    drift_occured = []
    for i in range(df.shape[0]):
        # for c in stream_concepts:
        #     concept_start= c[0]
        #     if concept_start <= i < c[1]:
        #         ground_truth_concept_index = stream_names.index(c[3])
        ground_truth_concept_index = get_ground_truth_concept_idx(i, stream_concepts, unique_stream_names)
        ground_truth_concept.append(ground_truth_concept_index)
        d_o = False
        for c in stream_concepts[1:]:
            concept_start= c[0]
            if i == concept_start:
                d_o = True
        drift_occured.append(d_o)
    if 'system_concept' in df.columns:
        df['active_model'] = df['system_concept']
    if 'Change detected' in df.columns:
        df['change_detected'] = df['Change detected'].astype(str) == '1.0'
    df['ground_truth_concept'] = np.array(ground_truth_concept)
    df['drift_occured'] = np.array(drift_occured)
    df['deletions'] = np.array([""]*len(drift_occured))
    df['feature_weights'] = np.array([None]*len(drift_occured))
    if 'learning evaluation instances' in df.columns:
        df['example'] = df['learning evaluation instances']
    if 'evaluation time (cpu seconds)' in df.columns:
        df['cpu_time'] = df['evaluation time (cpu seconds)']
    if 'model serialized size (bytes)' in df.columns:
        df['ram_use'] = df['model serialized size (bytes)']
    elif 'memory' in df.columns:
        df['ram_use'] = df['memory']
    df['all_state_active_similarity'] = 0


    df.to_csv(log_dump_path, index=False)
    with options_dump_path.open('w+') as f:
        json.dump(option, f, cls=NpEncoder)

    for partial_log in partial_logs:
        partial_log.unlink()
    options_dump_path_partial.unlink()

    dump_results(option, df, log_dump_path, log_dump_path, results_dump_path, {}, df)
    
    if queue:
        queue.put(mp_id)

    return options_dump_path


if __name__ == "__main__":
    freeze_support()
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--datasets', default='cmc', nargs="*", type=str)
    my_parser.add_argument('--classifier', default='CC', nargs="*", type=str)
    my_parser.add_argument('--seeds', default=1, nargs="*", type=int)
    my_parser.add_argument('--seedaction', default="list",
                           type=str, choices=['new', 'list', 'reuse'])
    my_parser.add_argument('--datalocation', default="RawData", type=str)
    my_parser.add_argument('--outputlocation', default="output", type=str)
    my_parser.add_argument('--loglocation', default="experimentlog", type=str)
    my_parser.add_argument('--moalocation', default="moa", type=str)
    my_parser.add_argument('--moatype', default="bat", type=str)
    my_parser.add_argument('--experimentname', default="expDefault", type=str)
    my_parser.add_argument('--desc', default="", type=str)
    my_parser.add_argument('--optionslocation', default=None, type=str)
    my_parser.add_argument(
        '--fsmethod', default="fisher_overall", type=str, nargs="*")
    my_parser.add_argument('--fingerprintmethod',
                           default="descriptive", type=str, nargs="*")
    my_parser.add_argument('--fingerprintbins',
                           default=10, type=int, nargs="*")
    my_parser.add_argument('--logging', action='store_true')
    my_parser.add_argument('--discritize_stream', action='store_true')
    my_parser.add_argument('--minimal_output', action='store_true')
    my_parser.add_argument('--MAP_selection', action='store_true')
    my_parser.add_argument('--pyinstrument', action='store_true')
    my_parser.add_argument('--ficsum', action='store_true')
    my_parser.add_argument('--single', action='store_true')
    my_parser.add_argument('--lockpbar', action='store_false')
    my_parser.add_argument('--forcegitcheck', action='store_true')
    my_parser.add_argument('--experimentopts', action='store_true')
    my_parser.add_argument('--cpu', default=2, type=int)
    my_parser.add_argument('--setaffinity', action='store_true')
    my_parser.add_argument('--pcpu', default=-1, type=int)
    my_parser.add_argument('--poisson', default=6, type=int)
    my_parser.add_argument('--repeats', default=3, type=int)
    my_parser.add_argument('--concept_length', default=5000, type=int)
    my_parser.add_argument('--concept_max', default=6, type=int)
    my_parser.add_argument('--repeatproportion', default=1.0, type=float)
    my_parser.add_argument('--TMdropoff', default=1.0, type=float)
    my_parser.add_argument('--TMforward', default=1, type=int)
    my_parser.add_argument('--TMnoise', default=0.0, type=float)
    my_parser.add_argument('--drift_width', default=0, type=float, nargs="*")
    my_parser.add_argument('--noise', default=0, type=float, nargs="*")
    my_parser.add_argument('--conceptdifficulty', default=0, type=float, nargs="*")
    my_parser.add_argument('--maxrows', default=75000, type=int)
    my_parser.add_argument('--d_hard_concepts', default=3, type=int)
    my_parser.add_argument('--d_easy_concepts', default=1, type=int)
    my_parser.add_argument('--n_hard_concepts', default=15, type=int)
    my_parser.add_argument('--n_easy_concepts', default=15, type=int)
    my_parser.add_argument('--p_hard_concepts', default=0.5, type=float)
    my_parser.add_argument('--repository_cap', default=-1, nargs="*", type=int)
    my_parser.add_argument('--mem_management', default='rA', nargs="*", type=str)
    my_parser.add_argument('--sim', default='metainfo', nargs="*", type=str)
    my_parser.add_argument('--MIcalc', default='metainfo', nargs="*", type=str)
    my_parser.add_argument('--window_size', default=100, nargs="*", type=int)
    my_parser.add_argument('--sensitivity', default=0.05, nargs="*", type=float)
    my_parser.add_argument('--min_window_ratio', default=0.65, nargs="*", type=float)
    my_parser.add_argument('--fingerprint_grace_period', default=10, nargs="*", type=int)
    my_parser.add_argument('--state_grace_period_window_multiplier', default=10, nargs="*", type=int)
    my_parser.add_argument('--bypass_grace_period_threshold', default=0.2, nargs="*", type=float)
    my_parser.add_argument('--state_estimator_risk', default=0.5, nargs="*", type=float)
    my_parser.add_argument('--state_estimator_swap_risk', default=0.75, nargs="*", type=float)
    my_parser.add_argument('--minimum_concept_likelihood', default=0.005, nargs="*", type=float)
    my_parser.add_argument('--min_drift_likelihood_threshold', default=0.175, nargs="*", type=float)
    my_parser.add_argument('--min_estimated_posterior_threshold', default=0.2, nargs="*", type=float)
    my_parser.add_argument('--sim_gap', default=5, nargs="*", type=int)
    my_parser.add_argument('--fp_gap', default=15, nargs="*", type=int)
    # my_parser.add_argument('--fp_gap', default=6, nargs="*", type=int)
    my_parser.add_argument('--na_fp_gap', default=50, nargs="*", type=int)
    my_parser.add_argument('--ob_gap', default=5, nargs="*", type=int)
    my_parser.add_argument('--sim_stdevs', default=3, nargs="*", type=float)
    my_parser.add_argument(
        '--min_sim_stdev', default=0.015, nargs="*", type=float)
    my_parser.add_argument(
        '--max_sim_stdev', default=0.175, nargs="*", type=float)
    my_parser.add_argument(
        '--buffer_ratio', default=0.20, nargs="*", type=float)
    my_parser.add_argument(
        '--merge_threshold', default=0.95, nargs="*", type=float)
    my_parser.add_argument(
        '--background_state_prior_multiplier', default=0.4, nargs="*", type=float)
    my_parser.add_argument('--zero_prob_minimum', default=0.7, nargs="*", type=float)
    my_parser.add_argument('--multihop_penalty', default=0.7, nargs="*", type=float)
    my_parser.add_argument('--prev_state_prior', default=50, nargs="*", type=float)
    my_parser.add_argument('--no_merge', action='store_true')
    my_parser.add_argument('--optdetect', action='store_true')
    my_parser.add_argument('--optselect', action='store_true')
    my_parser.add_argument('--opthalf', action='store_true')
    my_parser.add_argument('--opthalflock', action='store_true')
    my_parser.add_argument('--shuffleconcepts', action='store_true')
    my_parser.add_argument('--save_feature_weights', action='store_true')
    my_parser.add_argument('--isources', nargs="*",
                           help="set sources to be ignored, from feature, f{i}, labels, predictions, errors, error_distances")
    my_parser.add_argument('--ifeatures', default=['IMF', 'MI', 'pacf'], nargs="*",
                           help="set features to be ignored, any meta-information feature")
    my_parser.add_argument('--classes', nargs='*', help='We try to detect classes automatically\
                                when the normalizer is set up, but sometimes this does not find\
                                rare classes. In this case, manually pass all clases in the dataset.')
    args = my_parser.parse_args()

    datasets = check_dataset_args(args)

    seeds = []
    num_seeds = 0
    base_seeds = []
    if args.seedaction == 'reuse':
        num_seeds = args.seeds if type(
            args.seeds) is not list else args.seeds[0]
        base_seeds = np.random.randint(0, 9999, size=num_seeds)
    if args.seedaction == 'new':
        num_seeds = args.seeds if type(
            args.seeds) is not list else args.seeds[0]
        seeds = np.random.randint(0, 9999, size=num_seeds)
    if args.seedaction == 'list':
        seeds = args.seeds if type(args.seeds) is list else [args.seeds]
        num_seeds = len(seeds)

    raw_data_path = pathlib.Path(args.datalocation).resolve()
    if not raw_data_path.exists():
        raise ValueError(f"Data location {raw_data_path} does not exist")
    for ds, ds_type in datasets:
        data_file_location = raw_data_path / ds_type / ds
        if not data_file_location.exists():
            if ds_type == 'Synthetic':
                data_file_location.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(
                    f"Data file {data_file_location} does not exist")
    base_output_path = pathlib.Path(args.outputlocation).resolve()
    if not base_output_path.exists():
        base_output_path.mkdir(parents=True)
    log_path = pathlib.Path(args.loglocation).resolve()
    if not log_path.exists():
        log_path.mkdir(parents=True)
    given_options = None
    if args.optionslocation is not None:
        options_path = pathlib.Path(args.optionslocation).resolve()
        if options_path.exists():
            with options_path.open() as f:
                given_options = json.load(f)
    

    desc_path = base_output_path / args.experimentname / "desc.txt"
    desc_ver = 1
    while desc_path.exists():
        desc_path = base_output_path / args.experimentname / f"desc_{desc_ver}.txt"
        desc_ver += 1

    desc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(desc_path, 'w+') as f:
        f.write(args.desc)

    log_name = f"{args.experimentname}-{time.time()}"
    if args.logging:
        logging.basicConfig(handlers=[RotatingFileHandler(
            f"{log_path}/{log_name}.log", maxBytes=500000000, backupCount=100)], level=logging.INFO)
    with (log_path / f"e{log_name}.txt").open('w+') as f:
        json.dump(args.__dict__, f)

    package_status = get_package_status(force=args.forcegitcheck)

    if given_options:
        option_set = given_options
    else:
        option_set = []
        dataset_options = list(datasets)

        classifier_options = args.classifier if type(args.classifier) is list else [args.classifier]
        similarity_options = args.sim if type(args.sim) is list else [args.sim]
        MIcalc = args.MIcalc if type(args.MIcalc) is list else [args.MIcalc]
        fs_options = args.fsmethod if type(
            args.fsmethod) is list else [args.fsmethod]
        fingerprint_options = args.fingerprintmethod if type(
            args.fingerprintmethod) is list else [args.fingerprintmethod]
        fingerprint_bins_options = args.fingerprintbins if type(
            args.fingerprintbins) is list else [args.fingerprintbins]
        window_size_options = args.window_size if type(
            args.window_size) is list else [args.window_size]
        sensitivity_options = args.sensitivity if type(
            args.sensitivity) is list else [args.sensitivity]
        min_window_ratio_options = args.min_window_ratio if type(
            args.min_window_ratio) is list else [args.min_window_ratio]
        fingerprint_grace_period_options = args.fingerprint_grace_period if type(
            args.fingerprint_grace_period) is list else [args.fingerprint_grace_period]
        state_grace_period_window_multiplier_options = args.state_grace_period_window_multiplier if type(
            args.state_grace_period_window_multiplier) is list else [args.state_grace_period_window_multiplier]
        bypass_grace_period_threshold_options = args.bypass_grace_period_threshold if type(
            args.bypass_grace_period_threshold) is list else [args.bypass_grace_period_threshold]
        state_estimator_risk_options = args.state_estimator_risk if type(
            args.state_estimator_risk) is list else [args.state_estimator_risk]
        state_estimator_swap_risk_options = args.state_estimator_swap_risk if type(
            args.state_estimator_swap_risk) is list else [args.state_estimator_swap_risk]
        minimum_concept_likelihood_options = args.minimum_concept_likelihood if type(
            args.minimum_concept_likelihood) is list else [args.minimum_concept_likelihood]
        min_drift_likelihood_threshold_options = args.min_drift_likelihood_threshold if type(
            args.min_drift_likelihood_threshold) is list else [args.min_drift_likelihood_threshold]
        min_estimated_posterior_threshold_options = args.min_estimated_posterior_threshold if type(
            args.min_estimated_posterior_threshold) is list else [args.min_estimated_posterior_threshold]
        sim_gap_options = args.sim_gap if type(
            args.sim_gap) is list else [args.sim_gap]
        fp_gap_options = args.fp_gap if type(
            args.fp_gap) is list else [args.fp_gap]
        na_fp_gap_options = args.na_fp_gap if type(
            args.na_fp_gap) is list else [args.na_fp_gap]
        ob_gap_options = args.ob_gap if type(
            args.ob_gap) is list else [args.ob_gap]
        sim_stdevs_options = args.sim_stdevs if type(
            args.sim_stdevs) is list else [args.sim_stdevs]
        min_sim_stdev_options = args.min_sim_stdev if type(
            args.min_sim_stdev) is list else [args.min_sim_stdev]
        max_sim_stdev_options = args.max_sim_stdev if type(
            args.max_sim_stdev) is list else [args.max_sim_stdev]
        buffer_ratio_options = args.buffer_ratio if type(
            args.buffer_ratio) is list else [args.buffer_ratio]
        merge_threshold_options = args.merge_threshold if type(
            args.merge_threshold) is list else [args.merge_threshold]
        background_state_prior_multiplier_options = args.background_state_prior_multiplier if type(
            args.background_state_prior_multiplier) is list else [args.background_state_prior_multiplier]
        zero_prob_minimum_options = args.zero_prob_minimum if type(
            args.zero_prob_minimum) is list else [args.zero_prob_minimum]
        multihop_penalty_options = args.multihop_penalty if type(
            args.multihop_penalty) is list else [args.multihop_penalty]
        prev_state_prior_options = args.prev_state_prior if type(
            args.prev_state_prior) is list else [args.prev_state_prior]
        drift_width_options = args.drift_width if type(
            args.drift_width) is list else [args.drift_width]
        noise_options = args.noise if type(
            args.noise) is list else [args.noise]
        conceptdifficulty_options = args.conceptdifficulty if type(
            args.conceptdifficulty) is list else [args.conceptdifficulty]
        repository_cap_options = args.repository_cap if type(
            args.repository_cap) is list else [args.repository_cap]
        mem_management_options = args.mem_management if type(
            args.mem_management) is list else [args.mem_management]
        
        if args.pcpu == -1:
            args.pcpu = args.cpu

        if not args.experimentopts:
            classifier_options = list(itertools.product(classifier_options,
                                                        similarity_options,
                                                        MIcalc,
                                                        fs_options,
                                                        fingerprint_options,
                                                        fingerprint_bins_options,
                                                        window_size_options,
                                                        sensitivity_options,
                                                        min_window_ratio_options,
                                                        fingerprint_grace_period_options,
                                                        state_grace_period_window_multiplier_options,
                                                        bypass_grace_period_threshold_options,
                                                        state_estimator_risk_options,
                                                        state_estimator_swap_risk_options,
                                                        minimum_concept_likelihood_options,
                                                        min_drift_likelihood_threshold_options,
                                                        min_estimated_posterior_threshold_options,
                                                        sim_gap_options,
                                                        fp_gap_options,
                                                        na_fp_gap_options,
                                                        ob_gap_options,
                                                        sim_stdevs_options,
                                                        min_sim_stdev_options,
                                                        max_sim_stdev_options,
                                                        buffer_ratio_options,
                                                        merge_threshold_options,
                                                        background_state_prior_multiplier_options,
                                                        zero_prob_minimum_options,
                                                        multihop_penalty_options,
                                                        prev_state_prior_options,
                                                        drift_width_options,
                                                        noise_options,
                                                        conceptdifficulty_options,
                                                        repository_cap_options,
                                                        mem_management_options,
                                                        ))
        else:
            classifier_options = list(itertools.product(classifier_options,
                                                        fingerprint_bins_options,
                                                        window_size_options,
                                                        sensitivity_options,
                                                        min_window_ratio_options,
                                                        fingerprint_grace_period_options,
                                                        state_grace_period_window_multiplier_options,
                                                        bypass_grace_period_threshold_options,
                                                        state_estimator_risk_options,
                                                        state_estimator_swap_risk_options,
                                                        minimum_concept_likelihood_options,
                                                        min_drift_likelihood_threshold_options,
                                                        min_estimated_posterior_threshold_options,
                                                        sim_gap_options,
                                                        fp_gap_options,
                                                        na_fp_gap_options,
                                                        ob_gap_options,
                                                        sim_stdevs_options,
                                                        min_sim_stdev_options,
                                                        max_sim_stdev_options,
                                                        buffer_ratio_options,
                                                        merge_threshold_options,
                                                        background_state_prior_multiplier_options,
                                                        zero_prob_minimum_options,
                                                        multihop_penalty_options,
                                                        prev_state_prior_options,
                                                        drift_width_options,
                                                        noise_options,
                                                        conceptdifficulty_options,
                                                        repository_cap_options,
                                                        mem_management_options,
                                                        ))

        for ds_name, ds_type in dataset_options:
            # If we are reusing, find out what seeds are already in use
            # otherwise, make a new one.
            if args.seedaction == 'reuse':
                seed_location = raw_data_path / ds_type / ds_name / "seeds"
                if not seed_location.exists():
                    seeds = []
                else:
                    seeds = [int(str(f.stem))
                             for f in seed_location.iterdir() if f.is_dir()]
                for i in range(num_seeds):
                    if i < len(seeds):
                        continue
                    seeds.append(base_seeds[i])
                if len(seeds) < 1:
                    raise ValueError(
                        f"Reuse seeds selected by no seeds exist for data set {ds_name}")

            for seed in seeds:
                if not args.experimentopts:
                    for (classifier_option, sim_opt, MIcalc, fs_opt, fingerprint_opt, fingerprint_bins_opt, ws_opt,                                                         sensitivity_options,
                                                        min_window_ratio_options,
                                                        fingerprint_grace_period_options,
                                                        state_grace_period_window_multiplier_options,
                                                        bypass_grace_period_threshold_options,
                                                        state_estimator_risk_options,
                                                        state_estimator_swap_risk_options,
                                                        minimum_concept_likelihood_options,
                                                        min_drift_likelihood_threshold_options,
                                                        min_estimated_posterior_threshold_options, sim_gap_opt, fp_gap_opt, na_fp_gap_opt, ob_gap_opt, sim_std_opt, min_sim_opt, max_sim_opt, br_opt, merge_threshold_options, background_state_prior_multiplier_options, zero_prob_minimum_options, multihop_penalty_options, prev_state_prior_options, drift_width_opt, noise_opt, conceptdifficulty_opt, repository_cap_opt, mem_management_opt) in classifier_options:
                        option = {
                            'classifier': classifier_option,
                            'base_output_path': base_output_path,
                            'raw_data_path': raw_data_path,
                            'moa_path': args.moalocation,
                            'moa_type': args.moatype,
                            'data_name': ds_name,
                            'data_type': ds_type,
                            'max_rows': args.maxrows,
                            'seed': seed,
                            'seed_action': args.seedaction,
                            'package_status': package_status,
                            'log_name': log_name,
                            'pyinstrument': args.pyinstrument,
                            'FICSUM': args.ficsum,
                            'minimal_output': args.minimal_output,
                            'MAP_selection': args.MAP_selection,
                            'setaffinity': args.setaffinity,
                            'pcpu': args.pcpu,
                            'cpu': args.cpu,
                            'experiment_name': args.experimentname,
                            'repeats': args.repeats,
                            'concept_max': args.concept_max,
                            'concept_length': args.concept_length,
                            'repeatproportion': args.repeatproportion,
                            'TMdropoff': args.TMdropoff,
                            'TMforward': args.TMforward,
                            'TMnoise': args.TMnoise,
                            'drift_width': drift_width_opt,
                            'noise': noise_opt,
                            'conceptdifficulty': conceptdifficulty_opt,
                            'framework': "system",
                            'isources': args.isources,
                            'ifeatures': args.ifeatures,
                            'optdetect': args.optdetect,
                            'optselect': args.optselect,
                            'opthalf': args.opthalf,
                            'opthalflock': args.opthalflock,
                            'save_feature_weights': args.save_feature_weights,
                            'shuffleconcepts': args.shuffleconcepts,
                            'similarity_option': sim_opt,
                            'MI_calc': MIcalc,
                            'window_size': ws_opt,
                            'sensitivity': sensitivity_options,
                            'min_window_ratio': min_window_ratio_options,
                            'fingerprint_grace_period': fingerprint_grace_period_options,
                            'state_grace_period_window_multiplier': state_grace_period_window_multiplier_options,
                            'bypass_grace_period_threshold': bypass_grace_period_threshold_options,
                            'state_estimator_risk': state_estimator_risk_options,
                            'state_estimator_swap_risk': state_estimator_swap_risk_options,
                            'minimum_concept_likelihood': minimum_concept_likelihood_options,
                            'min_drift_likelihood_threshold': min_drift_likelihood_threshold_options,
                            'min_estimated_posterior_threshold': min_estimated_posterior_threshold_options,
                            'similarity_gap': sim_gap_opt,
                            'fp_gap': fp_gap_opt,
                            'nonactive_fp_gap': na_fp_gap_opt,
                            'observation_gap': ob_gap_opt,
                            'take_observations': ob_gap_opt != 0,
                            'similarity_stdev_thresh': sim_std_opt,
                            'similarity_stdev_min': min_sim_opt,
                            'similarity_stdev_max': max_sim_opt,
                            'buffer_ratio': br_opt,
                            "merge_threshold": merge_threshold_options,
                            "background_state_prior_multiplier": background_state_prior_multiplier_options,
                            "zero_prob_minimum": zero_prob_minimum_options,
                            "multihop_penalty": multihop_penalty_options,
                            "prev_state_prior": prev_state_prior_options,
                            "correlation_merge": not args.no_merge,
                            'fs_method': fs_opt,
                            'fingerprint_method': fingerprint_opt,
                            'fingerprint_bins': fingerprint_bins_opt,
                            'd_hard_concepts': args.d_hard_concepts,
                            'd_easy_concepts': args.d_easy_concepts,
                            'n_hard_concepts': args.n_hard_concepts,
                            'n_easy_concepts': args.n_easy_concepts,
                            'p_hard_concepts': args.p_hard_concepts,
                            'repository_cap': repository_cap_opt,
                            'discritize_stream': args.discritize_stream,
                            'mem_management': mem_management_opt,
                            'poisson': args.poisson,
                        }
                        stream, stream_concepts, length, classes = make_stream(
                            option)
                        option_set.append(option)
                else:
                    for (classifier_option, fingerprint_bins_opt, ws_opt, sensitivity_options,
                            min_window_ratio_options,
                            fingerprint_grace_period_options,
                            state_grace_period_window_multiplier_options,
                            bypass_grace_period_threshold_options,
                            state_estimator_risk_options,
                            state_estimator_swap_risk_options,
                            minimum_concept_likelihood_options,
                            min_drift_likelihood_threshold_options,
                            min_estimated_posterior_threshold_options, sim_gap_opt, fp_gap_opt, na_fp_gap_opt, ob_gap_opt, sim_std_opt, min_sim_opt, max_sim_opt, br_opt, merge_threshold_options, background_state_prior_multiplier_options, zero_prob_minimum_options, multihop_penalty_options, prev_state_prior_options, drift_width_opt, noise_opt, conceptdifficulty_opt, repository_cap_opt, mem_management_opt) in classifier_options:
                        for exp_fingerprint, exp_fsmethod, sim_opt in [('cache', 'fisher_overall', 'metainfo'), ('cache', 'fisher', 'metainfo'), ('cache', 'CacheMIHy', 'metainfo'), ('cachehistogram', 'Cachehistogram_MI', 'metainfo'), ('cachesketch', 'sketch_MI', 'metainfo'), ('cachesketch', 'sketch_covMI', 'metainfo'), ('cachesketch', 'sketch_MI', 'sketch'), ('cachesketch', 'sketch_covMI', 'sketch')]:
                            # Only need to run default and fisher on one bin size, as it doesn't do anything
                            if exp_fsmethod in ['default', 'fisher', 'fisher_overall'] and fingerprint_bins_opt != fingerprint_bins_options[0]:
                                continue
                            option = {
                                'classifier_options': classifier_option,
                                'base_output_path': base_output_path,
                                'raw_data_path': raw_data_path,
                                'moa_path': args.moalocation,
                                'moa_type': args.moatype,
                                'data_name': ds_name,
                                'data_type': ds_type,
                                'max_rows': args.maxrows,
                                'seed': seed,
                                'seed_action': args.seedaction,
                                'package_status': package_status,
                                'log_name': log_name,
                                'pyinstrument': args.pyinstrument,
                                'FICSUM': args.ficsum,
                                'minimal_output': args.minimal_output,
                                'MAP_selection': args.MAP_selection,
                                'setaffinity': args.setaffinity,
                                'pcpu': args.pcpu,
                                'cpu': args.cpu,
                                'experiment_name': args.experimentname,
                                'repeats': args.repeats,
                                'concept_max': args.concept_max,
                                'concept_length': args.concept_length,
                                'repeatproportion': args.repeatproportion,
                                'TMdropoff': args.TMdropoff,
                                'TMforward': args.TMforward,
                                'TMnoise': args.TMnoise,
                                'drift_width': drift_width_opt,
                                'noise': noise_opt,
                                'conceptdifficulty': conceptdifficulty_opt,
                                'framework': "system",
                                'isources': args.isources,
                                'ifeatures': args.ifeatures,
                                'optdetect': args.optdetect,
                                'optselect': args.optselect,
                                'opthalf': args.opthalf,
                                'opthalflock': args.opthalflock,
                                'save_feature_weights': args.save_feature_weights,
                                'shuffleconcepts': args.shuffleconcepts,
                                'similarity_option': sim_opt,
                                'MI_calc': MIcalc[0],
                                'window_size': ws_opt,
                                'sensitivity': sensitivity_options,
                                'min_window_ratio': min_window_ratio_options,
                                'fingerprint_grace_period': fingerprint_grace_period_options,
                                'state_grace_period_window_multiplier': state_grace_period_window_multiplier_options,
                                'bypass_grace_period_threshold': bypass_grace_period_threshold_options,
                                'state_estimator_risk': state_estimator_risk_options,
                                'state_estimator_swap_risk': state_estimator_swap_risk_options,
                                'minimum_concept_likelihood': minimum_concept_likelihood_options,
                                'min_drift_likelihood_threshold': min_drift_likelihood_threshold_options,
                                'min_estimated_posterior_threshold': min_estimated_posterior_threshold_options,
                                'similarity_gap': sim_gap_opt,
                                'fp_gap': fp_gap_opt,
                                'nonactive_fp_gap': na_fp_gap_opt,
                                'observation_gap': ob_gap_opt,
                                'take_observations': ob_gap_opt != 0,
                                'similarity_stdev_thresh': sim_std_opt,
                                'similarity_stdev_min': min_sim_opt,
                                'similarity_stdev_max': max_sim_opt,
                                'buffer_ratio': br_opt,
                                "merge_threshold": merge_threshold_options,
                                "background_state_prior_multiplier": background_state_prior_multiplier_options,
                                "zero_prob_minimum": zero_prob_minimum_options,
                                "multihop_penalty": multihop_penalty_options,
                                "prev_state_prior": prev_state_prior_options,
                                "correlation_merge": not args.no_merge,
                                'fs_method': exp_fsmethod,
                                'fingerprint_method': exp_fingerprint,
                                'fingerprint_bins': fingerprint_bins_opt,
                                'd_hard_concepts': args.d_hard_concepts,
                                'd_easy_concepts': args.d_easy_concepts,
                                'n_hard_concepts': args.n_hard_concepts,
                                'n_easy_concepts': args.n_easy_concepts,
                                'p_hard_concepts': args.p_hard_concepts,
                                'repository_cap': repository_cap_opt,
                                'discritize_stream': args.discritize_stream,
                                'mem_management': mem_management_opt,
                                'poisson': args.poisson,
                            }
                            stream, stream_concepts, length, classes = make_stream(
                                option)
                            option_set.append(option)
    with (log_path / f"e{log_name}_option_set.txt").open('w+') as f:
        json.dump(option_set, f, cls=NpEncoder)
    if args.single:
        run_files = []
        for o in tqdm.tqdm(option_set, total=len(option_set), position=1, desc="Experiment", leave=True):
            # print(o)
            run_files.append(process_option((o, None, None)))
    else:
        manager = mp.Manager()

        lock = manager.RLock() if args.lockpbar else None
        tqdm.tqdm.set_lock(lock)
        print(lock)
        pool = mp.Pool(processes=args.cpu, maxtasksperchild=1)
        run_files = []
        queue = manager.Queue()
        for c_index in range(args.cpu):
            queue.put(c_index)
        overall_prog_bar = tqdm.tqdm(total=len(
            option_set), position=1, desc="Experiment", leave=True, miniters=1, lock_args=(True, 0.01), ascii=True)
        for result in pool.imap_unordered(func=process_option, iterable=((o, lock, queue) for o in option_set), chunksize=1):
            run_files.append(result)
            if lock or True:
                l = lock.acquire(blocking=False)
                if l:
                    overall_prog_bar.update(n=1)
                    overall_prog_bar.refresh(lock_args=(False))
                    lock.release()
            else:
                overall_prog_bar.update(n=1)
        # print(run_files)
        pool.close()
    with (log_path / f"e{log_name}_run_files.txt").open('w+') as f:
        json.dump(run_files, f, cls=NpEncoder)
