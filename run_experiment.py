import argparse
import importlib
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
from re import L
import sys, gc
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
# from pympler.classtracker import ClassTracker
# import pympler
# from pympler import muppy
# from pympler import summary
# import tracemalloc
# tracemalloc.start(10)

from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForestClassifier
from skmultiflow.meta.dynamic_weighted_majority import DynamicWeightedMajorityClassifier

from PhDCode.Classifier.hoeffding_tree_shap import \
    HoeffdingTreeSHAPClassifier
from PhDCode.Classifier.CNDPM.base_classifier_wrapper import CNDPMMLPClassifier
from PhDCode.Classifier.select_classifier import SELeCTClassifier
from PhDCode.Classifier.FiCSUM import FiCSUMClassifier
from PhDCode.Classifier.lower_bound_classifier import BoundClassifier
from PhDCode.Classifier.wrapper_classifier import WrapperClassifier
from PhDCode.Classifier.advantage_wrapper import AdvantageWrapperClassifier
from PhDCode.Classifier.airstream_wrapper import AirstreamWrapperClassifier

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
    if len(stream_concepts) <= 1 and option['GT_context_location'] is None:
        input("You are running a dataset with one concept. This usually means you are trying to run a dataset, but you havn't passed a context. This means your data will be treated as stationary and shuffled. If this is NOT intended, please rerun this command and pass a context or the str zero to indicate unknown context.")
    gt_context_values = None
    if option['GT_context_location']:
        
        print(option['GT_context_location'])
        if option['GT_context_location'] == 'zero':
            gt_context_values = np.zeros(length)
        else:
            try:
                gt_context_path = pathlib.Path(option['GT_context_location'])
                gt_context_df = pd.read_csv(gt_context_path.open('r'))
                print(gt_context_df.values)
                if gt_context_df.values.shape[1] > 1:
                    raise ValueError("Two values for context, make sure the index is not saved in the csv")
                flat_contexts = gt_context_df.values.ravel()
                print(flat_contexts)
                gt_context_values = np.tile(gt_context_df.values.flatten(), option['repeats'])
                print(gt_context_values)
                print(length)
                assert len(gt_context_values) >= length
            except Exception as e:
                raise e
    if stream is None:
        return None
    UID = hash(tuple(f"{k}{str(v)}" for k, v in option.items()))
    window_size = option['window_size']

    learner = lambda : HoeffdingTreeSHAPClassifier(nominal_attributes=nominal_attributes)

    # for i in range(10):
    #     X, y = stream.next_sample()
    #     print(X, y)
    #     print(y[0])
    # print(classes)
    # print(nominal_attributes)
    # exit()

    # use an observation_gap of -1 as auto, take 1000 observations across the stream
    if option['observation_gap'] == -1:
        option['observation_gap'] = math.floor(length / 1000)
    
    # Find ground truth active concept
    stream_names = [c[3] for c in stream_concepts]
    unique_stream_names = get_unique_stream_names(stream_concepts)
    # print(stream_names)
    # print(unique_stream_names)
    # ground_truth_concept_init = None
    # for c in stream_concepts:
    #     concept_start = c[0]
    #     if concept_start <= 0 < c[1]:
    #         ground_truth_concept_init = unique_stream_names.index(c[3])
    ground_truth_concept_init = get_ground_truth_concept_idx(0, stream_concepts, unique_stream_names, gt_context_values)

    if option['classifier'].lower() == 'cc':
        classifier = SELeCTClassifier(
            learner=learner,
            window_size=option['window_size'],
            active_head_monitor_gap=option['similarity_gap'],
            fingerprint_update_gap=option['fp_gap'],
            non_active_fingerprint_update_gap=option['nonactive_fp_gap'],
            observation_gap=option['observation_gap'],
            sim_measure=option['similarity_option'],
            ignore_sources=option['isources'],
            ignore_features=option['ifeatures'],
            similarity_num_stdevs=option['similarity_stdev_thresh'],
            similarity_min_stdev=option['similarity_stdev_min'],
            similarity_max_stdev=option['similarity_stdev_max'],
            buffer_ratio=option['buffer_ratio'],
            feature_selection_method=option['fs_method'],
            fingerprint_method=option['fingerprint_method'],
            fingerprint_bins=option['fingerprint_bins'],
            MI_calc=option['MI_calc'],
            sensitivity = option['sensitivity'],
            min_window_ratio = option['min_window_ratio'],
            fingerprint_grace_period = option['fingerprint_grace_period'],
            state_grace_period_window_multiplier = option['state_grace_period_window_multiplier'],
            bypass_grace_period_threshold = option['bypass_grace_period_threshold'],
            state_estimator_risk = option['state_estimator_risk'],
            state_estimator_swap_risk = option['state_estimator_swap_risk'],
            minimum_concept_likelihood = option['minimum_concept_likelihood'],
            min_drift_likelihood_threshold = option['min_drift_likelihood_threshold'],
            min_estimated_posterior_threshold = option['min_estimated_posterior_threshold'],
            correlation_merge=option['correlation_merge'],
            merge_threshold=option['merge_threshold'],
            background_state_prior_multiplier=option['background_state_prior_multiplier'],
            zero_prob_minimum=option['zero_prob_minimum'],
            multihop_penalty=option['multihop_penalty'],
            prev_state_prior=option['prev_state_prior'],
            MAP_selection=option['MAP_selection'],
            poisson=option['poisson'],
            repository_max=option['repository_max'],
            valuation_policy=option['valuation_policy'],
            )
    elif option['classifier'].lower() == 'cc_cndpm_base':
        classifier = SELeCTClassifier(
            learner=lambda : CNDPMMLPClassifier(train_weight_required_for_evolution=option['train_weight_required_for_evolution'], batch_learning=option['batch_learning'], cndpm_use_large=option['cndpm_use_large']),
            window_size=option['window_size'],
            active_head_monitor_gap=option['similarity_gap'],
            fingerprint_update_gap=option['fp_gap'],
            non_active_fingerprint_update_gap=option['nonactive_fp_gap'],
            observation_gap=option['observation_gap'],
            sim_measure=option['similarity_option'],
            ignore_sources=option['isources'],
            ignore_features=option['ifeatures'],
            similarity_num_stdevs=option['similarity_stdev_thresh'],
            similarity_min_stdev=option['similarity_stdev_min'],
            similarity_max_stdev=option['similarity_stdev_max'],
            buffer_ratio=option['buffer_ratio'],
            feature_selection_method=option['fs_method'],
            fingerprint_method=option['fingerprint_method'],
            fingerprint_bins=option['fingerprint_bins'],
            MI_calc=option['MI_calc'],
            sensitivity = option['sensitivity'],
            min_window_ratio = option['min_window_ratio'],
            fingerprint_grace_period = option['fingerprint_grace_period'],
            state_grace_period_window_multiplier = option['state_grace_period_window_multiplier'],
            bypass_grace_period_threshold = option['bypass_grace_period_threshold'],
            state_estimator_risk = option['state_estimator_risk'],
            state_estimator_swap_risk = option['state_estimator_swap_risk'],
            minimum_concept_likelihood = option['minimum_concept_likelihood'],
            min_drift_likelihood_threshold = option['min_drift_likelihood_threshold'],
            min_estimated_posterior_threshold = option['min_estimated_posterior_threshold'],
            correlation_merge=option['correlation_merge'],
            merge_threshold=option['merge_threshold'],
            background_state_prior_multiplier=option['background_state_prior_multiplier'],
            zero_prob_minimum=option['zero_prob_minimum'],
            multihop_penalty=option['multihop_penalty'],
            prev_state_prior=option['prev_state_prior'],
            MAP_selection=option['MAP_selection'],
            poisson=option['poisson'],
            repository_max=option['repository_max'],
            valuation_policy=option['valuation_policy'],
            )
    
    elif option['classifier'].lower() == 'cc_basicprior':
        classifier = SELeCTClassifier(
            learner=learner,
            window_size=option['window_size'],
            active_head_monitor_gap=option['similarity_gap'],
            fingerprint_update_gap=option['fp_gap'],
            non_active_fingerprint_update_gap=option['nonactive_fp_gap'],
            observation_gap=option['observation_gap'],
            sim_measure=option['similarity_option'],
            ignore_sources=option['isources'],
            ignore_features=option['ifeatures'],
            similarity_num_stdevs=option['similarity_stdev_thresh'],
            similarity_min_stdev=option['similarity_stdev_min'],
            similarity_max_stdev=option['similarity_stdev_max'],
            buffer_ratio=option['buffer_ratio'],
            feature_selection_method=option['fs_method'],
            fingerprint_method=option['fingerprint_method'],
            fingerprint_bins=option['fingerprint_bins'],
            MI_calc=option['MI_calc'],
            sensitivity = option['sensitivity'],
            min_window_ratio = option['min_window_ratio'],
            fingerprint_grace_period = option['fingerprint_grace_period'],
            state_grace_period_window_multiplier = option['state_grace_period_window_multiplier'],
            bypass_grace_period_threshold = option['bypass_grace_period_threshold'],
            state_estimator_risk = option['state_estimator_risk'],
            state_estimator_swap_risk = option['state_estimator_swap_risk'],
            minimum_concept_likelihood = option['minimum_concept_likelihood'],
            min_drift_likelihood_threshold = option['min_drift_likelihood_threshold'],
            min_estimated_posterior_threshold = option['min_estimated_posterior_threshold'],
            correlation_merge=option['correlation_merge'],
            merge_threshold=option['merge_threshold'],
            background_state_prior_multiplier=1.0,
            zero_prob_minimum=0.9999,
            multihop_penalty=0.0,
            prev_state_prior=0.0,
            MAP_selection=option['MAP_selection'],
            poisson=option['poisson'],
            repository_max=option['repository_max'],
            valuation_policy=option['valuation_policy'],
            )
    elif option['classifier'].lower() == 'cc_map':
        classifier = SELeCTClassifier(
            learner=learner,
            window_size=option['window_size'],
            active_head_monitor_gap=option['similarity_gap'],
            fingerprint_update_gap=option['fp_gap'],
            non_active_fingerprint_update_gap=option['nonactive_fp_gap'],
            observation_gap=option['observation_gap'],
            sim_measure=option['similarity_option'],
            ignore_sources=option['isources'],
            ignore_features=option['ifeatures'],
            similarity_num_stdevs=option['similarity_stdev_thresh'],
            similarity_min_stdev=option['similarity_stdev_min'],
            similarity_max_stdev=option['similarity_stdev_max'],
            buffer_ratio=option['buffer_ratio'],
            feature_selection_method=option['fs_method'],
            fingerprint_method=option['fingerprint_method'],
            fingerprint_bins=option['fingerprint_bins'],
            MI_calc=option['MI_calc'],
            sensitivity = option['sensitivity'],
            min_window_ratio = option['min_window_ratio'],
            fingerprint_grace_period = option['fingerprint_grace_period'],
            state_grace_period_window_multiplier = option['state_grace_period_window_multiplier'],
            bypass_grace_period_threshold = option['bypass_grace_period_threshold'],
            state_estimator_risk = option['state_estimator_risk'],
            state_estimator_swap_risk = option['state_estimator_swap_risk'],
            minimum_concept_likelihood = option['minimum_concept_likelihood'],
            min_drift_likelihood_threshold = option['min_drift_likelihood_threshold'],
            min_estimated_posterior_threshold = option['min_estimated_posterior_threshold'],
            correlation_merge=option['correlation_merge'],
            merge_threshold=option['merge_threshold'],
            background_state_prior_multiplier=option['background_state_prior_multiplier'],
            zero_prob_minimum=option['zero_prob_minimum'],
            multihop_penalty=option['multihop_penalty'],
            prev_state_prior=option['prev_state_prior'],
            MAP_selection=True,
            poisson=option['poisson'],
            repository_max=option['repository_max'],
            valuation_policy=option['valuation_policy'],
            )
    elif option['classifier'].lower() == 'cc_nomerge':
        classifier = SELeCTClassifier(
            learner=learner,
            window_size=option['window_size'],
            active_head_monitor_gap=option['similarity_gap'],
            fingerprint_update_gap=option['fp_gap'],
            non_active_fingerprint_update_gap=option['nonactive_fp_gap'],
            observation_gap=option['observation_gap'],
            sim_measure=option['similarity_option'],
            ignore_sources=option['isources'],
            ignore_features=option['ifeatures'],
            similarity_num_stdevs=option['similarity_stdev_thresh'],
            similarity_min_stdev=option['similarity_stdev_min'],
            similarity_max_stdev=option['similarity_stdev_max'],
            buffer_ratio=option['buffer_ratio'],
            feature_selection_method=option['fs_method'],
            fingerprint_method=option['fingerprint_method'],
            fingerprint_bins=option['fingerprint_bins'],
            MI_calc=option['MI_calc'],
            sensitivity = option['sensitivity'],
            min_window_ratio = option['min_window_ratio'],
            fingerprint_grace_period = option['fingerprint_grace_period'],
            state_grace_period_window_multiplier = option['state_grace_period_window_multiplier'],
            bypass_grace_period_threshold = option['bypass_grace_period_threshold'],
            state_estimator_risk = option['state_estimator_risk'],
            state_estimator_swap_risk = option['state_estimator_swap_risk'],
            minimum_concept_likelihood = option['minimum_concept_likelihood'],
            min_drift_likelihood_threshold = option['min_drift_likelihood_threshold'],
            min_estimated_posterior_threshold = option['min_estimated_posterior_threshold'],
            correlation_merge=False,
            merge_threshold=1.0,
            background_state_prior_multiplier=option['background_state_prior_multiplier'],
            zero_prob_minimum=option['zero_prob_minimum'],
            multihop_penalty=option['multihop_penalty'],
            prev_state_prior=option['prev_state_prior'],
            MAP_selection=option['MAP_selection'],
            poisson=option['poisson'],
            repository_max=option['repository_max'],
            valuation_policy=option['valuation_policy'],
            )
    elif option['classifier'].lower() == 'ficsum':
        classifier = FiCSUMClassifier(
            learner=learner,
            window_size=option['window_size'],
            similarity_gap=option['similarity_gap'],
            fingerprint_update_gap=option['fp_gap'],
            non_active_fingerprint_update_gap=option['nonactive_fp_gap'],
            observation_gap=option['observation_gap'],
            sim_measure=option['similarity_option'],
            ignore_sources=option['isources'],
            ignore_features=option['ifeatures'],
            similarity_num_stdevs=option['similarity_stdev_thresh'],
            similarity_min_stdev=option['similarity_stdev_min'],
            similarity_max_stdev=option['similarity_stdev_max'],
            buffer_ratio=option['buffer_ratio'],
            feature_selection_method=option['fs_method'],
            fingerprint_method=option['fingerprint_method'],
            fingerprint_bins=option['fingerprint_bins'],
            MI_calc=option['MI_calc'],
            min_window_ratio = option['min_window_ratio'],
            poisson=option['poisson'],
            repository_max=option['repository_max'],
            valuation_policy=option['valuation_policy'],)
    elif option['classifier'].lower() == 'advantage':
        classifier = AdvantageWrapperClassifier(
            learner=learner,
            window_size=option['window_size'],
            concept_limit=option['repository_max'],
            memory_management=option['valuation_policy'],
            poisson=option['poisson'],)
    elif option['classifier'].lower() == 'airstream':
        classifier = AirstreamWrapperClassifier(
            learner=learner,
            window_size=option['window_size'],
            concept_limit=-1,
            memory_management="rA",
            allow_backtrack=True,
            poisson=option['poisson'],
            repository_max=option['repository_max'],
            valuation_policy=option['valuation_policy'],)
    elif option['classifier'].lower() == 'airstream_nobacktrack':
        classifier = AirstreamWrapperClassifier(
            learner=learner,
            window_size=option['window_size'],
            concept_limit=-1,
            memory_management="rA",
            allow_backtrack=False,
            poisson=option['poisson'],
            repository_max=option['repository_max'],
            valuation_policy=option['valuation_policy'],)
    # elif option['lower_bound']:
    elif option['classifier'].lower() == 'lower_bound':
        option['optselect'] = True
        option['optdetect'] = True

        classifier = BoundClassifier(
            learner=learner,
            window_size=option['window_size'],
            bounds="lower",
            init_concept_id=ground_truth_concept_init,
            poisson=option['poisson'],
        )
    # elif option['upper_bound']:
    elif option['classifier'].lower() == 'upper_bound':
        option['optselect'] = True
        option['optdetect'] = True
        classifier = BoundClassifier(
            learner=learner,
            window_size=option['window_size'],
            bounds="upper",
            init_concept_id=ground_truth_concept_init,
            poisson=option['poisson'],
        )
    elif option['classifier'].lower() == 'upper_bound_cndpm_base':
        option['optselect'] = True
        option['optdetect'] = True
        classifier = BoundClassifier(
            learner=lambda : CNDPMMLPClassifier(train_weight_required_for_evolution=option['train_weight_required_for_evolution'], batch_learning=option['batch_learning'], cndpm_use_large=option['cndpm_use_large']),
            window_size=option['window_size'],
            bounds="upper",
            init_concept_id=ground_truth_concept_init,
            poisson=option['poisson'],
        )
    elif option['classifier'].lower() == 'middle_bound':
        option['optselect'] = True
        option['optdetect'] = True
        classifier = BoundClassifier(
            learner=learner,
            window_size=option['window_size'],
            bounds="middle",
            init_concept_id=ground_truth_concept_init,
            poisson=option['poisson'],
        )
    elif option['classifier'].lower() == 'arf':
        classifier = WrapperClassifier(
        learner=AdaptiveRandomForestClassifier,
        init_concept_id=ground_truth_concept_init
        )
    elif option['classifier'].lower() == 'dwm':
        classifier = WrapperClassifier(
        learner=DynamicWeightedMajorityClassifier,
        init_concept_id=ground_truth_concept_init
        )
    elif option['classifier'].lower() == 'cndpm':
        from PhDCode.Classifier.CNDPM.CNDPM_wrapper import CNDPMWrapperClassifier
        classifier = CNDPMWrapperClassifier(learner=learner,
            window_size=option['window_size'],
            concept_limit=option['repository_max'],
            memory_management=option['valuation_policy'],
            poisson=option['poisson'],
            use_prior=option['cndpm_use_prior'],
            batch_learning=option['batch_learning'],
            cndpm_use_large=option['cndpm_use_large'])
    else:
        raise ValueError(f"classifier option '{option['classifier']}' is not valid")

    # print(type(classifier))
    output_path = option['base_output_path'] / option['experiment_name'] / \
        option['package_status'] / option['data_name'] / str(option['seed'])
    output_path.mkdir(parents=True, exist_ok=True)
    run_index = 0
    run_name = f"run_{UID}_{run_index}"
    log_dump_path = output_path / f"{run_name}.csv"
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
    logging.info(option)
    logging.info(classifier.__dict__)
    start_time = time.process_time()

    def memory_usage_psutil():
        # return the memory usage in MB
        mem = proc.memory_info()[0] / float(2 ** 20)
        return mem
    start_mem = memory_usage_psutil()
    # tracker = ClassTracker()
    if not hasattr(classifier, 'fingerprint_type'):
        classifier.fingerprint_type = {}
        classifier.active_state = -1
    # tracker.track_class(classifier.fingerprint_type)
    # try:
    #     tracker.create_snapshot()
    # except:
    #     pass
    ram_use = 0
    aff = proc.cpu_affinity() if option['setaffinity'] else -1

    progress_bar = None
    if mp_id < 100:
        if lock:
            l = lock.acquire(timeout=10, blocking=True)
            if l:
                progress_bar = tqdm.tqdm(total=option['length'], position=min(mp_id+2, 11), desc=f"CPU {aff} - Worker {mp_id} {str(UID)[:4]}...", leave=False, mininterval=3.0, lock_args=(True, 0.01), ascii=True)
                lock.release()
        else:
            progress_bar = tqdm.tqdm(total=option['length'], position=min(mp_id+2, 11), desc=f"CPU {aff} - Worker {mp_id} {str(UID)[:4]}...", leave=False, mininterval=3.0, ascii=True)
    pbar_updates = 0

    # noise_rng = np.random.RandomState(option['seed'])
    noise_rng = np.random.default_rng(option['seed'])
    last_gt = ground_truth_concept_init
    for i in range(option['length']):
        # logging.info(f"At {i}")
        current_merges = None
        observation_monitoring = {}
        observation_monitoring['example'] = i
        X, y = stream.next_sample()
        if option['noise'] > 0:
            noise_roll = noise_rng.random()
            if noise_roll < option['noise']:
                y = np.array([noise_rng.choice(classes)])

        observation_monitoring['y'] = int(y[0])
        p = classifier.predict(X)
        # print(p, y)
        observation_monitoring['p'] = int(p[0])
        e = y[0] == p[0]
        observation_monitoring['is_correct'] = int(e)
        right += y[0] == p[0]
        observation_monitoring['right_sum'] = right
        wrong += y[0] != p[0]
        observation_monitoring['wrong_sum'] = wrong
        observation_monitoring['overall_accuracy'] = right / (right + wrong)
        if option['minimal_output']:
            observation_monitoring['overall_accuracy'] = round(observation_monitoring['overall_accuracy'] * 1000000) / 1000000

        # # Find ground truth active concept
        # ground_truth_concept_index = None
        # for c in stream_concepts:
        #     concept_start = c[0]
        #     if concept_start <= i < c[1]:
        #         ground_truth_concept_index = stream_names.index(c[3])
        
        ground_truth_concept_index = get_ground_truth_concept_idx(i, stream_concepts, unique_stream_names, gt_context_values)

        # Control parameters
        drift_occured = False
        concept_drift = False
        concept_drift_target = None
        concept_transition = False
        classifier.manual_control = False
        classifier.force_stop_learn_fingerprint = False
        classifier.force_transition = False
        classifier.force_transition_only = False

        # Controls for giving classifier perfect information
        # num_concepts_to_lock = 6
        # if option['optdetect'] or ((option['opthalf'] or option['opthalflock']) and i < option['concept_length'] * num_concepts_to_lock):
        #     classifier.force_transition_only = True

        # For control, find if there was a ground truth
        # drift, with some delay.
        drift_occured = last_gt != ground_truth_concept_index
        # for c in stream_concepts[1:]:
        #     concept_start = c[0]
        #     if i == concept_start:
        for c in stream_concepts[1:]:
            concept_start = c[0]
            if i == concept_start + window_size + 10:
                concept_drift = True
                if option['classifier'].lower() in ['upper_bound', 'upper_bound_cndpm_base']:
                    concept_drift_target = stream_names.index(c[3])

        # Controls for giving classifier perfect information
        # Only used for upper bound classifier
        if option['classifier'].lower() in ['upper_bound', 'upper_bound_cndpm_base']:
            if type(classifier) != BoundClassifier:
                raise ValueError("Using perfect information for wrong classifier!")
            if concept_drift and (option['optdetect'] or ((option['opthalf'] or option['opthalflock']) and i < option['concept_length'] * num_concepts_to_lock)):
                classifier.manual_control = True
                classifier.force_transition = True
                classifier.force_stop_learn_fingerprint = True
                if option['optselect'] or ((option['opthalf'] or option['opthalflock']) and i < option['concept_length'] * num_concepts_to_lock):
                    classifier.force_transition_to = concept_drift_target
            if concept_transition:
                classifier.force_stop_learn_fingerprint = True

        classifier.partial_fit(X, y, classes=classes)
        # Collect monitoring data for storage.
        current_active_model = classifier.active_state
        observation_monitoring['active_model'] = current_active_model
        observation_monitoring['ground_truth_concept'] = int(ground_truth_concept_index) if ground_truth_concept_index is not None else ground_truth_concept_index
        observation_monitoring['drift_occured'] = int(drift_occured)
        observation_monitoring['change_detected'] = int(classifier.detected_drift)
        observation_monitoring['model_evolution'] = classifier.get_active_state(
        ).current_evolution

        if option['classifier'] == 'CC':
            if classifier.monitor_active_state_active_similarity is not None and not option['minimal_output']:
                observation_monitoring['active_state_active_similarity'] = classifier.monitor_active_state_active_similarity
            else:
                observation_monitoring['active_state_active_similarity'] = -1 if not option['minimal_output'] else None

            if classifier.monitor_active_state_buffered_similarity is not None and not option['minimal_output']:
                observation_monitoring['active_state_buffered_similarity'] = classifier.monitor_active_state_buffered_similarity
            else:
                observation_monitoring['active_state_buffered_similarity'] = -1 if not option['minimal_output'] else None

            buffered_data = classifier.monitor_all_state_buffered_similarity
            if buffered_data is not None and not option['minimal_output']:
                buffered_accuracy, buffered_stats, buffered_window, buffered_similarities = buffered_data
                concept_similarities = [
                    (int(k), f"{v:.4f}") for k, v in buffered_similarities.items() if k != 'active']
                concept_similarities.sort(key=lambda x: x[0])
                observation_monitoring['all_state_buffered_similarity'] = ';'.join(
                    [str(x[1]) for x in concept_similarities])
            else:
                observation_monitoring['all_state_buffered_similarity'] = -1 if not option['minimal_output'] else None

            weights = classifier.monitor_feature_selection_weights
            if weights is not None and option['save_feature_weights'] and not option['minimal_output']:
                observation_monitoring['feature_weights'] = ';'.join(
                    [f"{s}{f}:{v}" for s, f, v in weights])
            else:
                observation_monitoring['feature_weights'] = -1 if not option['minimal_output'] else None

            # if not option['FICSUM']:
            concept_likelihoods = classifier.concept_likelihoods
            if concept_likelihoods is not None and not option['minimal_output']:
                observation_monitoring['concept_likelihoods'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in concept_likelihoods.items()])
            else:
                observation_monitoring['concept_likelihoods'] = -1 if not option['minimal_output'] else None

            # concept_likelihoods_smoothed = classifier.concept_likelihoods_smoothed
            # if concept_likelihoods_smoothed is not None:
            #     observation_monitoring['concept_likelihoods_smoothed'] = ';'.join(
            #         [f"{cid}:{v}" for cid, v in concept_likelihoods_smoothed.items()])
            # else:
            #     observation_monitoring['concept_likelihoods_smoothed'] = -1
            # concept_posteriors_smoothed = classifier.concept_posteriors_smoothed
            # if concept_posteriors_smoothed is not None:
            #     observation_monitoring['concept_posteriors_smoothed'] = ';'.join(
            #         [f"{cid}:{v}" for cid, v in concept_posteriors_smoothed.items()])
            # else:
            #     observation_monitoring['concept_posteriors_smoothed'] = -1

            concept_priors = classifier.concept_priors
            if concept_priors is not None and not option['minimal_output']:
                observation_monitoring['concept_priors'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in concept_priors.items()])
            else:
                observation_monitoring['concept_priors'] = -1 if not option['minimal_output'] else None

            concept_priors_2h = classifier.concept_priors_2h
            if concept_priors_2h is not None and not option['minimal_output']:
                observation_monitoring['concept_priors_2h'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in concept_priors_2h.items()])
            else:
                observation_monitoring['concept_priors_2h'] = -1 if not option['minimal_output'] else None
            concept_priors_1h = classifier.concept_priors_1h
            if concept_priors_1h is not None and not option['minimal_output']:
                observation_monitoring['concept_priors_1h'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in concept_priors_1h.items()])
            else:
                observation_monitoring['concept_priors_1h'] = -1 if not option['minimal_output'] else None
            concept_posteriors = classifier.concept_posteriors
            if concept_posteriors is not None and not option['minimal_output']:
                observation_monitoring['concept_posteriors'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in concept_posteriors.items()])
            else:
                observation_monitoring['concept_posteriors'] = -1 if not option['minimal_output'] else None

            adwin_likelihood_estimate = {}
            # adwin_likelihood_estimate[-1] = classifier.background_state.get_estimated_likelihood() if classifier.background_state else -1
            adwin_likelihood_estimate.update({i:s.get_estimated_likelihood() for i,s in classifier.state_repository.items()})
            if adwin_likelihood_estimate is not None and not option['minimal_output']:
                observation_monitoring['adwin_likelihood_estimate'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in adwin_likelihood_estimate.items()])
            else:
                observation_monitoring['adwin_likelihood_estimate'] = "0:0" if not option['minimal_output'] else None
            observation_monitoring['adwin_likelihood_estimate_background'] = classifier.background_state.get_estimated_likelihood() if classifier.background_state and not option['minimal_output'] else 0
            
            adwin_posterior_estimate = {}
            # adwin_posterior_estimate[-1] = classifier.background_state.get_estimated_posterior() if classifier.background_state else -1
            adwin_posterior_estimate.update({i:s.get_estimated_posterior() for i,s in classifier.state_repository.items()})
            if adwin_posterior_estimate is not None and not option['minimal_output']:
                observation_monitoring['adwin_posterior_estimate'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in adwin_posterior_estimate.items()])
            else:
                observation_monitoring['adwin_posterior_estimate'] = "0:0" if not option['minimal_output'] else None
            observation_monitoring['adwin_posterior_estimate_background'] = classifier.background_state.get_estimated_posterior() if classifier.background_state and not option['minimal_output'] else 0

        state_relevance = None
        if hasattr(classifier, "state_relevance"):
            state_relevance = classifier.state_relevance
        elif option['classifier'].lower() == 'cc':
            state_relevance = classifier.concept_posteriors
        
        if state_relevance is not None and not option['minimal_output']:
            observation_monitoring['state_relevance'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in state_relevance.items()])
        else:
            observation_monitoring['state_relevance'] = -1 if not option['minimal_output'] else None

        
        # Save merges to progress on change
        merges = classifier.merges if hasattr(classifier, "merges") else {}
        if merges != current_merges:
            observation_monitoring['merges'] = ';'.join([f"{from_id}:{to_id}" for from_id, to_id in merges.items()])
            current_merges = merges

        deletions = classifier.deletions if hasattr(classifier, "deletions") else []
        observation_monitoring['deletions'] = ';'.join([str(v) for v in deletions]) if len(deletions) > 0 else ""
        


        all_state_active_similarity = classifier.monitor_all_state_active_similarity
        if all_state_active_similarity is not None and not option['minimal_output']:
            active_accuracy, active_stats, active_fingerprints, active_window, active_similarities = all_state_active_similarity
            concept_similarities = [
                (int(k), f"{v:.4f}") for k, v in active_similarities.items() if k != 'active']
            concept_similarities.sort(key=lambda x: x[0])
            observation_monitoring['all_state_active_similarity'] = ';'.join(
                [str(x[1]) for x in concept_similarities])
        else:
            observation_monitoring['all_state_active_similarity'] = -1 if not option['minimal_output'] else None

        observation_monitoring['detected_drift'] = classifier.detected_drift
        observation_monitoring['concept_drift'] = concept_drift

        observation_monitoring['cpu_time'] = time.process_time() - start_time
        observation_monitoring['ram_use'] = ram_use
        # last_memory_snap = tracker.snapshots[-1]
        # if hasattr(last_memory_snap, "classes"):
        #     observation_monitoring['fingerprint_ram'] = last_memory_snap.classes[-1]['avg']
        # else:
        observation_monitoring['fingerprint_ram'] = 0.0

        monitoring_data.append(observation_monitoring)
        last_gt = ground_truth_concept_index
        dump_start = time.process_time()
        if len(monitoring_data) >= partial_log_size:
            # try:
            #     tracker.create_snapshot()
            # except:
            #     pass
            # print("Starting Mem print:")
            # print(f"Classifier: {actualsize(classifier)}, C2: {actualsize(classifier.classifier)}, fsm: {actualsize(classifier.classifier.fsm)}, stream: {actualsize(stream)}")
            ram_use = memory_usage_psutil() - start_mem
            log_dump_path_partial = output_path / \
                f"partial_{run_name}_{partial_log_index}.csv"
            df = pd.DataFrame(monitoring_data, columns=monitoring_header)
            df.to_csv(log_dump_path_partial, index=False)
            partial_log_index += 1
            partial_logs.append(log_dump_path_partial)
            monitoring_data = []
            df = None

            if hasattr(classifier, 'reset_stats'):
                classifier.reset_stats(rem_state_log=True)
        dump_end = time.process_time()
        dump_time = dump_end - dump_start
        start_time -= dump_time

        # try to aquire the lock to update progress bar.
        # We don't care too much so use a short timeout!
        pbar_updates += 1
        if progress_bar:
            if lock:
                if pbar_updates >= 250:
                    l = lock.acquire(blocking=False)
                    if l:
                        progress_bar.update(n=pbar_updates)
                        pbar_updates = 0
                        progress_bar.refresh(lock_args=(False))
                        lock.release()
            else:
                progress_bar.update(n=1)



    df = None
    for partial_log in partial_logs:
        if df is None:
            df = pd.read_csv(partial_log)
        else:
            next_log = pd.read_csv(partial_log)
            df = df.append(next_log)
    if df is None:
        df = pd.DataFrame(monitoring_data, columns=monitoring_header)
    else:
        df = df.append(pd.DataFrame(
            monitoring_data, columns=monitoring_header))
    df = df.reset_index(drop=True)
    df.to_csv(log_dump_path, index=False)
    with options_dump_path.open('w+') as f:
        json.dump(option, f, cls=NpEncoder)

    for partial_log in partial_logs:
        partial_log.unlink()
    options_dump_path_partial.unlink()

    dump_results(option, df, log_dump_path, log_dump_path, results_dump_path, classifier.merges if hasattr(classifier, "merges") else {}, df)
    if option['pyinstrument']:
        profiler.stop()
        res = profiler.output_text(unicode=True, color=True)
        print(res)
        with open("profile.txt", 'w+') as f:
            f.write(res)
    
    # if progress_bar:
    #     if lock or True:
    #         l = lock.acquire(timeout=10, blocking=True)
    #         if l:
    #             progress_bar.close()
    #             lock.release()
    #     else:
            # progress_bar.close()
    
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
                           type=str, choices=['new', 'list', 'reuse', 'default', 'half', 'full'])
    my_parser.add_argument('--datalocation', default="RawData", type=str)
    my_parser.add_argument('--outputlocation', default="output", type=str)
    my_parser.add_argument('--loglocation', default="experimentlog", type=str)
    my_parser.add_argument('--experimentname', default="expDefault", type=str)
    my_parser.add_argument('--desc', default="", type=str)
    my_parser.add_argument('--optionslocation', default=None, type=str)
    my_parser.add_argument('--GT_context_location', default=None, type=str)
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
    my_parser.add_argument('--train_weight_required_for_evolution', default=100, type=float, nargs="*")
    my_parser.add_argument('--conceptdifficulty', default=100, type=float, nargs="*")
    my_parser.add_argument('--maxrows', default=75000, type=int)
    my_parser.add_argument('--d_hard_concepts', default=3, type=int)
    my_parser.add_argument('--d_easy_concepts', default=1, type=int)
    my_parser.add_argument('--n_hard_concepts', default=15, type=int)
    my_parser.add_argument('--n_easy_concepts', default=15, type=int)
    my_parser.add_argument('--p_hard_concepts', default=0.5, type=float)
    my_parser.add_argument('--repository_max', default=-1, nargs="*", type=int)
    my_parser.add_argument('--valuation_policy', default='rA', nargs="*", type=str)
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
    my_parser.add_argument('--batch_learning', action='store_true')
    my_parser.add_argument('--cndpm_dont_use_prior', action='store_false')
    my_parser.add_argument('--cndpm_use_large', action='store_true')
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
    if args.seedaction == 'full':
        seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 45]
        num_seeds = len(seeds)
    if args.seedaction == 'half':
        seeds = [1, 22, 33, 44, 55, 66, 77, 88, 99, 10, 111, 122, 133, 144, 155, 166, 177, 188, 199, 20,
                211, 222, 233, 244, 255, 266, 277, 288, 299, 30, 311, 322]
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
        train_weight_required_for_evolution_options = args.train_weight_required_for_evolution if type(
            args.train_weight_required_for_evolution) is list else [args.train_weight_required_for_evolution]
        repository_max_options = args.repository_max if type(
            args.repository_max) is list else [args.repository_max]
        valuation_policy_options = args.valuation_policy if type(
            args.valuation_policy) is list else [args.valuation_policy]
        
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
                                                        train_weight_required_for_evolution_options,
                                                        repository_max_options,
                                                        valuation_policy_options,
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
                                                        train_weight_required_for_evolution_options,
                                                        repository_max_options,
                                                        valuation_policy_options,
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
                                                        min_estimated_posterior_threshold_options, sim_gap_opt, fp_gap_opt, na_fp_gap_opt, ob_gap_opt, sim_std_opt, min_sim_opt, max_sim_opt, br_opt, merge_threshold_options, background_state_prior_multiplier_options, zero_prob_minimum_options, multihop_penalty_options, prev_state_prior_options, drift_width_opt, noise_opt, conceptdifficulty_opt, train_weight_required_for_evolution_opt, repository_max_opt, valuation_policy_opt) in classifier_options:
                        option = {
                            'classifier': classifier_option,
                            'base_output_path': base_output_path,
                            'raw_data_path': raw_data_path,
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
                            'train_weight_required_for_evolution': train_weight_required_for_evolution_opt,
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
                            'repository_max': repository_max_opt,
                            'discritize_stream': args.discritize_stream,
                            'valuation_policy': valuation_policy_opt,
                            'poisson': args.poisson,
                            'GT_context_location': args.GT_context_location,
                            'batch_learning': args.batch_learning,
                            'cndpm_use_prior': args.cndpm_dont_use_prior,
                            'cndpm_use_large': args.cndpm_use_large,
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
                            min_estimated_posterior_threshold_options, sim_gap_opt, fp_gap_opt, na_fp_gap_opt, ob_gap_opt, sim_std_opt, min_sim_opt, max_sim_opt, br_opt, merge_threshold_options, background_state_prior_multiplier_options, zero_prob_minimum_options, multihop_penalty_options, prev_state_prior_options, drift_width_opt, noise_opt, conceptdifficulty_opt, train_weight_required_for_evolution_opt, repository_max_opt, valuation_policy_opt) in classifier_options:
                        for exp_fingerprint, exp_fsmethod, sim_opt in [('cache', 'fisher_overall', 'metainfo'), ('cache', 'fisher', 'metainfo'), ('cache', 'CacheMIHy', 'metainfo'), ('cachehistogram', 'Cachehistogram_MI', 'metainfo'), ('cachesketch', 'sketch_MI', 'metainfo'), ('cachesketch', 'sketch_covMI', 'metainfo'), ('cachesketch', 'sketch_MI', 'sketch'), ('cachesketch', 'sketch_covMI', 'sketch')]:
                            # Only need to run default and fisher on one bin size, as it doesn't do anything
                            if exp_fsmethod in ['default', 'fisher', 'fisher_overall'] and fingerprint_bins_opt != fingerprint_bins_options[0]:
                                continue
                            option = {
                                'classifier_options': classifier_option,
                                'base_output_path': base_output_path,
                                'raw_data_path': raw_data_path,
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
                                'train_weight_required_for_evolution': train_weight_required_for_evolution_opt,
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
                                'repository_max': repository_max_opt,
                                'discritize_stream': args.discritize_stream,
                                'valuation_policy': valuation_policy_opt,
                                'poisson': args.poisson,
                                'GT_context_location': args.GT_context_location,
                                'batch_learning': args.batch_learning,
                                'cndpm_use_prior': args.cndpm_dont_use_prior,
                                'cndpm_use_large': args.cndpm_use_large,
                            }
                            stream, stream_concepts, length, classes = make_stream(
                                option)
                            option_set.append(option)
    with (log_path / f"e{log_name}_option_set.txt").open('w+') as f:
        json.dump(option_set, f, cls=NpEncoder)
    if args.single:
        run_files = []
        # try:
        for o in tqdm.tqdm(option_set, total=len(option_set), position=1, desc="Experiment", leave=True):
            # print(o)
            run_files.append(process_option((o, None, None)))
        # except:
        #     all_objects = muppy.get_objects() 
        #     sum1 = summary.summarize(all_objects)   
        #     summary.print_(sum1) 
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
