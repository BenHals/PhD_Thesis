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

from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForestClassifier
from skmultiflow.meta.dynamic_weighted_majority import DynamicWeightedMajorityClassifier

from PhDCode.Classifier.hoeffding_tree_shap import \
    HoeffdingTreeSHAPClassifier
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

import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style():
    sns.set_context('paper')
    plt.rcParams["font.family"] = "Times New Roman"

def actualsize(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size

class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        elif isinstance(obj, pathlib.Path):
            return str(obj)

        return json.JSONEncoder.default(self, obj)

# Load synthetic concepts (up to concept_max) and turn into a stream
# Concepts are placed in order (repeats) number of times.
# The idea is to use half the stream to learn concepts, then half
# to test for similarity, so 2 repeats to learn concepts then
# test on 2.


def make_stream(options, cat_features_idx=None):
    # print(options['conceptdifficulty'])
    num_concepts = options["concept_max"]
    boost_first_occurence = False
    if options['data_type'] == "Synthetic":
        difficulty_groups = [(options['conceptdifficulty'] if options['conceptdifficulty'] > 0 else None, num_concepts)]
        if options["data_name"][:3] == 'LM_':
            # Stands for Long Mixture dataset,
            # In this case, we have a mix of concepts with different difficulties
            # so we pass a tuple with (difficulty, number of cocnepts, number of appearences) for each group
            n_hard_concepts = math.floor(num_concepts * options['p_hard_concepts'])
            difficulty_groups = [(options['d_hard_concepts'], n_hard_concepts), (options['d_easy_concepts'], num_concepts - n_hard_concepts)]
            boost_first_occurence = True
        # print(difficulty_groups)
        concepts = load_synthetic_concepts(options['data_name'],
                                           options['seed'],
                                           raw_data_path=options['raw_data_path'] / 'Synthetic',
                                           difficulty_groups=difficulty_groups)
    else:
        # If no context location is set, we are injecting context in constructing the dataset
        # Otherwise, we keep data as it is (no shuffling etc) to preserve the context
        inject_context = options['GT_context_location'] is None
        concepts = load_real_concepts(options['data_name'],
                                      options['seed'],
                                      nrows=options['max_rows'],
                                      raw_data_path=options['raw_data_path'] / 'Real',
                                      sort_examples=True,
                                      inject_context=inject_context)
    
    # print(len(concepts))
    # print(options['concept_max'])
    stream_concepts, length = get_inorder_concept_ranges(concepts, concept_length=options['concept_length'], seed=options['seed'], repeats=options['repeats'], concept_max=options[
                                                         'concept_max'], repeat_proportion=options['repeatproportion'], shuffle=options['data_type'] != "Synthetic" or options['shuffleconcepts'],
                                                         dropoff=options['TMdropoff'], nforward=options['TMforward'], noise=options['TMnoise'], boost_first_occurence=boost_first_occurence)
    options['length'] = length
    try:
        stream = AbruptDriftStream(
            stream_concepts,  length, random_state=options['seed'], width=None if options['drift_width'] == 0 else options['drift_width'],
            cat_features_idx=cat_features_idx)
    except Exception as e:
        return None, None, None, None
    all_classes = stream._get_target_values()
    return stream, stream_concepts, length, list(all_classes)

def extract_nominal_attributes(options):
    stream, stream_concepts, length, n_classes = make_stream(options)
    data = []
    for i in range(500):
        X, y = stream.next_sample()
        # print(X, y)
        data.append(X[0])
    data = np.array(data)
    # print(data)
    data_df = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    nominal_attributes = []
    for c_i,c in enumerate(data_df.columns):
        init_vals = data_df[c].values[:500]
        n_u = len(np.unique(init_vals))
        if n_u <= 10:
            # print(f"Factoizing {c}")
            # print(pd.factorize(data_df[c])[0].shape)
            data_df[c] = pd.factorize(data_df[c])[0]
            nominal_attributes.append(c_i)

    return nominal_attributes

@contextmanager
def aquire_lock(lock, timeout=-1):
    l = lock.acquire(timeout=timeout, blocking=True)
    try:
        yield l
    finally:
        if l:
            lock.release()

def get_package_status(force=False):
    data = []
    for package in ["PhDCode"]:
        try:
            loc = str(importlib.util.find_spec(
                package).submodule_search_locations[0])
        except Exception as e:
            try:
                loc = str(importlib.util.find_spec(
                    package).submodule_search_locations._path[0])
            except:
                namespace = importlib.util.find_spec(
                    package).submodule_search_locations
                loc = str(namespace).split('[')[1].split(']')[0]
                loc = loc.split(',')[0]
                loc = loc.replace("'", "")
        loc = str(pathlib.Path(loc).resolve())
        try:
            commit = subprocess.check_output(
                ["git", "describe", "--always"], cwd=loc).strip().decode()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=loc).strip().decode()
            changes = subprocess.call(["git", "diff", "--quiet"], cwd=loc)
            changes_cached = subprocess.call(
                ["git", "diff", "--cached", "--quiet"])
        except:
            commit = "NOGITFOUND"
            branch = "NA"
            changes = None
        if changes and not force:
            print(f"{package} has uncommitted files: {changes}")
            input(
                "Are you sure you want to run with uncommitted code? Press any button to continue...")
        package_data = f"{package}-{branch}-{commit}"
        data.append(package_data)
    return '_'.join(data)


def get_drift_point_accuracy(log, follow_length=250):
    if not 'drift_occured' in log.columns or not 'is_correct' in log.columns:
        return 0, 0, 0, 0
    dpl = log.index[log['drift_occured'] == 1].tolist()
    dpl = dpl[1:]
    if len(dpl) == 0:
        return 0, 0, 0, 0

    following_drift = np.unique(np.concatenate(
        [np.arange(i, min(i+follow_length+1, len(log))) for i in dpl]))
    filtered = log.iloc[following_drift]
    num_close = filtered.shape[0]
    accuracy, kappa, kappa_m, kappa_t = get_performance(filtered)
    return accuracy, kappa, kappa_m, kappa_t

def get_driftdetect_point_accuracy(log, follow_length=250):
    if not 'change_detected' in log.columns:
        return 0, 0, 0, 0
    if not 'drift_occured' in log.columns:
        return 0, 0, 0, 0
    dpl = log.index[log['change_detected'] == 1].tolist()
    drift_indexes = log.index[log['drift_occured'] == 1].tolist()
    if len(dpl) < 1:
        return 0, 0, 0, 0
    if len(drift_indexes) < 1:
        return 0, 0, 0, 0
    following_drift = np.unique(np.concatenate(
        [np.arange(i, min(i+1000+1, len(log))) for i in drift_indexes]))
    following_detect = np.unique(np.concatenate(
        [np.arange(i, min(i+follow_length+1, len(log))) for i in dpl]))
    following_both = np.intersect1d(
        following_detect, following_drift, assume_unique=True)
    filtered = log.iloc[following_both]
    num_close = filtered.shape[0]
    if num_close == 0:
        return 0, 0, 0, 0
    accuracy, kappa, kappa_m, kappa_t = get_performance(filtered)
    return accuracy, kappa, kappa_m, kappa_t

def get_performance(log):
    sum_correct = log['is_correct'].sum()
    num_observations = log.shape[0]
    accuracy = sum_correct / num_observations
    values, counts = np.unique(log['y'], return_counts=True)
    majority_class = values[np.argmax(counts)]
    majority_correct = log.loc[log['y'] == majority_class]
    num_majority_correct = majority_correct.shape[0]
    majority_accuracy = num_majority_correct / num_observations
    if majority_accuracy < 1:
        kappa_m = (accuracy - majority_accuracy) / (1 - majority_accuracy)
    else:
        kappa_m = 0
    temporal_filtered = log['y'].shift(1, fill_value=0.0)
    temporal_correct = log['y'] == temporal_filtered
    temporal_accuracy = temporal_correct.sum() / num_observations
    kappa_t = (accuracy - temporal_accuracy) / (1 - temporal_accuracy)

    our_counts = Counter()
    gt_counts = Counter()
    for v in values:
        our_counts[v] = log.loc[log['p'] == v].shape[0]
        gt_counts[v] = log.loc[log['y'] == v].shape[0]

    expected_accuracy = 0
    for cat in values:
        expected_accuracy += (gt_counts[cat]
                                * our_counts[cat]) / num_observations
    expected_accuracy /= num_observations
    if expected_accuracy < 1:
        kappa = (accuracy - expected_accuracy) / (1 - expected_accuracy)
    else:
        kappa = 0

    return accuracy, kappa, kappa_m, kappa_t

def get_recall_precision(log, model_column="active_model"):
    ground_truth = log['ground_truth_concept'].fillna(
        method='ffill').astype(int).values
    system = log[model_column].fillna(method='ffill').astype(int).values
    gt_values, gt_total_counts = np.unique(
        ground_truth, return_counts=True)
    sys_values, sys_total_counts = np.unique(system, return_counts=True)
    matrix = np.array([ground_truth, system]).transpose()
    recall_values = {}
    precision_values = {}
    gt_results = {}
    sys_results = {}
    overall_results = {
        'Max Recall': 0,
        'Max Precision': 0,
        'Precision for Max Recall': 0,
        'Recall for Max Precision': 0,
        'GT_mean_f1': 0,
        'GT_mean_recall': 0,
        'GT_mean_precision': 0,
        'MR by System': 0,
        'MP by System': 0,
        'PMR by System': 0,
        'RMP by System': 0,
        'MODEL_mean_f1': 0,
        'MODEL_mean_recall': 0,
        'MODEL_mean_precision': 0,
        'Num Good System Concepts': 0,
        'GT_to_MODEL_ratio': 0,
    }
    gt_proportions = {}
    sys_proportions = {}

    for gt_i, gt in enumerate(gt_values):
        gt_total_count = gt_total_counts[gt_i]
        gt_mask = matrix[matrix[:, 0] == gt]
        sys_by_gt_values, sys_by_gt_counts = np.unique(
            gt_mask[:, 1], return_counts=True)
        gt_proportions[gt] = gt_mask.shape[0] / matrix.shape[0]
        max_recall = None
        max_recall_sys = None
        max_precision = None
        max_precision_sys = None
        max_f1 = None
        max_f1_sys = None
        max_f1_recall = None
        max_f1_precision = None
        for sys_i, sys in enumerate(sys_by_gt_values):
            sys_by_gt_count = sys_by_gt_counts[sys_i]
            sys_total_count = sys_total_counts[sys_values.tolist().index(
                sys)]
            if gt_total_count != 0:
                recall = sys_by_gt_count / gt_total_count
            else:
                recall = 1

            recall_values[(gt, sys)] = recall

            sys_proportions[sys] = sys_total_count / matrix.shape[0]
            if sys_total_count != 0:
                precision = sys_by_gt_count / sys_total_count
            else:
                precision = 1
            precision_values[(gt, sys)] = precision

            f1 = 2 * ((recall * precision) / (recall + precision))

            if max_recall == None or recall > max_recall:
                max_recall = recall
                max_recall_sys = sys
            if max_precision == None or precision > max_precision:
                max_precision = precision
                max_precision_sys = sys
            if max_f1 == None or f1 > max_f1:
                max_f1 = f1
                max_f1_sys = sys
                max_f1_recall = recall
                max_f1_precision = precision
        precision_max_recall = precision_values[(gt, max_recall_sys)]
        recall_max_precision = recall_values[(gt, max_precision_sys)]
        gt_result = {
            'Max Recall': max_recall,
            'Max Precision': max_precision,
            'Precision for Max Recall': precision_max_recall,
            'Recall for Max Precision': recall_max_precision,
            'f1': max_f1,
            'recall': max_f1_recall,
            'precision': max_f1_precision,
        }
        gt_results[gt] = gt_result
        overall_results['Max Recall'] += max_recall
        overall_results['Max Precision'] += max_precision
        overall_results['Precision for Max Recall'] += precision_max_recall
        overall_results['Recall for Max Precision'] += recall_max_precision
        overall_results['GT_mean_f1'] += max_f1
        overall_results['GT_mean_recall'] += max_f1_recall
        overall_results['GT_mean_precision'] += max_f1_precision

    for sys in sys_values:
        max_recall = None
        max_recall_gt = None
        max_precision = None
        max_precision_gt = None
        max_f1 = None
        max_f1_sys = None
        max_f1_recall = None
        max_f1_precision = None
        for gt in gt_values:
            if (gt, sys) not in recall_values:
                continue
            if (gt, sys) not in precision_values:
                continue
            recall = recall_values[(gt, sys)]
            precision = precision_values[(gt, sys)]

            f1 = 2 * ((recall * precision) / (recall + precision))

            if max_recall == None or recall > max_recall:
                max_recall = recall
                max_recall_gt = gt
            if max_precision == None or precision > max_precision:
                max_precision = precision
                max_precision_gt = gt
            if max_f1 == None or f1 > max_f1:
                max_f1 = f1
                max_f1_sys = sys
                max_f1_recall = recall
                max_f1_precision = precision

        precision_max_recall = precision_values[(max_recall_gt, sys)]
        recall_max_precision = recall_values[(max_precision_gt, sys)]
        sys_result = {
            'Max Recall': max_recall,
            'Max Precision': max_precision,
            'Precision for Max Recall': precision_max_recall,
            'Recall for Max Precision': recall_max_precision,
            'f1': max_f1
        }
        sys_results[sys] = sys_result
        overall_results['MR by System'] += max_recall * \
            sys_proportions[sys]
        overall_results['MP by System'] += max_precision * \
            sys_proportions[sys]
        overall_results['PMR by System'] += precision_max_recall * \
            sys_proportions[sys]
        overall_results['RMP by System'] += recall_max_precision * \
            sys_proportions[sys]
        overall_results['MODEL_mean_f1'] += max_f1 * sys_proportions[sys]
        overall_results['MODEL_mean_recall'] += max_f1_recall * \
            sys_proportions[sys]
        overall_results['MODEL_mean_precision'] += max_f1_precision * \
            sys_proportions[sys]
        if max_recall > 0.75 and precision_max_recall > 0.75:
            overall_results['Num Good System Concepts'] += 1

    # Get average over concepts by dividing by number of concepts
    # Don't need to average over models as we already multiplied by proportion.
    overall_results['Max Recall'] /= gt_values.size
    overall_results['Max Precision'] /= gt_values.size
    overall_results['Precision for Max Recall'] /= gt_values.size
    overall_results['Recall for Max Precision'] /= gt_values.size
    overall_results['GT_mean_f1'] /= gt_values.size
    overall_results['GT_mean_recall'] /= gt_values.size
    overall_results['GT_mean_precision'] /= gt_values.size
    overall_results['GT_to_MODEL_ratio'] = overall_results['Num Good System Concepts'] / \
        len(gt_values)
    return overall_results

def get_discrimination_results(log, model_column="active_model"):
    """ Calculate how many standard deviations the active state
    is from other states. 
    We first split the active state history into chunks representing 
    each segment.
    We then shrink this by 50 on each side to exclude transition periods.
    We then compare the distance from the active state to each non-active state
    in terms of stdev. We use the max of the active state stdev or comparison stdev
    for the given chunk, representing how much the active state could be discriminated
    from the comparison state.
    We return a set of all comparisons, a set of average per active state, and overall average.
    """
    models = log[model_column].unique()
    # Early similarity is unstable, so exclude first 250 obs
    all_state_active_similarity = log['all_state_active_similarity'].replace(
        '-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)[250:]
    if len(all_state_active_similarity.columns) == 0:
        return -1, None, None
    # Scale to between 0 and 1, so invariant
    # to the size of the similarity function.

    values = np.concatenate([all_state_active_similarity[m].dropna(
    ).values for m in all_state_active_similarity.columns])
    try:
        max_similarity = np.percentile(values, 90)
    except:
        return None, None, 0
    min_similarity = min(values)

    # Split into chunks using the active model.
    # I.E. new chunk every time the active model changes.
    # We shrink chunks by 50 each side to discard transition.
    model_changes = log[model_column] != log[model_column].shift(
        1).fillna(method='bfill')
    chunk_masks = model_changes.cumsum()
    chunks = chunk_masks.unique()
    divergences = {}
    active_model_mean_divergences = {}
    mean_divergence = []

    # Find the number of observations we are interested in.
    # by combining chunk masks.
    all_chunks = None
    for chunk in chunks:
        chunk_mask = chunk_masks == chunk
        chunk_shift = chunk_mask.shift(50, fill_value=0)
        smaller_mask = chunk_mask & chunk_shift
        chunk_shift = chunk_mask.shift(-50, fill_value=0)
        smaller_mask = smaller_mask & chunk_shift
        all_state_active_similarity = log['all_state_active_similarity'].loc[smaller_mask].replace(
            '-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)

        # We skip chunks with only an active state.
        if len(all_state_active_similarity.columns) < 2:
            continue
        if all_chunks is None:
            all_chunks = smaller_mask
        else:
            all_chunks = all_chunks | smaller_mask

    # If we only have one state, we don't
    # have any divergences
    if all_chunks is None:
        return None, None, 0

    for chunk in chunks:
        chunk_mask = chunk_masks == chunk
        chunk_shift = chunk_mask.shift(50, fill_value=0)
        smaller_mask = chunk_mask & chunk_shift
        chunk_shift = chunk_mask.shift(-50, fill_value=0)
        smaller_mask = smaller_mask & chunk_shift

        # state similarity is saved in the csv as a ; seperated list, where the index is the model ID.
        # This splits this column out into a column per model.
        all_state_active_similarity = log['all_state_active_similarity'].loc[smaller_mask].replace(
            '-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
        if all_state_active_similarity.shape[0] < 1:
            continue
        active_model = log[model_column].loc[smaller_mask].unique()[0]
        if active_model not in all_state_active_similarity:
            continue
        for m in all_state_active_similarity.columns:
            all_state_active_similarity[m] = (
                all_state_active_similarity[m] - min_similarity) / (max_similarity - min_similarity)
            all_state_active_similarity[m] = np.clip(
                all_state_active_similarity[m], 0, 1)
        # Find the proportion this chunk takes up of the total.
        # We use this to proportion the results.
        chunk_proportion = smaller_mask.sum() / all_chunks.sum()
        chunk_mean = []
        for m in all_state_active_similarity.columns:
            if m == active_model:
                continue

            # If chunk is small, we may only see 0 or 1 observations.
            # We can't get a standard deviation from this, so we skip.
            if all_state_active_similarity[m].shape[0] < 2:
                continue
            # Use the max of the active state, and comparison state as the Stdev.
            # You cannot distinguish if either is larger than difference.
            if active_model in all_state_active_similarity:
                scale = np.mean([all_state_active_similarity[m].std(
                ), all_state_active_similarity[active_model].std()])
            else:
                scale = all_state_active_similarity[m].std()
            divergence = all_state_active_similarity[m] - \
                all_state_active_similarity[active_model]
            avg_divergence = divergence.sum() / divergence.shape[0]

            scaled_avg_divergence = avg_divergence / scale if scale > 0 else 0

            # Mutiply by chunk proportion to average across data set.
            # Chunks are not the same size, so cannot just mean across chunks.
            scaled_avg_divergence *= chunk_proportion
            if active_model not in divergences:
                divergences[active_model] = {}
            if m not in divergences[active_model]:
                divergences[active_model][m] = scaled_avg_divergence
            if active_model not in active_model_mean_divergences:
                active_model_mean_divergences[active_model] = []
            active_model_mean_divergences[active_model].append(
                scaled_avg_divergence)
            chunk_mean.append(scaled_avg_divergence)

        if len(all_state_active_similarity.columns) > 1 and len(chunk_mean) > 0:
            mean_divergence.append(np.mean(chunk_mean))

    # Use sum because we multiplied by proportion already, so just need to add up.
    mean_divergence = np.sum(mean_divergence)
    for m in active_model_mean_divergences:
        active_model_mean_divergences[m] = np.sum(
            active_model_mean_divergences[m])

    return divergences, active_model_mean_divergences, mean_divergence

def plot_feature_weights(log):
    feature_weights = log['feature_weights'].replace(
        '-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True)

def get_unique_stream_names(all_stream_concepts):
    unique_stream_names = []
    for c in all_stream_concepts:
        c_name = c[3]
        if c_name not in unique_stream_names:
            unique_stream_names.append(c_name)
    return unique_stream_names

def get_ground_truth_concept_idx(current_observation, all_stream_concepts, unique_stream_name_list, gt_context_values):
    # If we are given the context as a list
    if gt_context_values is not None:
        return gt_context_values[current_observation]
    
    # Otherwise, we are given as ranges and must find with range contains the current observation
    ground_truth_concept_init = None
    for c in all_stream_concepts:
        concept_start = c[0]
        if concept_start <= current_observation <= c[1]:
            ground_truth_concept_init = unique_stream_name_list.index(c[3])
    return ground_truth_concept_init

def dump_results(option, df, log_dump_path, log_path, result_path, merges, log=None):
    log_df = None
    if log is not None:
        log_df = log
    else:
        log_df = pd.read_csv(log_dump_path)
    
    # Find the final merged identities for each model ID
    df['merge_model'] = df['active_model'].copy()
    for m_init in merges:
        m_to = merges[m_init]
        while m_to in merges:
            m_from = m_to
            m_to = merges[m_from]
            if m_to == m_from:
                break
        df['merge_model'] = df['merge_model'].replace(m_init, m_to)
    
    # Fill in deleted models with the next model, as is done in AiRStream
    df['repair_model'] = df['merge_model'].copy()
    # Get deleted models from the progress log, some will have been deleted from merging
    # but others will just have been deleted
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
    
    # set deleted vals to nan
    for dm in deleted_models:
        df['repair_model'] = df['repair_model'].replace(dm, np.nan)
    df['repair_model'] = df['repair_model'].fillna(method="bfill")

        

    overall_accuracy = log_df['overall_accuracy'].values[-1]
    overall_time = log_df['cpu_time'].values[-1]
    overall_mem = log_df['ram_use'].values[-1]
    peak_fingerprint_mem = log_df['ram_use'].values.max()
    average_fingerprint_mem = log_df['ram_use'].values.mean()
    final_feature_weight = log_df['feature_weights'].values[-1]
    try:
        feature_weights_strs = final_feature_weight.split(';')
        feature_weights = {}
        for ftr_weight_str in feature_weights_strs:
            feature_name, feature_value = ftr_weight_str.split(':')
            feature_weights[feature_name] = float(feature_value)
    except:
        feature_weights = {"NoneRecorded": -1}

    acc, kappa, kappa_m, kappa_t = get_performance(log_df)
    result = {
        'overall_accuracy': overall_accuracy,
        'acc': acc,
        'kappa': kappa,
        'kappa_m': kappa_m,
        'kappa_t': kappa_t,
        'overall_time': overall_time,
        'overall_mem': overall_mem,
        'peak_fingerprint_mem': peak_fingerprint_mem,
        'average_fingerprint_mem': average_fingerprint_mem,
        'feature_weights': feature_weights,
        **option
    }
    for delta in [50, 250, 500]:
        acc, kappa, kappa_m, kappa_t = get_drift_point_accuracy(
            log_df, delta)
        result[f"drift_{delta}_accuracy"] = acc
        result[f"drift_{delta}_kappa"] = kappa
        result[f"drift_{delta}_kappa_m"] = kappa_m
        result[f"drift_{delta}_kappa_t"] = kappa_t
        acc, kappa, kappa_m, kappa_t = get_driftdetect_point_accuracy(
            log_df, delta)
        result[f"driftdetect_{delta}_accuracy"] = acc
        result[f"driftdetect_{delta}_kappa"] = kappa
        result[f"driftdetect_{delta}_kappa_m"] = kappa_m
        result[f"driftdetect_{delta}_kappa_t"] = kappa_t

    match_results = get_recall_precision(log_df, 'active_model')
    for k, v in match_results.items():
        result[f"nomerge-{k}"] = v
    match_results = get_recall_precision(log_df, 'merge_model')
    for k, v in match_results.items():
        result[f"m-{k}"] = v
    match_results = get_recall_precision(log_df, 'repair_model')
    for k, v in match_results.items():
        result[f"r-{k}"] = v
    for k, v in match_results.items():
        result[f"{k}"] = v

    discriminations, active_model_mean_discriminations, mean_discrimination = get_discrimination_results(
        log_df, 'active_model')
    result['nomerge_mean_discrimination'] = mean_discrimination
    discriminations, active_model_mean_discriminations, mean_discrimination = get_discrimination_results(
        log_df, 'merge_model')
    result['m_mean_discrimination'] = mean_discrimination
    discriminations, active_model_mean_discriminations, mean_discrimination = get_discrimination_results(
        log_df, 'repair_model')
    result['r_mean_discrimination'] = mean_discrimination
    result['mean_discrimination'] = mean_discrimination

    with result_path.open('w+') as f:
        json.dump(result, f, cls=NpEncoder)
    log_df.to_csv(log_dump_path, index=False)

def check_dataset_args(args):
    real_drift_datasets = ['AQSex', 'AQTemp',
                           'Arabic', 'cmc', 'UCI-Wine', 'qg', 'covtype-Elevation', 'covtype-Slope', "poker-LastCard"]
    real_unknown_datasets = ['Rangiora_test2-day-nonordered', 'Rangiora_test2-WS_4-nonordered', 'Rangiora_test2-WD_4-nonordered', 'Rangiora_test-WS_3-nonordered', 'Rangiora_test-WD_3-nonordered', 'Rangiora_test-WD_1-nonordered', 'Rangiora_test-WD_1', 'cmcContextTest', 'ArabicContextTest', 'Airlines', 'Arrowtown', 'AWS', 'Beijing', 'covtype', 'Elec', 'gassensor', 'INSECTS-abn', 'INSECTS-irbn', 'INSECTS-oocn',
                             'NasaFlight', 'KDDCup', 'Luxembourg', 'NOAA', 'outdoor', 'ozone', 'Poker', 'PowerSupply', 'Rangiora', 'rialto', 'Sensor', 'Spam', 'SpamAssassin']
    synthetic_MI_datasets = ['RTREESAMPLE', 'HPLANESAMPLE']
    synthetic_perf_only_datasets = ['STAGGER', 'RTREEMedSAMPLE', 'RBFMed']
    synthetic_unused = ['STAGGERS', 'RTREE', 'HPLANE', 'RTREEEasy', 'RTREEEasySAMPLE', 'RBFEasy', 'RTREEEasyF',
                        'RTREEEasyA', 'SynEasyD', 'SynEasyA', 'SynEasyF', 'SynEasyDA', 'SynEasyDF', 'SynEasyAF', 'SynEasyDAF']
    synthetic_dist = ["LM_WINDSIM", "LM_RTREE", "RTREESAMPLE_Diff", "RTREESAMPLE_HARD", "WINDSIM", 'SigNoiseGenerator-1', 'SigNoiseGenerator-2', 'SigNoiseGenerator-3', 'SigNoiseGenerator-4', 'SigNoiseGenerator-5', 'SigNoiseGenerator-6', 'SigNoiseGenerator-7', 'SigNoiseGenerator-8', 'SigNoiseGenerator-9', 'SigNoiseGenerator-10', 'FeatureWeightExpGenerator',
                      'RTREESAMPLEHP-23', 'RTREESAMPLEHP-14', 'RTREESAMPLEHP-A', 'RTREESAMPLE-UB', 'RTREESAMPLE-NB', 'RTREESAMPLE-DB', 'RTREESAMPLE-UU', 'RTREESAMPLE-UN', 'RTREESAMPLE-UD', 'RTREESAMPLE-NU', 'RTREESAMPLE-NN', 'RTREESAMPLE-ND', 'RTREESAMPLE-DU', 'RTREESAMPLE-DN', 'RTREESAMPLE-DD']
    datasets = set()
    for ds in (args.datasets if type(args.datasets) is list else [args.datasets]):
        if ds == 'all_exp':
            for x in [*real_drift_datasets]:
                datasets.add((x, 'Real'))
            for x in [*synthetic_MI_datasets, *synthetic_perf_only_datasets]:
                datasets.add((x, 'Synthetic'))
        elif ds == 'real':
            for x in [*real_drift_datasets]:
                datasets.add((x, 'Real'))
        elif ds == 'synthetic':
            for x in [*synthetic_MI_datasets, *synthetic_perf_only_datasets, *synthetic_dist]:
                datasets.add((x, 'Synthetic'))
        elif ds in real_drift_datasets:
            datasets.add((ds, 'Real'))
        elif ds in real_unknown_datasets:
            datasets.add((ds, 'Real'))
        elif ds in synthetic_MI_datasets:
            datasets.add((ds, 'Synthetic'))
        elif ds in synthetic_perf_only_datasets:
            datasets.add((ds, 'Synthetic'))
        elif ds in synthetic_unused:
            datasets.add((ds, 'Synthetic'))
        elif ds in synthetic_dist:
            datasets.add((ds, 'Synthetic'))
        else:
            raise ValueError("Dataset not found")
    return datasets
