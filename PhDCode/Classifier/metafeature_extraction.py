import itertools
from copy import deepcopy

import numpy as np
import scipy.stats
import shap
from PhDCode.Classifier.feature_selection.fisher_score import \
    fisher_score
from PhDCode.Classifier.feature_selection.mutual_information import *
from PhDCode.utils.utils import (SingleTree, shap_values,
                                      tree_ensemble_init)
from entropy import perm_entropy
from numpy import inf
from PyEMD import EMD
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import argrelmax, argrelmin

# Moneky patch shap to use scikit-multiflow hoeffding trees.
shap.explainers._tree.TreeEnsemble.__init__ = tree_ensemble_init
shap.explainers._tree.Tree.shap_values = shap_values
shap.SingleTree = SingleTree


def window_to_timeseries(window):
    """ Function to take a window of data of the form:
    [(X, y, prediction, is_error), "name"]
    and return 5 timeseries of
    features, labels, predictions, errors and error distances.
    """
    features = []
    for f in window[0][0][0]:
        features.append([])
    labels = []
    predictions = []
    errors = []
    error_distances = []
    last_distance = None

    for i, row in enumerate(window[0]):
        X = row[0]
        y = row[1]
        p = row[2]
        e = row[3]
        for fi, f in enumerate(X):
            features[fi].append(f)
        labels.append(y)
        predictions.append(p)
        errors.append(e)
        if not e:
            if last_distance is None:
                last_distance = i
            else:
                distance = i - last_distance
                error_distances.append(distance)
                last_distance = i
    if len(error_distances) == 0:
        error_distances = [0]

    return (features, labels, predictions, errors, error_distances)

def observations_to_timeseries(observations):
    """ Function to take a window of Observations
    and return 5 timeseries of
    features, labels, predictions, errors and error distances.
    """
    features = []
    n_features = len(observations[0].X)
    for f in range(n_features):
        features.append([])
    labels = []
    predictions = []
    errors = []
    error_distances = []
    last_distance = None

    for i, ob in enumerate(observations):
        for fi, f in enumerate(ob.X):
            features[fi].append(f)
        labels.append(ob.y)
        predictions.append(ob.p)
        errors.append(ob.correctly_classified)
        if not ob.correctly_classified:
            if last_distance is None:
                last_distance = i
            else:
                distance = i - last_distance
                error_distances.append(distance)
                last_distance = i
    if len(error_distances) == 0:
        error_distances = [0]

    return (features, labels, predictions, errors, error_distances)


def update_timeseries(sources, window, window_size, num_timesteps):
    """ Using a source set of timeseries, and set of most recent
    elements, remove num_timestep elements from the end of each
    time series and add num_timesteps most recent elements to the 
    start from window.
    """
    source_features = sources[0]
    source_labels = sources[1]
    source_predictions = sources[2]
    source_errors = sources[3]
    source_error_distances = sources[4]

    num_elements_to_remove = max(
        0, len(source_labels) + num_timesteps - window_size)
    updated_features = []
    for fi in range(len(source_features)):
        updated_features.append(source_features[fi][num_elements_to_remove:])
    updated_labels = source_labels[num_elements_to_remove:]
    updated_predictions = source_predictions[num_elements_to_remove:]
    updated_errors = source_errors[num_elements_to_remove:]

    window_update_elements = itertools.islice(
        window[0], max(len(window[0])-num_timesteps, 0), None)
    for i, row in enumerate(window_update_elements):
        X = row[0]
        y = row[1]
        p = row[2]
        e = row[3]
        for fi, f in enumerate(X):
            updated_features[fi].append(f)
        updated_labels.append(y)
        updated_predictions.append(p)
        updated_errors.append(e)
    updated_error_distances = []
    last_distance = None
    for i, e in enumerate(updated_errors):
        if not e:
            if last_distance is None:
                last_distance = i
            else:
                distance = i - last_distance
                updated_error_distances.append(distance)
                last_distance = i
    if len(updated_error_distances) == 0:
        updated_error_distances = [0]

    return (updated_features, updated_labels, updated_predictions, updated_errors, updated_error_distances)


def get_concept_stats_from_base(timeseries, model, feature_base, feature_base_flat, stored_shap=None, ignore_sources=None, ignore_features=None, normalizer=None):
    """ Get concept stats by reusing an already calculated base.
    This avoids recalculations on static features between models
    on the same window.
    """
    return get_concept_stats(timeseries, model, feature_base, feature_base_flat, stored_shap, ignore_sources, ignore_features, normalizer=normalizer)


def make_shap_model(model):
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", check_additivity=False)


def get_concept_stats(timeseries, model, feature_base=None, feature_base_flat=None, stored_shap=None, ignore_sources=None, ignore_features=None, normalizer=None):
    """ Wrapper for calculating the stats of a set
    of timeseries generated by a given model.
    """
    if ignore_sources is None:
        ignore_sources = []
    if ignore_features is None:
        ignore_features = []
    concept_stats = {}

    normalizer_initialized = False
    flat_ignore_vec = None
    if normalizer.ignore_indexes is not None:
        flat_ignore_vec = np.empty(normalizer.ignore_num_signals)
        normalizer_initialized = True

    features = timeseries[0]
    labels = timeseries[1]
    predictions = timeseries[2]
    errors = timeseries[3]
    error_distances = timeseries[4]

    length = len(features[0])
    X = []
    for i in range(length):
        X.append([])
        for f in features:
            X[-1].append(f[i])
        X[-1] = np.array(X[-1])
    X = np.array(X)
    # np.testing.assert_array_almost_equal(X, X2)
    # X = np.hstack([np.array(f).reshape(-1, 1) for f in features])

    if 'FI' not in ignore_features:
        if not stored_shap:
            shap_model = make_shap_model(model)
        else:
            shap_model = stored_shap
        shaps = shap_model.shap_values(
            X, check_additivity=False, approximate=True)
        # If there is only 1 label, shaps just returns the matrix, otherwise it returns
        # a list of matricies. This converts the single case into a list.
        if not isinstance(shaps, list):
            shaps = [shaps]
        mean_shaps = np.sum(np.abs(shaps[0]), axis=0)
        SHAP_vals = [abs(x) for x in mean_shaps]
    else:
        SHAP_vals = [0 for x in range(len(features))]

    if 'features' not in ignore_sources:
        for f1, f in enumerate(features):
            feature_name = f"f{f1}"
            if feature_name not in ignore_sources:
                if not feature_base:
                    stats = get_timeseries_stats(
                        f, SHAP_vals[f1], ignore_features=ignore_features)
                    if normalizer_initialized:
                        for stats_feature, stat_value in stats.items():
                            ignore_index = normalizer.ignore_indexes[feature_name][stats_feature]
                            flat_ignore_vec[ignore_index] = stat_value
                else:
                    stats = deepcopy(feature_base[feature_name])
                    if 'FI' not in ignore_features:
                        stats["FI"] = deepcopy(SHAP_vals[f1])
                    if normalizer_initialized:
                        source_start, source_end = normalizer.ignore_source_ranges[feature_name]
                        flat_ignore_vec[source_start:source_end] = feature_base_flat[source_start:source_end]
                        if 'FI' not in ignore_features:
                            flat_ignore_vec[normalizer.ignore_indexes[feature_name]["FI"]] = deepcopy(
                                SHAP_vals[f1])

                concept_stats[feature_name] = stats

    if 'labels' not in ignore_sources:
        if not feature_base:
            stats = get_timeseries_stats(
                list(map(float, labels)), ignore_features=ignore_features)
            if normalizer_initialized:
                for stats_feature, stat_value in stats.items():
                    ignore_index = normalizer.ignore_indexes['labels'][stats_feature]
                    flat_ignore_vec[ignore_index] = stat_value
        else:
            stats = deepcopy(feature_base["labels"])
            if normalizer_initialized:
                source_start, source_end = normalizer.ignore_source_ranges['labels']
                flat_ignore_vec[source_start:source_end] = feature_base_flat[source_start:source_end]
        concept_stats["labels"] = stats
    if 'predictions' not in ignore_sources:
        stats = get_timeseries_stats(
            list(map(float, predictions)), ignore_features=ignore_features)
        concept_stats["predictions"] = stats
        if normalizer_initialized:
            for stats_feature, stat_value in stats.items():
                ignore_index = normalizer.ignore_indexes['predictions'][stats_feature]
                flat_ignore_vec[ignore_index] = stat_value
    if 'errors' not in ignore_sources:
        stats = get_timeseries_stats(
            list(map(float, errors)), ignore_features=ignore_features)
        concept_stats["errors"] = stats
        if normalizer_initialized:
            for stats_feature, stat_value in stats.items():
                ignore_index = normalizer.ignore_indexes['errors'][stats_feature]
                flat_ignore_vec[ignore_index] = stat_value
    if 'error_distances' not in ignore_sources:
        stats = get_timeseries_stats(
            list(map(float, error_distances)), ignore_features=ignore_features)
        concept_stats["error_distances"] = stats
        if normalizer_initialized:
            for stats_feature, stat_value in stats.items():
                ignore_index = normalizer.ignore_indexes['error_distances'][stats_feature]
                flat_ignore_vec[ignore_index] = stat_value
    if not normalizer_initialized:
        normalizer.init_signals(concept_stats)
        return get_concept_stats(timeseries, model, feature_base=feature_base, feature_base_flat=feature_base_flat, stored_shap=stored_shap, ignore_sources=ignore_sources, ignore_features=ignore_features, normalizer=normalizer)
    return concept_stats, flat_ignore_vec


def turningpoints(lst):
    dx = np.diff(lst)
    return np.sum(dx[1:] * dx[:-1] < 0)
# def turningpoints(lst):
#     return len([*argrelmin(np.array(lst)), *argrelmax(np.array(lst))])


def get_timeseries_stats(timeseries, FI=None, ignore_features=None):
    """ Calculates a set of statistics for a given
    timeseries.

    """
    stats = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        if len(timeseries) < 3:
            if 'mean' not in ignore_features:
                stats["mean"] = np.mean(timeseries)
            if 'stdev' not in ignore_features:
                stats["stdev"] = 0
            if 'skew' not in ignore_features:
                stats["skew"] = 0
            if 'kurtosis' not in ignore_features:
                stats['kurtosis'] = 0
            if len(timeseries) > 1:
                if 'stdev' not in ignore_features:
                    stats["stdev"] = np.std(timeseries, ddof=1)
                if 'skew' not in ignore_features:
                    stats["skew"] = scipy.stats.skew(timeseries, bias=True)
                if 'kurtosis' not in ignore_features:
                    stats['kurtosis'] = scipy.stats.kurtosis(timeseries, bias=True)
            if 'turning_point_rate' not in ignore_features:
                stats["turning_point_rate"] = 0
            if 'acf' not in ignore_features:
                if 'acf_1' not in ignore_features:
                    stats["acf_1"] = 0
                if 'acf_2' not in ignore_features:
                    stats["acf_2"] = 0
            if 'pacf' not in ignore_features:
                if 'pacf_1' not in ignore_features:
                    stats["pacf_1"] = 0
                if 'pacf_2' not in ignore_features:
                    stats["pacf_2"] = 0
            if 'MI' not in ignore_features:
                stats["MI"] = 0
            if 'FI' not in ignore_features:
                stats["FI"] = 0
            if 'IMF' not in ignore_features:
                if 'IMF_0' not in ignore_features:
                    stats["IMF_0"] = 0
                if 'IMF_1' not in ignore_features:
                    stats["IMF_1"] = 0
                if 'IMF_2' not in ignore_features:
                    stats["IMF_2"] = 0
            return stats

        if 'IMF' not in ignore_features:
            emd = EMD(max_imf=2, spline_kind='slinear')
            IMFs = emd(np.array(timeseries), max_imf=2)
            for i, imf in enumerate(IMFs):
                if f"IMF_{i}" not in ignore_features:
                    stats[f"IMF_{i}"] = perm_entropy(imf)
            for i in range(3):
                if f"IMF_{i}" not in stats and f"IMF_{i}" not in ignore_features:
                    stats[f"IMF_{i}"] = 0
        if 'mean' not in ignore_features:
            stats["mean"] = np.mean(timeseries)
        if 'stdev' not in ignore_features:
            stats["stdev"] = np.std(timeseries, ddof=1)
        if 'skew' not in ignore_features:
            stats["skew"] = scipy.stats.skew(timeseries, bias=True)
        if 'kurtosis' not in ignore_features:
            stats['kurtosis'] = scipy.stats.kurtosis(timeseries, bias=True)
            # if stats["stdev"] == 0:
            #     stats['kurtosis'] = 0
        if 'turning_point_rate' not in ignore_features:
            tp = int(turningpoints(timeseries))
            tp_rate = tp / len(timeseries)
            stats['turning_point_rate'] = tp_rate

        if 'acf' not in ignore_features:
            acf_vals = acf(timeseries, nlags=3, fft=True)
            for i, v in enumerate(acf_vals):
                if i == 0:
                    continue
                if i > 2:
                    break
                if f"acf_{i}" not in ignore_features:
                    # TODO try other values
                    stats[f"acf_{i}"] = v if not np.isnan(v) else -1
        if 'pacf' not in ignore_features:
            try:
                pacf_vals = pacf(timeseries, nlags=3)
            except:
                # TODO try other values
                pacf_vals = [-1 for x in range(6)]
            for i, v in enumerate(pacf_vals):
                if i == 0:
                    continue
                if i > 2:
                    break
                if f"pacf_{i}" not in ignore_features:
                    # TODO try other values
                    stats[f"pacf_{i}"] = v if not np.isnan(v) else -1

        if 'MI' not in ignore_features:
            if len(timeseries) > 4:
                current = np.array(timeseries)
                previous = np.roll(current, -1)
                current = current[:-1]
                previous = previous[:-1]
                X = np.array(current).reshape(-1, 1)
                # Setting the random state is mostly for testing.
                # It can induce randomness in MI which is weird for paired
                # testing, getting different results with the same feature vec.
                MI = mutual_info_regression(
                    X=X, y=previous, random_state=42, copy=False)[0]
            else:
                MI = 0
            stats["MI"] = MI

        if 'FI' not in ignore_features:
            stats["FI"] = FI if FI is not None else 0

    return stats
