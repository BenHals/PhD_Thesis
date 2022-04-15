
#%%
from PhDCode.Classifier.advantage_classifier import FSMClassifier
from PhDCode.Classifier.hoeffding_tree_evolution import HoeffdingTreeEvoClassifier as TS_HoeffdingTree
# from PhDCode.Classifier.advantage_fsm_o.fsm_classifier import FSMClassifier
# from PhDCode.Classifier.advantage_fsm_o.tracksplit_hoeffding_tree import TS_HoeffdingTree


from skmultiflow.data.data_stream import DataStream

from scipy.io import arff
import pandas as pd
import pathlib
import json
import pickle
from collections import Counter
import numpy as np
import tqdm

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



#%%
datastream_filename = pathlib.Path(r"S:\PhD\testing\MemManagementTest\2199.ARFF")
info_filename = pathlib.Path(r"S:\PhD\testing\MemManagementTest\2199_dsinfo.txt")
cc_filename = pathlib.Path(r"S:\PhD\testing\MemManagementTest\2199_concept_chain.pickle")
dsinfo_str = info_filename.open('r').readlines()
print(dsinfo_str[0][:-1])
dsinfo = json.loads(dsinfo_str[0][:-1])
concept_chain = dsinfo['concept_chain']
concept_chain_2 = pickle.load(cc_filename.open('rb'))
print(concept_chain_2)

def get_gt_id(i, cc):
    found_val = 0
    for start, c_id in cc.items():
        if start <= i:
            found_val = c_id
        else:
            break
    return found_val

#%%
print(get_gt_id(0, concept_chain_2))
print(get_gt_id(1420100, concept_chain_2))
print(get_gt_id(1426000, concept_chain_2))

#%%
# data = arff.loadarff(datastream_filename)
# ds_df = pd.DataFrame(data[0])

# #%%
# ds_df.head()
# stream = DataStream(ds_df)

# learner = HoeffdingTreeEvoClassifier

# classifier = FSMClassifier(
#     concept_limit=35,
#     memory_management="rA",
#     learner=learner,
#     window=175,
#     sensitivity=0.05,
#     concept_chain=concept_chain_2,
#     use_clean=True,
#     merge_strategy="sur",
#     suppress=False
# )
try:
    data = arff.loadarff(datastream_filename)
    ds_df = pd.DataFrame(data[0])
except Exception as e:
    print(e)
    print("trying csv")
    ds_df = pd.read_csv(datastream_filename, header=None)

for c_i,c in enumerate(ds_df.columns):
    
    if pd.api.types.is_string_dtype(ds_df[c]):
        print(f"Factoizing {c}")
        print(pd.factorize(ds_df[c])[0].shape)
        ds_df[c] = pd.factorize(ds_df[c])[0]
    
    # print(f"{c_i}: {len(df.columns) - 1}")
    # if c_i == len(df.columns) - 1:
    #     print(f"converting {c}")
    #     df[c] = df[c].astype('category')
    

print(ds_df.info())

datastream = DataStream(ds_df)
datastream.concept_chain = concept_chain
print(concept_chain)
datastream.prepare_for_use()
#%%
# df.head()
# stream = DataStream(df)
# stream.concept_chain = concept_chain_2
# stream.prepare_for_use()

# learner = HoeffdingTreeEvoClassifier
learner = lambda : TS_HoeffdingTree(max_byte_size = 33554432, memory_estimate_period = 1000)

classifier = FSMClassifier(
    concept_limit=35,
    memory_management="rA",
    learner=learner,
    window=175,
    sensitivity=0.05,
    concept_chain=concept_chain_2,
    use_clean=True,
    merge_strategy="sur",
    poisson=10,
)

print(classifier.__dict__)

#%%
option = {}

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

def get_ground_truth_concept_idx(current_observation, all_stream_concepts, unique_stream_name_list):
    ground_truth_concept_init = None
    for c in all_stream_concepts:
        concept_start = c[0]
        if concept_start <= current_observation < c[1]:
            ground_truth_concept_init = unique_stream_name_list.index(c[3])
    return ground_truth_concept_init

def dump_results(option, log_path, result_path, merges, log=None):
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

#%%
option = {
    'length': ds_df.shape[0],
    'noise': 0.0,
}
UID = "UID123"

classes = list(ds_df['y0'].unique())
print(classes)

# print(type(classifier))
output_path = pathlib.Path(r"S:\PhD\testing\MemManagementTest")
run_index = 0
run_name = f"run_{UID}_{run_index}"
log_dump_path = output_path / f"{run_name}.csv"
options_dump_path = output_path / f"{run_name}_options.txt"
options_dump_path_partial = output_path / f"partial_{run_name}_options.txt"
results_dump_path = output_path / f"results_{run_name}.txt"

partial_log_size = 2500
partial_logs = []
partial_log_index = 0

with options_dump_path_partial.open('w+') as f:
    json.dump(option, f, cls=NpEncoder)

right = 0
wrong = 0

monitoring_data = []
monitoring_header = ('example', 'y', 'p', 'is_correct', 'right_sum', 'wrong_sum', 'overall_accuracy', 'active_model', 'ground_truth_concept', 'drift_occured', 'change_detected', 'model_evolution',
                        'active_state_active_similarity', 'active_state_buffered_similarity', 'all_state_buffered_similarity', 'all_state_active_similarity', 'feature_weights', 'concept_likelihoods', 'concept_priors', 'concept_priors_1h', 'concept_priors_2h', 'concept_posteriors', "adwin_likelihood_estimate", "adwin_posterior_estimate", "adwin_likelihood_estimate_background", "adwin_posterior_estimate_background", "merges", 'deletions', 'cpu_time', 'ram_use', 'fingerprint_ram')

progress_bar = tqdm.tqdm(total=option['length'])
pbar_updates = 0

# noise_rng = np.random.RandomState(option['seed'])
noise_rng = np.random.default_rng()
last_active_model = 0
for i in range(option['length']):
# for i in range(125000):
    current_merges = None
    observation_monitoring = {}
    observation_monitoring['example'] = i
    X, y = datastream.next_sample()
    if option['noise'] > 0:
        noise_roll = noise_rng.rand()
        if noise_roll < option['noise']:
            y = np.array([noise_rng.choice(classes)])

    observation_monitoring['y'] = int(y[0])
    p = classifier.predict(X)
    observation_monitoring['p'] = int(p[0])
    e = y[0] == p[0]
    observation_monitoring['is_correct'] = int(e)
    right += y[0] == p[0]
    observation_monitoring['right_sum'] = right
    wrong += y[0] != p[0]
    observation_monitoring['wrong_sum'] = wrong
    observation_monitoring['overall_accuracy'] = right / (right + wrong)

    ground_truth_concept_index = get_gt_id(i, concept_chain_2)

    # Control parameters
    drift_occured = False

    for concept_start, c in list(concept_chain_2.items())[1:]:
        if i == concept_start:
            drift_occured = True

    classifier.partial_fit(X, y, classes=classes)
    # Collect monitoring data for storage.
    current_active_model = classifier.fsm.active_state_id 
    observation_monitoring['active_model'] = current_active_model
    observation_monitoring['ground_truth_concept'] = int(ground_truth_concept_index) if ground_truth_concept_index is not None else ground_truth_concept_index
    observation_monitoring['drift_occured'] = int(drift_occured)
    observation_monitoring['change_detected'] = int(last_active_model != current_active_model)
    observation_monitoring['model_evolution'] = len(classifier.fsm.get_state().evolution)

    observation_monitoring['detected_drift'] = int(last_active_model != current_active_model)
    last_active_model = current_active_model


    monitoring_data.append(observation_monitoring)

    if len(monitoring_data) >= partial_log_size:
        log_dump_path_partial = output_path / \
            f"partial_{run_name}_{partial_log_index}.csv"
        df = pd.DataFrame(monitoring_data, columns=monitoring_header)
        df.to_csv(log_dump_path_partial, index=False)
        partial_log_index += 1
        partial_logs.append(log_dump_path_partial)
        monitoring_data = []
        df = None

    # try to aquire the lock to update progress bar.
    # We don't care too much so use a short timeout!
    pbar_updates += 1
    if progress_bar:
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

dump_results(option, log_dump_path, results_dump_path, classifier.merges if hasattr(classifier, "merges") else {}, df)


