#%%
import pathlib
import pandas as pd
import numpy as np
import skmultiflow
from PhDCode.Classifier.select_classifier import SELeCTClassifier
from PhDCode.Classifier.hoeffding_tree_shap import HoeffdingTreeSHAPClassifier
from PhDCode.utils.eval_utils import (
    extract_nominal_attributes,
    make_stream,
)

data_path = pathlib.Path("S:\PhD\Packages\PhDCode\RawData\Real\cmcContextTest") / 'cmc_0.csv'
option = {'classifier': 'CC', 'base_output_path': pathlib.Path('S:/PhD/Packages/PhDCode/output'), 'raw_data_path': pathlib.Path('S:/PhD/Packages/PhDCode/RawData'), 'data_name': 'cmcContextTest', 'data_type': 'Real', 'max_rows': 75000, 'seed': 1, 'seed_action': 'list', 'package_status': 'PhDCode-master-ed2331a', 'log_name': 'expDefault-1637741782.8178604', 'pyinstrument': False, 'FICSUM': False, 'minimal_output': False, 'MAP_selection': False, 'setaffinity': False, 'pcpu': 2, 'cpu': 2, 'experiment_name': 'expDefault', 'repeats': 1, 'concept_max': 6, 'concept_length': 5000, 'repeatproportion': 1.0, 'TMdropoff': 1.0, 'TMforward': 1, 'TMnoise': 0.0, 'drift_width': 0, 'noise': 0, 'conceptdifficulty': 0, 'framework': 'system', 'isources': None, 'ifeatures': ['IMF', 'MI', 'pacf'], 'optdetect': False, 'optselect': False, 'opthalf': False, 'opthalflock': False, 'save_feature_weights': False, 'shuffleconcepts': False, 'similarity_option': 'metainfo', 'MI_calc': 'metainfo', 'window_size': 100, 'sensitivity': 0.05, 'min_window_ratio': 0.65, 'fingerprint_grace_period': 10, 'state_grace_period_window_multiplier': 10, 'bypass_grace_period_threshold': 0.2, 'state_estimator_risk': 0.5, 'state_estimator_swap_risk': 0.75, 'minimum_concept_likelihood': 0.005, 'min_drift_likelihood_threshold': 0.175, 'min_estimated_posterior_threshold': 0.2, 'similarity_gap': 5, 'fp_gap': 15, 'nonactive_fp_gap': 50, 'observation_gap': 5, 'take_observations': True, 'similarity_stdev_thresh': 3, 'similarity_stdev_min': 0.015, 'similarity_stdev_max': 0.175, 'buffer_ratio': 0.2, 'merge_threshold': 0.95, 'background_state_prior_multiplier': 0.4, 'zero_prob_minimum': 0.7, 'multihop_penalty': 0.7, 'prev_state_prior': 50, 'correlation_merge': True, 'fs_method': 'fisher_overall', 'fingerprint_method': 'descriptive', 'fingerprint_bins': 10, 'd_hard_concepts': 3, 'd_easy_concepts': 1, 'n_hard_concepts': 15, 'n_easy_concepts': 15, 'p_hard_concepts': 0.5, 'repository_max': -1, 'discritize_stream': False, 'valuation_policy': 'rA', 'poisson': 6, 'GT_context_location': 'RawData\\Real\\cmcContextTest\\context.csv', 'length': 4419}

#%%
data_df = pd.read_csv(data_path)
#Drop example ID column
data_df = data_df.drop(data_df.columns[0], axis=1)
stream = skmultiflow.data.DataStream(data_df)
#%%
classifier = SELeCTClassifier(
    learner=HoeffdingTreeSHAPClassifier,
    # window_size=100,
    # sensitivity=0.05,
    # similarity_max_stdev=0.175,
    # buffer_ratio=0.2,
    # feature_selection_method='fisher_overall',
    # fingerprint_method='descriptive',
    # merge_threshold=0.95,
    # background_state_prior_multiplier=0.4,
    # multihop_penalty=0.7,
    # prev_state_prior=50,
    )
#%%
right = 0
wrong = 0
while stream.n_remaining_samples() > 0:
    X, y = stream.next_sample()
    p = classifier.predict(X)
    print(f"{y}, {p}, {classifier.active_state_id}")
    classifier.partial_fit(X, y, classes=[0.0, 1.0])
    if p[0] == y[0]:
        right += 1
    else:
        wrong += 1

print(right / (right + wrong))

# %%
