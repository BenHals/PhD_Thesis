import logging
import math
import warnings
from collections import deque
from copy import deepcopy, copy
import itertools

import numpy as np
import scipy.stats
from PhDCode.Classifier.feature_selection.mutual_information import *
from PhDCode.Classifier.normalizer import Normalizer

from PhDCode.Classifier.fingerprint import (FingerprintCache,
                                                       FingerprintBinningCache,
                                                       FingerprintSketchCache)
from PhDCode.Classifier.feature_selection.online_feature_selection import (
    feature_selection_None,
    feature_selection_original,
    feature_selection_fisher,
    feature_selection_fisher_overall,
    feature_selection_cached_MI,
    feature_selection_histogramMI,
    feature_selection_histogram_covredMI,
    mi_from_cached_fingerprint_bins,
    mi_from_fingerprint_sketch,
    mi_cov_from_fingerprint_sketch,
    mi_from_fingerprint_histogram_cache)
from PhDCode.Classifier.airstream_classifier import AirstreamClassifier
from PhDCode.Classifier.rolling_stats import RollingTimeseries
from scipy.spatial.distance import correlation, cosine, jaccard, euclidean
from scipy.stats.stats import pearsonr
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.utils import check_random_state, get_dimensions



warnings.filterwarnings('ignore')

class ConceptState:
    def __init__(self, id):
        self.id = id
        self.current_evolution = 0

class AirstreamWrapperClassifier:
    def __init__(self,
                 suppress=False,
                 learner=None,
                 poisson=6,
                 window_size=50,
                 init_concept_id=0,
                concept_limit=-1,
                memory_management='rA',
                sensitivity=0.05,
                concept_chain=None,
                optimal_selection=False,
                optimal_drift=False,
                rand_weights=False,
                similarity_measure='KT',
                merge_strategy="both",
                merge_similarity=0.9,
                allow_backtrack=True,
                allow_proactive_sensitivity=False,
                num_alternative_states=5,
                conf_sensitivity_drift=0.05,
                conf_sensitivity_sustain=0.125,
                min_proactive_stdev=500,
                alt_test_length=2000,
                alt_test_period=2000,
                max_sensitivity_multiplier=1.5,
                drift_detector="adwin",
                repository_max=-1,
                valuation_policy='rA'):

        if learner is None:
            raise ValueError('Need a learner')

        self.learner = learner

        # suppress debug info
        self.suppress = suppress

        # rand_weights is if a strategy is setting sample
        # weights for training
        self.rand_weights = poisson > 1

        # poisson is the strength of sample weighting
        # based on leverage bagging
        self.poisson = poisson
        
        self.window_size = window_size

        # init the current number of states
        self.max_state_id = init_concept_id

        # Maximum repository size before a valuation policy is applied for deletion.
        # Set to -1 for no max size.
        self.repository_max = repository_max

        # Valuation policy for deletion.
        # Can use:
        #   rA: Based on number of evolutions to estimate advantage
        #   Age: Delete oldest
        #   LRU: Delete least recently used
        self.valuation_policy = valuation_policy

        # init randomness
        self.random_state = None
        self._random_state = check_random_state(self.random_state)

        self.ex = -1
        self.classes = None
        self._train_weight_seen_by_model = 0

        self.active_state_is_new = True

        # init data which is exposed to evaluators
        self.found_change = False
        self.num_states = 1
        self.active_state = self.max_state_id
        self.states = []

        self.detected_drift = False
        self.deletions = []

        # track the last predicted label
        self.last_label = 0

        self.concept_transitions_standard = {}
        self.concept_transitions_warning = {}
        self.concept_transitions_drift = {}

        self.classifier = AirstreamClassifier(
                suppress=suppress,
                concept_limit=concept_limit,
                memory_management=memory_management,
                learner=learner,
                window=window_size,
                sensitivity=sensitivity,
                concept_chain=concept_chain,
                optimal_selection=optimal_selection,
                optimal_drift=optimal_drift,
                rand_weights=rand_weights,
                poisson=poisson,
                similarity_measure=similarity_measure,
                merge_strategy=merge_strategy,
                merge_similarity=merge_similarity,
                allow_backtrack=allow_backtrack,
                allow_proactive_sensitivity=allow_proactive_sensitivity,
                num_alternative_states=num_alternative_states,
                conf_sensitivity_drift=conf_sensitivity_drift,
                conf_sensitivity_sustain=conf_sensitivity_sustain,
                min_proactive_stdev=min_proactive_stdev,
                alt_test_length=alt_test_length,
                alt_test_period=alt_test_period,
                max_sensitivity_multiplier=max_sensitivity_multiplier,
                drift_detector=drift_detector,
                repository_max=self.repository_max,
                valuation_policy=self.valuation_policy,
        )

        self.active_state_id = self.classifier.active_state_id 

        self.observations_since_last_transition = 0

        self.force_transition_to = None
        self.force_transition = False
        
        self.merges = {}
        self.deactivated_states = {}

        self.in_warning = False

        self.fingerprint_type = ConceptState

        self.current_evolution = self.classifier.get_active_state().classifier.evolution

        self.monitor_all_state_active_similarity = None
        self.monitor_all_state_buffered_similarity = None
        self.monitor_active_state_active_similarity = None
        self.monitor_active_state_buffered_similarity = None
        self.monitor_feature_selection_weights = None
        self.concept_likelihoods = {}



    def get_active_state(self):
        return self.classifier.state_repository[self.classifier.active_state_id]

    def reset(self):
        pass

    def get_temporal_x(self, X):
        return np.concatenate([X], axis=None)

    def predict(self, X):
        """
        Predict using the model of the currently active state.
        """
        temporal_X = self.get_temporal_x(X)

        return self.classifier.predict([temporal_X])

    def partial_fit(self, X, y, classes=None, sample_weight=None, masked=False):
        """
        Fit an array of observations.
        Splits input into individual observations and
        passes to a helper function _partial_fit.
        Randomly weights observations depending on 
        Config.
        """

        if masked:
            return
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError(
                    'Inconsistent number of instances ({}) and weights ({}).'
                    .format(row_cnt, len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    self.ex += 1
                    if self.rand_weights and self.poisson >= 1:
                        k = self.poisson
                        sample_weight[i] = k
                    self._partial_fit(X[i], y[i], sample_weight[i], masked)

    def get_imputed_label(self, X, prediction, last_label):
        """ Get a label.
        Imputes when the true label is masked
        """

        return prediction

    def perform_drift_detection_for_bounds(self, initial_state):
        return initial_state != self.classifier.get_active_state().id

    def perform_transition(self, transition_target_id):
        """ Sets the active state and associated meta-info
        Resets detectors, buffer and both observation windows so clean data from the new concept can be collected.
        """
        # We reset drift detection as the new performance will not
        # be the same, and reset history for the new state.
        from_id = self.active_state_id
        to_id = transition_target_id
        #logging.info(f"Transition to {transition_target_id}")
        # print(f"Transition to {to_id}")
        is_new_state = to_id not in self.classifier.state_repository
        self.active_state_is_new = is_new_state

        # Set triggers to future evaluation of this attempt
        self.observations_since_last_transition = 0

    def _partial_fit(self, X, y, sample_weight, masked=False):

        # init defaults for trackers
        found_change = False
        self.detected_drift = False
        self.warning_detected = False
        self.observations_since_last_transition += 1
        initial_state = self.active_state_id

        # get_temporal_x, and get_imputed_label are to deal with masked
        # values where we don't see true label.
        # As the functions are now, they don't do anything extra.
        # But could be extended to reuse last made prediction as
        # the label for example.
        temporal_X = self.get_temporal_x(X)

        foreground_p = self.classifier.predict([temporal_X])[0]
        # self.get_active_state().partial_fit(temporal_X, y, sample_weight=sample_weight, classes=self.classes)
        self.classifier.partial_fit([temporal_X], [y], sample_weight=[sample_weight], classes=self.classes)

        self.active_state_id = self.classifier.active_state_id 

        detected_drift = self.perform_drift_detection_for_bounds(initial_state)

        # We have three reasons for attempting a check: We detected a drift, we want to evaluate the current state, or we have a propagating transition attempt.
        # If we have detected a drift, we consider state similarity directly.
        # Otherwise, we behave probabalistically.
        #<TODO> combine these approaches cleanly
        if detected_drift is not None:
            self.perform_transition(self.classifier.get_active_state().id)

        self.state_repository = self.classifier.state_repository
        self.record_transition(detected_drift, initial_state)

        # Set exposed info for evaluation.
        self.active_state = self.active_state_id
        self.found_change = detected_drift
        self.detected_drift = detected_drift
        self.states = self.state_repository
        self.current_sensitivity = 0.05


    def add_to_transition_matrix(self, init_s, curr_s, matrix):
        if init_s not in matrix:
            matrix[init_s] = {}
            matrix[init_s]['total'] = 0
        matrix[init_s][curr_s] = matrix[init_s].get(curr_s, 0) + 1
        matrix[init_s]['total'] += 1
        return matrix

    def record_transition(self, detected_drift, initial_state):
        """ Record an observation to observation level transition from the initial_state to the current state.
        Depends on if a drift was detected, or if in warning which records are updated.
        """
        current_state = self.classifier.active_state_id
        # We always add to the standard matrix
        self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_standard)

        # If we are in a warning period we add to the warning, and if we have detected drift we add to the warning and drift matricies
        if self.in_warning:
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_warning)
        elif detected_drift is not None:
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_warning)
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_drift)


