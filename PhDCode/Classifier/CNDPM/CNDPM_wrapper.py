import warnings
import numpy as np
import pathlib
import yaml
from PhDCode.Classifier.CNDPM.models import NdpmModel
from tensorboardX import SummaryWriter
import torch

from skmultiflow.utils import check_random_state, get_dimensions



warnings.filterwarnings('ignore')

class ConceptState:
    def __init__(self, id):
        self.id = id
        self.current_evolution = 0

class CNDPMWrapperClassifier:
    def __init__(self,
                 suppress=False,
                 learner=None,
                 poisson=10,
                 window_size=50,
                 init_concept_id=0,
                 concept_limit = 10, memory_management = 'score',
                sensitivity = 0.05,
                run_non_active_states = False, concept_chain = None, optimal_selection = False, optimal_drift = False, rand_weights = True,
                similarity_measure = 'KT', merge_strategy = "sur", use_clean = True, merge_similarity = 0.9, drift_detector = 'adwin',
                use_prior = True,
                batch_learning=False,
                cndpm_use_large=False):

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

        # Whether or not to use the CNDPM prior. The prior hinders performance in some but not all streaming tasks.
        self.use_prior = use_prior

        # Whether or not to store up batches to emulate original CNDPM
        self.batch_learning = batch_learning
        if self.batch_learning:
            raise ValueError("The batch learning option does not currently work with CNDPM. Learns with batches by default in Sleep phase.")
        # If we do batch learning, use the default size of 10 used in the original paper
        self.batch_size = 10 if self.batch_learning else 1
        self.batch = []

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

        self.cndpm_use_large = cndpm_use_large
        # Default config
        # self.config_path = pathlib.Path(__file__).parent / "configs" / "mlp_classifier-mlp_vae-split_mnist.yaml"
        # self.config_path = pathlib.Path(__file__).parent / "configs" / "mnist-cndpm.yaml"
        # self.config_path = pathlib.Path(__file__).parent / "configs" / "cndpm-ADL.yaml"
        # self.config_path = pathlib.Path(__file__).parent / "configs" / "cndpm-ADL-small.yaml"
        # self.config_path = pathlib.Path(__file__).parent / "configs" / "cndpm-ADL-small-nosleep.yaml"
        if not self.cndpm_use_large:
            self.config_path = pathlib.Path(__file__).parent / "configs" / "cndpm-ADL-small-nosleep-batch.yaml"
        else:
            self.config_path = pathlib.Path(__file__).parent / "configs" / "cndpm-ADL-norm-nosleep-batch.yaml"
        with open(self.config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.writer = SummaryWriter(".\\experimentlog")
        self.classifier = None



        # set up repository
        self.state_repository = {}
        self.active_state_id = 0
        self.last_assignment = 0
        next_state = ConceptState(0)
        self.state_repository[next_state.id] = next_state

        self.observations_since_last_transition = 0

        self.force_transition_to = None
        self.force_transition = False
        
        self.merges = {}
        self.deactivated_states = {}

        self.in_warning = False

        self.fingerprint_type = NdpmModel

        self.current_evolution = 0

        self.monitor_all_state_active_similarity = None
        self.monitor_all_state_buffered_similarity = None
        self.monitor_active_state_active_similarity = None
        self.monitor_active_state_buffered_similarity = None
        self.monitor_feature_selection_weights = None
        self.state_relevance = {}





    def get_active_state(self):
        return self.state_repository[self.active_state_id]

    def reset(self):
        pass

    def get_temporal_x(self, X):
        return np.concatenate([X], axis=None)

    def predict(self, X, using_batch=False):
        """
        Predict using the model of the currently active state.
        """
        if self.classes is None:
            return [0]
        temporal_X = self.get_temporal_x(X)
        if self.classifier is None:
            return [self.classes[0]]
        # print(f"n Experts: {len(self.classifier.ndpm.experts)}")
        if len(self.classifier.ndpm.experts) == 1:
            logits = self.classifier.ndpm.experts[-1].forward(torch.from_numpy(temporal_X).view([self.batch_size if using_batch else 1, 1, -1, 1]).float())[0]
            assignments = 0
            self.last_assignment = assignments
        else:
            logits, assignments = self.classifier(
                                torch.from_numpy(temporal_X).view([self.batch_size if using_batch else 1, 1, -1, 1]).float(), return_assignments=True)
            top_assign_vals, top__assign_idxs = assignments.topk(1, dim=0)
            self.last_assignment = assignments.tolist()[0]

        top_vals, top_idxs = logits.topk(1, dim=0)
        # print(f"Logits: {logits}, assignments: {assignments}")
        # print(f"Top Logits: {top_idxs}, Top assignments: {self.last_assignment}")
        return [self.classes[top_idxs.tolist()[0]]]

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
        # return False
        return initial_state != self.last_assignment

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
        is_new_state = to_id not in self.state_repository
        self.active_state_is_new = is_new_state
        self.active_state_id = to_id
        if is_new_state:
            n_state = ConceptState(to_id)
            self.state_repository[to_id] = n_state

        # Set triggers to future evaluation of this attempt
        self.observations_since_last_transition = 0

    def _partial_fit(self, X, y, sample_weight, masked=False):

        # init defaults for trackers
        found_change = False
        self.detected_drift = False
        self.warning_detected = False
        self.observations_since_last_transition += 1
        initial_state = self.active_state_id
        if self.classes is None:
            raise ValueError("Classes not initialized")
        y_idx = self.classes.index(y)

        if self.batch_learning:
            if len(self.batch) < self.batch_size:
                self.batch.append((X, y_idx))
                return
            else:
                batch_inputs = [b[0] for b in self.batch]
                batch_labels = [b[1] for b in self.batch]
                X = np.vstack(batch_inputs)
                y_idx = np.vstack(batch_labels)

        # get_temporal_x, and get_imputed_label are to deal with masked
        # values where we don't see true label.
        # As the functions are now, they don't do anything extra.
        # But could be extended to reuse last made prediction as
        # the label for example.

        temporal_X = self.get_temporal_x(X)

        foreground_p = self.predict(X, using_batch=self.batch_learning)
        # print(foreground_p, y, y_idx, foreground_p==y)
        # self.get_active_state().partial_fit(temporal_X, y, sample_weight=sample_weight, classes=self.classes)
        # self.classifier.partial_fit([temporal_X], [y], sample_weight=[sample_weight], classes=self.classes)
        if self.classifier is None:
            self.config['x_h'] = X.shape[0] if not self.batch_learning else X.shape[1]
            self.config['y_c'] = len(self.classes)
            self.config['batch_size'] = self.batch_size

            self.classifier = NdpmModel(self.config, self.writer, use_prior=self.use_prior)
        if self.batch_size == 1:
            self.classifier.learn(torch.from_numpy(X).view(self.batch_size, 1, -1, 1).float(), torch.from_numpy(np.array([y_idx])).to(torch.int64), -1, self.ex)
        else:
            y_torch = torch.from_numpy(y_idx).to(torch.int64).view(-1)
            self.classifier.learn(torch.from_numpy(X).view(self.batch_size, 1, -1, 1).float(), y_torch, -1, self.ex)

        detected_drift = self.perform_drift_detection_for_bounds(initial_state)

        # We have three reasons for attempting a check: We detected a drift, we want to evaluate the current state, or we have a propagating transition attempt.
        # If we have detected a drift, we consider state similarity directly.
        # Otherwise, we behave probabalistically.
        #<TODO> combine these approaches cleanly
        if detected_drift is not None:
            self.perform_transition(self.last_assignment)
            # pass

        self.state_repository[self.active_state_id].current_evolution = 0
        self.record_transition(detected_drift, initial_state)

        # Set exposed info for evaluation.
        self.active_state = self.active_state_id
        self.found_change = detected_drift
        self.detected_drift = detected_drift
        self.states = self.state_repository
        self.current_sensitivity = 0.05

        # self.merges = self.classifier.fsm.observed_merges
        # self.deletions = self.classifier.fsm.observed_deletions
        # self.state_relevance = self.classifier.state_relevance


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
        current_state = self.active_state_id
        # We always add to the standard matrix
        self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_standard)

        # If we are in a warning period we add to the warning, and if we have detected drift we add to the warning and drift matricies
        if self.in_warning:
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_warning)
        elif detected_drift is not None:
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_warning)
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_drift)
    
    def reset_stats(self, rem_state_log):
        pass
        # self.classifier.reset_stats(rem_state_log=rem_state_log)


