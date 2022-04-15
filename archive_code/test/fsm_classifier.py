from skmultiflow.utils import get_dimensions, normalize_values_in_dict, check_random_state, check_weights
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.ddm import DDM
from fsm.fsm import FSM
from fsm.fsm_clean import FSM_CLEAN
from fsm.systemStats import systemStats
from copy import deepcopy
import time
import numpy as np
import pickle

def make_detector(warn = False, s = 1e-5, drift_detector = 'adwin'):
    sensitivity = s * 10 if warn else s
    if drift_detector == 'adwin':
        return ADWIN(delta=sensitivity)
    if drift_detector == 'EDDM':
        return EDDM()
    if drift_detector == 'DDM':
        return DDM()

class FSMClassifier:
    def __init__(self, suppress = True,
    concept_limit = 10, memory_management = 'score', learner = None, window = 50,
    sensitivity = 0.05,
 run_non_active_states = False, concept_chain = None, optimal_selection = False, optimal_drift = False, rand_weights = True, poisson = 3,
 similarity_measure = 'KT', merge_strategy = "sur", use_clean = False, merge_similarity = 0.9, drift_detector = 'adwin'):

        if learner is None:
            raise ValueError('Need a learner')
        self.concept_limit = concept_limit
        self.memory_management = memory_management
        self.learner = learner
        self.window = window
        self.sensitivity = sensitivity
        self.suppress = suppress
        self.run_non_active_states = run_non_active_states
        self.concept_chain = concept_chain
        self.optimal_selection = optimal_selection
        self.optimal_drift = optimal_drift
        self.rand_weights = rand_weights
        self.poisson = poisson
        self.merge_similarity = merge_similarity
        self.drift_detector = drift_detector

        self.detector_init = make_detector(s=sensitivity, drift_detector = self.drift_detector)
        self.warn_detector_init = make_detector(warn=True,s=sensitivity, drift_detector = self.drift_detector)

        self.detector = deepcopy(self.detector_init)
        self.warn_detector = deepcopy(self.warn_detector_init)

        self.in_warning = False
        if not use_clean:
            self.fsm = FSM(concept_limit =concept_limit, memory_strategy=memory_management, merge_strategy = merge_strategy, merge_similarity = merge_similarity)
        else:
            self.fsm = FSM_CLEAN(concept_limit =concept_limit, memory_strategy=memory_management, merge_strategy = merge_strategy, merge_similarity = merge_similarity)
        self.fsm.suppress = suppress
        # Initialize components
        starting_id = 0
        if not (concept_chain is None) and 0 in concept_chain and (self.optimal_selection):
            starting_id = concept_chain[0]
        
        self.fsm.make_state(learner, s_id = starting_id)
        self.fsm.active_state_id = starting_id

        self.system_stats = systemStats()
        self.system_stats.state_control_log.append([0, 0, None])
        self.system_stats.state_control_log_altered.append([0, 0, None])
        self.system_stats.last_seen_window_length =window

        self.random_state = None
        self._random_state = check_random_state(self.random_state)
        self.cancelled = False
        self.percent_start = time.process_time()
        self.stream_examples = []
        # Run the main loop
        self.ex = -1
        self.state_recurrence_checks = 0
        if optimal_selection or optimal_drift:
            print(concept_chain)
            self.optimal_alias = {}
            self.drift_matrix = {}
            self.gt_order = {concept_chain[0]: 0}
        
        self.classes = None
        self._train_weight_seen_by_model = 0
        self.similarity_measure = similarity_measure
        
        self.restore_state = None
        

    def reset(self):
        pass

    def partial_fit(self, X, y, classes = None, sample_weight = None):
        """
        Fit an array of observations. Splits input into individual observations and
        passes to a helper function _partial_fit. Randomly weights observations depending on 
        Config.
        """
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.format(row_cnt,
                                                                                                  len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    self.ex += 1
                    if self.rand_weights and self.poisson >= 1:
                        # k = self._random_state.poisson(self.poisson)
                        k = self.poisson
                        sample_weight[i] = k
                    self._partial_fit(X[i], y[i], sample_weight[i])

    def add_eval(self, X, y, prediction, correctly_classifies, ex):
        self.system_stats.add_prediction(X, y, prediction, correctly_classifies, ex)
        self.fsm.get_state().add_main_prediction(X, y, prediction, correctly_classifies, ex)

    def _partial_fit(self, X, y, sample_weight):
        """
        Partially fit on a single observation.
        """

        signal_backtrack = self.fsm.get_state().signal_backtrack
        if signal_backtrack:
            old_state = self.fsm.get_state()
            old_transition_point = old_state.shadow_transition_point
            old_transition_from = old_state.shadow_transition_from
            
            print(f"old ID {old_state.id}")
            shadow_state = old_state.shadow_state
            print(f"shadow ID {shadow_state.id}")
            new_state_id = self.fsm.make_state(self.learner)
            print(f"new ID {new_state_id}")

            shadow_state.id = new_state_id


            self.fsm.states[new_state_id]= deepcopy(shadow_state)
            self.fsm.states[new_state_id].id = new_state_id

            r_state = deepcopy(self.restore_state)
            print(f"r_state id {r_state.id}")
            if r_state.id != old_state.id:
                print("PROBLEM")
                exit()
            self.fsm.states[r_state.id] = r_state


            self.fsm.undo_transition(old_transition_point, old_transition_from, old_state.id)

            self.fsm.transition_to(new_state_id, old_transition_point)

            self.system_stats.log_change_detection(self.ex)

            # Set end of last state
            self.system_stats.state_control_log[-1][2] = self.ex
            # state new state log
            self.system_stats.state_control_log.append([self.fsm.active_state_id, self.ex, None])

            # On altered record we revise the last transition
            self.system_stats.state_control_log_altered[-1][0] = self.fsm.active_state_id




            # We reset the detector and model structure tracking, as the previous models accuracy
            # should not affect new changes.
            self.detector = make_detector(s=self.sensitivity, drift_detector = self.drift_detector)
            self.warn_detector = make_detector(warn=True,s=self.sensitivity, drift_detector = self.drift_detector)
            self.in_warning = False
            self.system_stats.clear_warn_log()


        # Predict before we fit, to detect drifts.
        prediction = self.fsm.get_state().main_model.predict(np.asarray([X]))[0]
        correctly_classifies = prediction == y

        # Fit main model to observation.
        self.fsm.get_state().main_model.partial_fit(np.asarray([X]), np.asarray([y]), sample_weight = np.asarray([sample_weight]))
        


        self.system_stats.add_prediction(X, y, prediction, correctly_classifies, self.ex)
        self.fsm.get_state().add_main_prediction(X, y, prediction, correctly_classifies, self.ex)

        

        
        
        # Handle transitions. We skip drift if using optimal selection.
        skip_selection = self.concept_chain != None and self.optimal_selection
        found_change = False
        know_optimal = False
        if (not skip_selection) and (not self.optimal_drift):
            # Add to detector, and get any alerts
            self.detector.add_element(int(not correctly_classifies))
            self.warn_detector.add_element(int(not correctly_classifies))
            found_change = self.detector.detected_change()

            if self.warn_detector.detected_change() or self.warn_detector.detected_warning_zone():
                self.in_warning = True
                self.system_stats.clear_warn_log()
                self.warn_detector = make_detector(warn=True,s=self.sensitivity, drift_detector = self.drift_detector)

            if self.in_warning:
                self.system_stats.add_warn_prediction(X, y, prediction, correctly_classifies, self.ex)
        else:
            # If the current observation is a switch point, find the alias of the transition.
            # Aliases occur when we drop a state and pick it back up later. 
            # The later state will have a different ID than in our concept chain.
            # We store a linked list of IDs that relate to the same base concept.
            
            # if just optimal drifting, we need the window to check for recurrences.
            check_ex = self.ex
            if self.optimal_drift:
                check_ex = self.ex - self.window

            if check_ex in self.concept_chain and check_ex != 0:
                found_change = True
                switch_to_id = self.concept_chain[check_ex]

                # Traverse the alias linked list until we find the most recent id to check for.
                while switch_to_id in self.optimal_alias:
                    if self.optimal_alias[switch_to_id] == switch_to_id:
                        break
                    switch_to_id = self.optimal_alias[switch_to_id]
                know_optimal = True
                    
        gt_alias = 0

        if(found_change):
            
            self.state_recurrence_checks += 1
            print(" ")
            print(f"change detected at {self.ex}")

            if not self.concept_chain is None:
                Last_ccex = None
                for ccex in self.concept_chain:
                    if ccex > self.ex:
                        print(f"Real cc: {Last_ccex}, {ccex}")
                        if know_optimal:
                            gt_alias = len(list(self.gt_order.keys()))
                            if switch_to_id in self.gt_order:
                                gt_alias = self.gt_order[switch_to_id]
                            else:
                                self.gt_order[switch_to_id] = gt_alias
                            print(f"Real Transition: {gt_alias}")
                        break
                    Last_ccex = ccex
            # We take a recent window between warning and drift, with a minimum.
            recent_window = self.system_stats.warn_log
            if len(recent_window) < self.window or len(recent_window) > 1000:
                recent_window = self.system_stats.last_seen_examples
            
            if not skip_selection:
                # Check for state to transition to using FSM rules.
                shadow = None
                if self.similarity_measure == 'PREVACC':
                    new_state_id, is_recurring,shadow = self.fsm.get_recurring_state_or_new_prev_acc_comparison(recent_window, self.learner, self.system_stats)
                elif self.similarity_measure == 'AD':
                    new_state_id, is_recurring,shadow = self.fsm.get_AD_reccurence(recent_window, self.learner, self.system_stats)
                elif self.similarity_measure == 'KT':
                    new_state_id, is_recurring,shadow = self.fsm.get_KT_reccurence(recent_window, self.learner, self.system_stats)
                else:
                    # new_state_id, is_recurring, shadow = self.fsm.get_recurring_state_or_new(recent_window, self.learner, self.system_stats)
                    new_state_id, is_recurring,shadow = self.fsm.get_KT_reccurence(recent_window, self.learner, self.system_stats)
                if new_state_id != self.fsm.active_state_id:
                    self.restore_state = deepcopy(self.fsm.states[new_state_id])
                    shadow_state = self.fsm.construct_state(-1, self.learner)
                    shadow_state.main_model = shadow
                    self.fsm.transition_to(new_state_id, self.ex, shadow_state)
                
            else:
                
                # Check if the state to transition to is currently in the repo
                is_recurring = switch_to_id in [s.id for s in self.fsm.states.values()]
                if not is_recurring:

                    # If not, make a new state and add its ID to the alias list.
                    new_id = self.fsm.make_state(self.learner, s_id=switch_to_id)
                    self.fsm.transition_to(new_id, self.ex)
                    self.optimal_alias[switch_to_id] = new_id
                    new_state_id = new_id
                else:
                    self.fsm.transition_to(switch_to_id, self.ex)
                    new_state_id = switch_to_id
            self.fsm.cull_states(recent_window)

            
            # Logging
            if not self.suppress: print(f"Transitioned to {new_state_id}")
            print(f"Transitioned to {new_state_id}")
            if know_optimal:
                print(f"Real Transition: {gt_alias}, correct {new_state_id == gt_alias}")
            # if self.know_optimal:
            #     if new_state_id not in self.drift_matrix:
            #         self.drift_matrix{}
            self.system_stats.log_change_detection(self.ex)

            # Set end of last state
            self.system_stats.state_control_log[-1][2] = self.ex
            self.system_stats.state_control_log_altered[-1][2] = self.ex

            # state new state log
            self.system_stats.state_control_log.append([self.fsm.active_state_id, self.ex, None])
            self.system_stats.state_control_log_altered.append([self.fsm.active_state_id, self.ex, None])

            # We reset the detector and model structure tracking, as the previous models accuracy
            # should not affect new changes.
            self.detector = make_detector(s=self.sensitivity, drift_detector = self.drift_detector)
            self.warn_detector = make_detector(warn=True,s=self.sensitivity, drift_detector = self.drift_detector)
            self.in_warning = False
            self.system_stats.clear_warn_log()
            if(hasattr(self.fsm.get_state().main_model, 'splits_since_reset')):
                self.system_stats.model_update_status = self.fsm.get_state().main_model.splits_since_reset

        # We want to track splits in the hoeffding tree.
        # Splits effect accuracy (in either direction), which can trigger
        # a change detection even if no drift is present. So we reset.
        if( hasattr(self.fsm.get_state().main_model, 'splits_since_reset') and self.fsm.get_state().main_model.splits_since_reset > self.system_stats.model_update_status):
            self.system_stats.log_model_update(self.ex, self.fsm.get_state().main_model.splits_since_reset)
            # self.detector = make_detector(s=self.sensitivity, drift_detector = self.drift_detector)
            # self.warn_detector = make_detector(warn=True,s=self.sensitivity, drift_detector = self.drift_detector)
            self.fsm.main_state_evolve(self.ex)
            
        
    def predict(self, X):
        """
        Predict using the model of the currently active state.
        """
        return self.fsm.get_state().main_model.predict(X)
    
    def reset_stats(self):
        """
        Reset logs of states, (Call after they have been writen)
        """
        for s in self.fsm.states.values():
            s.comparison_model_stats.sliding_window_accuracy_log = []
            s.comparison_model_stats.correct_log = []
            s.comparison_model_stats.p_log = []
            s.comparison_model_stats.y_log = []
            s.main_model_stats.sliding_window_accuracy_log = []
            s.main_model_stats.correct_log = []
            s.main_model_stats.p_log = []
            s.main_model_stats.y_log = []
        self.system_stats.model_stats.sliding_window_accuracy_log = []
        self.system_stats.model_stats.correct_log = []
        self.system_stats.model_stats.p_log = []
        self.system_stats.model_stats.y_log = []

    def finish_up(self, ex):
        self.system_stats.state_control_log[-1][2] = ex
        self.system_stats.state_control_log_altered[-1][2] = ex
        return self.fsm.merge_log


    
    