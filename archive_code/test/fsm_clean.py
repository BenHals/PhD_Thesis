from collections import Counter
from collections import namedtuple
from collections import deque
from graphviz import Digraph
import sys
import math
from .modelStats import modelStats
import numpy as np
sys.path.append('../similarityMeasure')
from .similarityMeasure.statsTests import knnTest
from .similarityMeasure.statsTests import quantTreeTest
from adaptiveLearner.tracksplit_hoeffding_tree import TS_HoeffdingTree
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from multiprocessing import Pool
from skmultiflow.drift_detection.adwin import ADWIN
from copy import deepcopy

class State:
    """A state in an FSM
    
    Parameters
    ----------
    id: int
        The unique id of the state
    
    learner: StreamModel generator
        A funtion which generates a StreamModel, which the state will use as a model
    """
    # Estimate for accuracy of prediction
    Q = 1e-5

    # Estimate for accuracy of new measurement
    R = 1e-4

    # Learning difficulty to reuse advantage proportionality constant
    GAMMA = 0.0017
    def __init__(self, id, learner):
        self.id = id                            # The ID of the state.
        self.main_model = learner()             # The main model, only trained when active.
        self.main_model_stats = modelStats(id, 'main')
        self.comparison_model = learner()       # A model to use as comparison, always trained.
        self.comparison_model_stats = modelStats(id, 'comp')
        self.evolution = [[0, 0, None, None, None]]                     # Tracks the evolution of the model.

        self.reuse_log = {
            'Q': 1e-2,
            'R': 1e-6,
            'transitions_to_model': 0,
            'transitions_not_to_model': 0,
            'future_proportion_estimate': 1,
            'error_estimate': 1,
            'fading_average': 0,
        }

        self.shadow_state = None
        self.shadow_model_stats = modelStats(id, 'shadow')

        self.restore_state = None
        self.signal_backtrack = False
        self.shadow_transition_point = None
        self.shadow_transition_from = None
        self.backtrack_better_count = 0

        self.temporal_classifier_prediction = None
        self.majority_classifier_prediction = None
        self.seen_y_proportions = Counter()
        self.recent_window = deque()
        self.overall_seen_stats = {'c': 0, 'w': 0, 'ktw': 0, 'kmw': 0, 'kac': 0, 'seen': 0}

        self.evolution_stats_generating = True
        self.evolution_samples_length = 300
        self.evolution_samples_seen = 0
        self.evolution_stats = {'acc': 0, 'kt': 0, 'km': 0, 'ka': 0}

        self.age = 0
        self.LRU = 0
        self.last_LRU = 0


    def __repr__(self):
        return f"<{self.id}: eR: {self.estimated_reuse_proportion}, eA: {self.estimated_reuse_advantage}, rA: {self.estimated_total_advantage}"
    
    def add_main_prediction(self, X, y, p, is_correct, ts):
        """
        Call on making a new prediction. Adds accuracy to state logs.
        """
        self.main_model_stats.add_prediction(X, y, p, is_correct, ts)

        # Check accuracy of the untrained shadow state, and indicate swap if needed
        if not self.shadow_state is None:
            shadow_pred = self.shadow_state.main_model.predict(np.asarray([X]))
            shadow_correct = shadow_pred == y
            self.shadow_state.main_model.partial_fit(np.asarray([X]), np.asarray([y]), sample_weight = [10])
            self.shadow_state.main_model_stats.add_prediction(X, y, shadow_pred, shadow_correct, ts)

            if len(self.shadow_state.main_model_stats.sliding_window) > 300:
                if self.shadow_state.main_model_stats.sliding_window_accuracy_log[-1][1] > self.main_model_stats.sliding_window_accuracy_log[-1][1] + 0.05:
                    self.backtrack_better_count += 1
                else:
                    self.backtrack_better_count = 0
                
                if self.backtrack_better_count > 30:
                    print("**************")
                    print(f"SHADOW BETTER : {self.shadow_state.main_model_stats.sliding_window_accuracy_log[-1][1]} : {self.main_model_stats.sliding_window_accuracy_log[-1][1]}")
                    print("**************")
                    self.signal_backtrack = True
        
        # Check accuracy of a temporal baseline and update temporal prediction for next check
        if self.temporal_classifier_prediction is None:
            self.temporal_classifier_prediction = p
        temporal_right = self.temporal_classifier_prediction == y
        self.temporal_classifier_prediction = y

        # Check accuracy of a majority baseline and update majority prediction for next check
        if self.majority_classifier_prediction is None:
            self.majority_classifier_prediction = p
        majority_right = self.majority_classifier_prediction == y
        self.seen_y_proportions[y] += 1
        if self.seen_y_proportions[y] > self.seen_y_proportions[self.majority_classifier_prediction]:
            self.majority_classifier_prediction = y

        self.recent_window.append((is_correct, temporal_right, majority_right, p, y))
        if len(self.recent_window) > 1500:
            self.recent_window.popleft()
        self.overall_seen_stats['seen'] += 1
        if is_correct:
            self.overall_seen_stats['c'] += 1
            kac = True
            if temporal_right:
                self.overall_seen_stats['ktw'] += 1
                kac = False

            if majority_right:
                self.overall_seen_stats['kmw'] += 1
                kac = False
            
            if kac:
                self.overall_seen_stats['kac'] += 1
        else:
            self.overall_seen_stats['w'] += 1

        if self.evolution_stats_generating:
            self.evolution_samples_seen += 1
            if self.evolution_samples_seen >= self.evolution_samples_length:
                # print("Seen 300")
                self.set_evolution_stats()
                self.evolution_stats_generating = False

        
    def set_evolution_stats(self):
        ts = self.overall_seen_stats['seen']
        window_since_evolution = list(self.recent_window)[-self.evolution_samples_seen:]
        if len(window_since_evolution) == 0:
            if len(self.evolution) >= 2:
                self.evolution_stats = self.evolution[-2][3]
        else:
            self.evolution_stats['acc'] = sum((x[0] for x in window_since_evolution)) / len(window_since_evolution)
            self.evolution_stats['kt'] = sum((x[0] and not x[1] for x in window_since_evolution)) / len(window_since_evolution)
            self.evolution_stats['km'] = sum((x[0] and not x[2] for x in window_since_evolution)) / len(window_since_evolution)
            self.evolution_stats['ka'] = sum((x[0] and not x[1] and not x[2] for x in window_since_evolution)) / len(window_since_evolution)

        self.evolution[-1][2] = ts
        self.evolution[-1][3] = self.evolution_stats
        self.evolution[-1][4] = self.get_AAC_measurement(ts, self.evolution_stats['kt'], self.evolution_stats['km'])
        




    def calculate_past_score_advantage(self):
        """
        Calculate a measurement of how good reusing this state is
        compared to retraining an alternative from scratch.
        Uses the max of recent accuracy measurement scores to calculate 
        an estimated number of correct predictions if reused.
        Compares this to the number of correct predictions made while training.
        The difference gives a 'score' for reuse over retrain.
        """
        if len(self.recent_window) == 0:
            recent_score = 0
        else:
            recent_score = sum((x[0] and not x[1] and not x[2] for x in self.recent_window)) / len(self.recent_window)

        past_total = self.overall_seen_stats['c'] + self.overall_seen_stats['w']

        reuse_right = recent_score * past_total

        retrain_right = self.overall_seen_stats['kac']

        return reuse_right - retrain_right

    def add_comparison_prediction(self, X, y, p, is_correct, ts):
        self.comparison_model_stats.add_prediction(X, y, p, is_correct, ts)
    
    def add_proportion_measurement(self):
        """Update our estimate of future proportion of the stream taken up by this state using a Kalman filter"""
        if (self.reuse_log['transitions_to_model'] + self.reuse_log['transitions_not_to_model']) > 0:
            proportion_since_active = self.reuse_log['transitions_to_model'] / (self.reuse_log['transitions_to_model'] + self.reuse_log['transitions_not_to_model'])
        else:
            proportion_since_active = 1

        last_p = self.reuse_log['error_estimate'] + self.reuse_log['Q']

        if (last_p + self.reuse_log['R']) > 0:
            k = last_p / (last_p + self.reuse_log['R'])
        else:
            print("Bad, 0 reuse R")
            k = 1
        estimate = self.reuse_log['future_proportion_estimate'] + k * (proportion_since_active - self.reuse_log['future_proportion_estimate'])

        self.reuse_log['future_proportion_estimate'] = estimate
        self.reuse_log['error_estimate'] = (1 - k) * last_p

    def get_AAC_measurement(self, ex, k_t, k_m):
        """Update our estimate of future proportion of the stream taken up by this state using a Kalman filter.
        Also update temporal and majority comparisons.
        Also build the 'auc' measure, of reuse vs retrain accuracy.
        """

        # Average recent temporal and majority kappa to get a combined accuracy score
        score = (k_t + k_m) / 2

        # Use this score to estimate 'auc' between reusing and retraining a model.
        # Basically if score is higher than max, it adds the area of the rectangle + triangle,
        # If lower by a small amount no change,
        # If lower by a large amount diminishes the whole 'auc'
        last_ex = 0
        last_score = 0
        last_AAC = 0
        if len(self.evolution) > 1:
            last_ex = self.evolution[-2][2]
            last_stats = self.evolution[-2][3]
            last_score = (last_stats['kt'] + last_stats['km']) / 2
            last_AAC = self.evolution[-2][4]
        AAC = last_AAC
        accuracy_delta = score - last_score
        score = max(score, last_score)
        if accuracy_delta > 0:
            ex_delta = ex - last_ex
            acc_inc = (accuracy_delta * last_ex) + ((ex_delta * abs(accuracy_delta)) / 2)
            future_potential = (1 - score) + 1
            acc_inc *= future_potential
            
            AAC += acc_inc
        else:
            if score > 0:
                if (abs(accuracy_delta) / score) > 0.05:
                    AAC = AAC * 0.99
        
        return AAC
    
    def calculate_total_advantage(self, recent_states):
        """
        Estimate future reuse, and future advantage to estimate total advantage 'rA' measure.
        """
        # Get current Kalman estimate (without setting a new measurement)
        if (self.reuse_log['transitions_to_model'] + self.reuse_log['transitions_not_to_model']) > 0:
            proportion_since_active = self.reuse_log['transitions_to_model'] / (self.reuse_log['transitions_to_model'] + self.reuse_log['transitions_not_to_model'])
        else:
            print("Bad, 0 trans")
            proportion_since_active = 0
        
        prop_recent = sum(map(lambda x: 1 if x == self.id else 0, recent_states)) / 100
        prop_recent = max(prop_recent, 0.1)

        last_p = self.reuse_log['error_estimate'] + self.reuse_log['Q']
        k = last_p / (last_p + self.reuse_log['R'])
        
        estimate = self.reuse_log['future_proportion_estimate'] + k * (prop_recent - self.reuse_log['future_proportion_estimate'])

        # if self.reuse_log['transitions_to_model'] + self.reuse_log['transitions_not_to_model'] > 3:
        #     self.estimated_reuse_proportion = estimate
        # else:
        #     self.estimated_reuse_proportion = 1
        self.estimated_reuse_proportion = estimate
        self.estimated_reuse_advantage = State.GAMMA * (len(self.evolution) + 1)

        self.estimated_total_advantage = self.estimated_reuse_proportion * self.estimated_reuse_advantage

        return self.estimated_total_advantage



class Transition:
    """A transition in an FSM"""
    def __init__(self):
        self.to_ids = Counter()                 # IDs of states this state transitions to.
        self.from_ids = Counter()               # The IDs of states which transition to this state.
        self.in_edges = 0                       # The number of incoming edges.
        self.out_edges = 0                      # The number of outgoing edges.

class FSM_CLEAN:
    """Represents the evolution of a datastream as an FSM."""
    def __init__(self, concept_limit = 15, save_states = False, memory_strategy = "rA", merge_strategy = 'sur', merge_similarity = 0.9):
        # Transitions for each state.
        # Key: state_id, Value: Transition tuple
        self.transitions = {}

        # A list of State tuples. 
        # Represents the model repository.
        self.states = {}

        self.active_state_id = 0
        self.suppress = True
        self.concept_limit = concept_limit
        self.ID_counter = 0
        self.state_log = []
        self.save_states = save_states
        self.merge_log = {}
        self.memory_strategy = memory_strategy
        self.deleted_states = []
        self.recent_states = deque()
        self.merge_similarity = merge_similarity

        self.merge_strategy = merge_strategy


    def construct_state(self, state_id, learner):
        return State(state_id, learner)

    def make_state(self, learner, s = None, s_id = None):
        """Adds a new state to the system. Only call when the state is unique
        
        Parameters
        ----------
        learner: StreamModel
            A learning algorithm to use as this node's model.
        
        s: State
            An optional state, if not specified a new one is created.
        Returns
        -------
        int
            The ID of the new state.
        """
        
        state_id = self.ID_counter if s_id is None else s_id
        self.ID_counter += 1
        while state_id in [s.id for s in self.states.values()] or state_id in self.deleted_states:
            state_id = self.ID_counter
            self.ID_counter += 1

        if s == None:
            s = self.construct_state(state_id, learner)
            # s.reuse_log['future_proportion_estimate'] = 1 / (len(list(self.states.keys())) + 1)
            if len(self.states.values()) > 0:
                # s.reuse_log['future_proportion_estimate'] = sum([x.reuse_log['future_proportion_estimate'] for x in self.states.values()]) / len(self.states.values())
                s.reuse_log['future_proportion_estimate'] = 1 / 100
            else:
                s.reuse_log['future_proportion_estimate'] = 1
        else:
            s.id = state_id
        
        self.states[state_id] = s
        if self.save_states:
            self.state_log.append(s)
        self.transitions.setdefault(state_id, Transition())
        return state_id

    def undo_transition(self, ex, from_id, to_id):
        self.transitions[from_id].to_ids[to_id] -= 1
        self.transitions[from_id].out_edges -= 1
        self.transitions[to_id].from_ids[from_id] -= 1
        self.transitions[to_id].in_edges -= 1
        self.states[to_id].LRU = self.states[to_id].last_LRU
        
    def transition_to(self, to_id, ex = 0, shadow = None):
        from_id = self.active_state_id
        self.add_transition(self.active_state_id, to_id)
        self.recent_states.append(to_id)
        if len(self.recent_states) > 100:
            self.recent_states.popleft()

        self.get_state().shadow_model_stats = None
        self.get_state().shadow_state = None
        self.get_state().restore_state = None

        self.active_state_id = to_id
        for s_id in self.states.keys():
            s = self.states[s_id]
            s.age += 1
            if s.id == to_id:
                s.reuse_log['transitions_to_model'] += 1
                s.add_proportion_measurement()
                s.last_LRU = s.LRU
                s.LRU = 0
            else:
                s.reuse_log['transitions_not_to_model'] += 1
                s.LRU += 1
        
        self.get_state().shadow_model_stats = modelStats(id, 'shadow')
        self.get_state().shadow_state = shadow
        self.get_state().restore_state = deepcopy(self.get_state().main_model)
        self.get_state().shadow_transition_point = ex
        self.get_state().shadow_transition_from = from_id
        

    def add_transition(self, from_id, to_id):
        """Add a transition to the system. IDs should exist
        
        Parameters
        ----------
        from_id: int
            The state_id of the starting state.
            
        to_id: int
            The state_id of the ending state.
        """ 
        from_store = self.transitions.setdefault(from_id, Transition())
        to_store = self.transitions.setdefault(to_id, Transition())

        from_store.to_ids[to_id] += 1
        from_store.out_edges += 1
        to_store.from_ids[from_id] += 1
        to_store.in_edges += 1

    def main_state_evolve(self, ex):
        current_state = self.get_state()
        # print("evolving")
        current_state.set_evolution_stats()
        current_state.evolution_stats_generating = True
        current_state.evolution_samples_seen = 0
        current_state.evolution_stats = {'acc': 0, 'kt': 0, 'km': 0, 'ka': 0}
        current_state.evolution.append([ex, len(current_state.evolution), None, None, None])

    def render_fsm(self, directory, show):
        dot = Digraph(comment="FSM")
        dot.graph_attr['rankdir'] = 'LR'
        for node_id in self.transitions:
            if not self.suppress: print(str(node_id))
            dot.node(str(node_id), str(node_id))
            
        for node in list(map(lambda s: (s.id, s.evolution), self.state_log)):
            last_e_id = None
            e_counter = 0
            for e in node[1]:
                e_id = f"{node[0]}_e_{e_counter}"
                dot.node(e_id, e_id)
                if last_e_id != None:
                    dot.edge(last_e_id, e_id)
                last_e_id = e_id
                e_counter += 1

        for from_id in self.transitions:
            to_ids_counter = self.transitions[from_id].to_ids
            for to_id in to_ids_counter:
                dot.edge(str(from_id), str(to_id), str(round(to_ids_counter[to_id] / self.transitions[from_id].out_edges * 100) / 100))

        dot.render(f'{directory}\FSM.gv', view=show)
    
    def get_state(self):
        """Get the currently active FSM state.

        Returns
        -------
        State tuple
            The current active state.
        """
        return self.states[self.active_state_id]
    
    def is_current_state(self, state_id):
        """Returns if the passed state_id is the currently active state_id.

        Parameters
        ----------
        state_id: int
            State Id to check
        
        Returns
        -------
        Bool
            If the passed state ID matches the current active state ID.
        """
        return state_id == self.active_state_id


    def close_state_merge_check(self, scores, learner, system_stats, recent_X, recent_y, stat = 'accuracy'):
        merged_scores = []
        ss_i = 0
        while ss_i < len(scores):
            # print(ss_i)
            ss = scores[ss_i]
            state_id = ss[0]
            state_accuracy = ss[1]

            if state_id not in self.states:
                break

            state = self.states[state_id]

            next_index = ss_i + 1
            if next_index >= len(scores):
                merged_scores.append(ss)
                break
            
            next_state_id = scores[next_index][0]

            if next_state_id not in self.states:
                break

            next_state = self.states[next_state_id]
            nex_state_accuracy = scores[next_index][1]
            
            merged = False
            if abs(state_accuracy - nex_state_accuracy) < 0.05 and (state_id != self.active_state_id and next_state_id != self.active_state_id):
                print(f"Check for merge: {state_accuracy} : {nex_state_accuracy}")
                merge_state, merge_similarity = self.check_for_merge(state, next_state)
                print(f"Check for merge: {merge_state}")
                if merge_state:
                    if not self.suppress: self.suppress: print('MERGING!!')
                    merged = True
                    state = self.merge_states(state, next_state, learner, system_stats.feature_description, system_stats.seen_features, merge_similarity)
                    if state == None:
                        merged = False
                        if not self.suppress: print("merge failed")
                    
                    if merged:
                        state_id = state.id
                        # t_i, test_accuracy = get_recent_window_accuracy(state, recent_X, recent_y)
                        predictions = state.main_model.predict(recent_X)
                        test = list(zip(list(predictions), list(recent_y)))
                        accuracy, k_temporal_acc, k_majority_acc, k_s = get_stats(test)

                        return_stat = accuracy
                        if stat != 'accuracy':
                            return_stat = k_temporal_acc
                        print(stat)
                        print(f"{return_stat}, {accuracy}, {k_temporal_acc}")
                        ss = (state_id, return_stat, 0.79, accuracy)
                        ss_i = next_index + 1
            if not merged:
                ss_i = next_index
                    
            
            merged_scores.append(ss)
        print(f"original: {scores}, merged: {merged_scores}")
        if scores != merged_scores:
            print("DEBUG")
            # exit()
        return merged_scores

    def get_KT_reccurence(self, recent_window, learner, system_stats):
        t = 0.85
        shadow_t = 0.95
        shadow_m = 1.025

        print("*****")
        print(f"Length of window: {len(recent_window)}")
        close_concepts = []

        # Setup recent window
        n_samples = len(recent_window)
        if not self.suppress: print(n_samples)
        n_features = len(recent_window[0].X)
        recent_X = np.array([e.X for e in recent_window]).reshape(n_samples, n_features)
        recent_y = np.array([e.y for e in recent_window])

        # setup and train shadow state
        shadow_state = self.construct_state(-1, learner)
        shadow_results = []
        pretrain_c = 0
        pretrain_l = 50

        # Reverse the recent window, with the idea that later obs are cleaner.
        for X,y in zip(reversed([e.X for e in recent_window]), reversed([e.y for e in recent_window])):
            prediction = shadow_state.main_model.predict([X])[0]
            if pretrain_c > pretrain_l:
                shadow_results.append((prediction, y))
            shadow_state.main_model.partial_fit([X],[y])
            pretrain_c += 1
        shadow_results.reverse()
        # predictions = shadow_state.main_model.predict(recent_X)
        # shadow_results = list(zip(list(predictions), list(recent_y)))
        shad_acc, shad_k_t, shad_k_m, shad_k_s = get_stats(shadow_results)

        
        # Test each state on recent window.
        for s_i, state in enumerate(self.states.values()):
            is_active_state = state.id == self.active_state_id
            predictions = state.main_model.predict(recent_X)

            test = list(zip(list(predictions), list(recent_y)))
            if is_active_state:
                print(-len(list(predictions)))
                test = list(zip([x[3] for x in state.recent_window][-len(list(predictions)):], list(recent_y)))

            accuracy, k_temporal_acc, k_majority_acc, k_s = get_stats(test)
            print(f"State {state.id}: Acc {accuracy}, kt: {k_temporal_acc}, ks: {k_s}")
            
            if len([x[3] for x in state.recent_window][-len(list(predictions)):]) == 0:
                recent_temp = 0 
            else:
                r_a, r_kt, r_km, r_ks = get_stats(list(zip([x[3] for x in state.recent_window][-len(list(predictions)):], [x[4] for x in state.recent_window][-len(list(predictions)):])))
                recent_temp = r_kt
            
            if is_active_state:
                if len([x[3] for x in state.recent_window][:-len(list(predictions))]) == 0:
                    recent_temp = 0 
                else:
                    r_a, r_kt, r_km, r_ks = get_stats(list(zip([x[3] for x in state.recent_window][:-len(list(predictions))], [x[4] for x in state.recent_window][:-len(list(predictions))])))
                    recent_temp = r_kt
        
            print(f"Recent: {recent_temp * t}")
            shadow_ks = get_kappa_agreement([x[0] for x in shadow_results], predictions)
            print(f"Shadow ks: {shadow_ks}")
            active_state_penalty = 1.05 if is_active_state else 1
            if k_temporal_acc * active_state_penalty > recent_temp * t  or shadow_ks * active_state_penalty > shadow_t:
                print("added")
                close_concepts.append((state.id, k_temporal_acc, shadow_ks, accuracy))

        close_concepts = self.close_state_merge_check(close_concepts, learner, system_stats, recent_X, recent_y, stat='kt')
           
        close_concepts.sort(key = lambda x: x[1], reverse = True)

        use_shadow = True
        print("shadow")
        print(shad_acc, shad_k_t, shad_k_m, shad_k_s)
        if len(close_concepts) > 0:
            
            print(close_concepts[0][1])
            current_accuracy_overall = system_stats.model_stats.right / (system_stats.model_stats.right + system_stats.model_stats.wrong)
            if close_concepts[0][1] > shad_k_t * shadow_m:
                use_shadow = False
                print("Top state better score")
            
            potential = 1 - current_accuracy_overall
            if close_concepts[0][1] > max(current_accuracy_overall + potential * 0.5, 0.8):
                use_shadow = False
                print("top state over average acc")
            
            # if close_concepts[0][2] > 0.8:
            #     use_shadow = False
            #     print("top state close to shadow")
        print(use_shadow)
        if use_shadow:
            new_state_id = self.make_state(learner)
            self.states[new_state_id].main_model = deepcopy(shadow_state.main_model)
            is_recurring = False
            if not self.suppress: print("using shadow")
        else:
            new_state_id = close_concepts[0][0]
            if new_state_id != self.active_state_id:
                # self.states[new_state_id].main_model.partial_fit(list(reversed(recent_X)), list(reversed(recent_y)))
                self.states[new_state_id].main_model.partial_fit(recent_X[-100::-1], recent_y[-100::-1])
            is_recurring = True
            if not self.suppress: print(f'using {new_state_id}')
        return (new_state_id, is_recurring, deepcopy(shadow_state.main_model))
    
    def check_for_merge(self, state_A, state_B):
        window = [e.X for e in state_A.main_model_stats.first_seen_examples + state_B.main_model_stats.first_seen_examples]

        n_samples = len(window)
        if n_samples < 50:
            return True, 1
        n_features = len(window[0])
        window_np = np.array(window).reshape(n_samples, n_features)

        state_A_predictions = state_A.main_model.predict(window_np)
        state_B_predictions = state_B.main_model.predict(window_np)

        kappa = get_kappa_agreement(state_A_predictions, state_B_predictions)
        print(kappa)
        return kappa > self.merge_similarity, kappa


    def merge_states(self, state_A, state_B, learner, feature_description, k, merge_similarity):
        new_state_id = self.make_state(learner)

        new_learner = self.construct_state(-1, learner)
        if merge_similarity > 0.95:
            new_learner, fitted = self.acc_merge(state_A, state_B, new_learner, feature_description, k)
        else:
            new_learner, fitted = self.surrogate_merge(state_A, state_B, new_learner, feature_description, k)

        
        if fitted:
            del self.states[state_A.id]
            del self.states[state_B.id]
            if self.active_state_id == state_A.id:
                self.active_state_id = new_state_id
            if self.active_state_id == state_B.id:
                self.active_state_id = new_state_id
            self.merge_log[state_A.id] = new_state_id
            self.merge_log[state_B.id] = new_state_id
            
            #Merge transitions
            for t_id in self.transitions:
                t = self.transitions[t_id]
                if t.to_ids[state_A.id] > 0:
                    t.to_ids[new_state_id] += t.to_ids[state_A.id]
                    t.to_ids[state_A.id] = 0
                if t.from_ids[state_A.id] > 0:
                    t.from_ids[new_state_id] += t.from_ids[state_A.id]
                    t.from_ids[state_A.id] = 0
                if t.to_ids[state_B.id] > 0:
                    t.to_ids[new_state_id] += t.to_ids[state_B.id]
                    t.to_ids[state_B.id] = 0
                if t.from_ids[state_B.id] > 0:
                    t.from_ids[new_state_id] += t.from_ids[state_B.id]
                    t.from_ids[state_B.id] = 0
            
            #Merge models
            
            new_learner.id = new_state_id

            self.states[new_state_id] = new_learner
            return new_learner
        else:
            return None

    def surrogate_merge(self, state_A, state_B, new_state, feature_description, k):
        categorical_proportions = {}
        if not self.suppress: print(" ")
        for x in feature_description.keys():
            desc = feature_description[x]
            if desc['type'] != 'numeric':
                categorical_proportions[x] = {'values': [], 'p': []}
                for val in desc['proportion'].keys():
                    categorical_proportions[x]['values'].append(val)
                    categorical_proportions[x]['p'].append(desc['proportion'][val] / k)
        
        fitted = False
        X = []
        for x in feature_description.keys():
            desc = feature_description[x]
            if desc['type'] == 'numeric':
                X.append(np.random.normal(loc = desc['m'], scale = math.sqrt(desc['var']), size = 100000))
            else:
                values = np.random.choice(categorical_proportions[x]['values'], p = categorical_proportions[x]['p'], size = 100000)
                X.append(values)
        
        np_x = np.array(X)
        if not self.suppress: print(np_x.shape)
        np_x = np_x.transpose()
        if not self.suppress: print(np_x.shape)
        
        try:
            y = np.zeros(100000)
            y[::2] = state_A.main_model.predict(np_x[::2])
            if not self.suppress: print(y[:5])
            y[1::2] = state_B.main_model.predict(np_x[1::2])
            if not self.suppress: print(y[:5])
            
            num_evolutions = 0
            for row_i, row_x in np_x:
                row_y = y[row_i]
                prediction = new_state.main_model.predict(np.asarray([row_x]))[0]
                correctly_classifies = prediction == y
                new_state.main_model.partial_fit(np_x, y)
                new_state.add_main_prediction(row_x, row_y, prediction, correctly_classifies, row_i)
                if( hasattr(new_state.main_model, 'splits_since_reset') and new_state.main_model.splits_since_reset > num_evolutions):
                    new_state.set_evolution_stats()
                    new_state.evolution_stats_generating = True
                    new_state.evolution_samples_seen = 0
                    new_state.evolution_stats = {'acc': 0, 'kt': 0, 'km': 0, 'ka': 0}
                    new_state.evolution.append([ex, len(new_state.evolution), None, None, None])
            fitted = True
        except:
            if not self.suppress: print("fit failed")
        
        # for i in range(50000):
        #     print(f"surrogate {i}\r", end = "")
        #     train_state = state_A if i % 2 == 0 else state_B
        #     X = []
        #     for x in feature_description.keys():
        #         desc = feature_description[x]
        #         if desc['type'] == 'numeric':
        #             X.append(np.random.normal(loc = desc['m'], scale = math.sqrt(desc['var'])))
        #         else:
        #             values = np.random.choice(categorical_proportions[x]['values'], p = categorical_proportions[x]['p'])
        #     np_X = np.array(X).reshape(1, len(X))
        #     y = train_state.main_model.predict(np_X)
        #     new_state.main_model.partial_fit(np_X, y)
        if not self.suppress: print("Done")
        return new_state, fitted


    def acc_merge(self, state_A, state_B, new_state, feature_description, k):
        window = [e.X for e in state_A.main_model_stats.first_seen_examples + state_B.main_model_stats.first_seen_examples]
        window_y = [e.y for e in state_A.main_model_stats.first_seen_examples + state_B.main_model_stats.first_seen_examples]

        n_samples = len(window)
        if n_samples < 50:
            return state_A, True

        n_features = len(window[0])
        window_np = np.array(window).reshape(n_samples, n_features)

        state_A_predictions = state_A.main_model.predict(window_np)
        state_B_predictions = state_B.main_model.predict(window_np)

        A_acc = sum(state_A_predictions == window_y) / len(window_y)
        B_acc = sum(state_B_predictions == window_y) / len(window_y)

        if A_acc > B_acc:
            return state_A, True
        return state_B, True


    def get_min_rA_state(self):
        min_reuse = None
        min_id = None
        for s in self.states.values():
            if s.id == self.active_state_id:
                continue
            reuse = s.calculate_total_advantage(self.recent_states)
            if min_reuse is None or reuse < min_reuse:
                min_id = s.id
                min_reuse = reuse
        return min_id
        
    def get_min_score_state(self):
        min_reuse = None
        min_id = None
        for s in self.states.values():
            if s.id == self.active_state_id:
                continue
            s.calculate_total_advantage(self.recent_states)
            score = s.calculate_past_score_advantage()
            reuse = s.estimated_reuse_proportion * score
            if min_reuse is None or reuse < min_reuse:
                min_id = s.id
                min_reuse = reuse
        return min_id

    def get_min_AAC_state(self):
        min_reuse = None
        min_id = None
        for s in self.states.values():
            if s.id == self.active_state_id:
                continue
            s.calculate_total_advantage(self.recent_states)
            if s.evolution[-1][4] is None:
                # print("checking")
                s.set_evolution_stats()
            reuse = s.estimated_reuse_proportion * s.evolution[-1][4]
            if min_reuse is None or reuse < min_reuse:
                min_id = s.id
                min_reuse = reuse
        return min_id

    def get_max_age_state(self):
        max_age = None
        max_id = None
        for s in self.states.values():
            if s.id == self.active_state_id:
                continue
            if not self.suppress: print(f"State {s.id} age: {s.age}")
            
            if max_age is None or s.age > max_age:
                max_id = s.id
                max_age = s.age
        return max_id

    def get_max_LRU_state(self):
        max_LRU = None
        max_id = None
        for s in self.states.values():
            if s.id == self.active_state_id:
                continue
            if not self.suppress: print(f"State {s.id} LRU: {s.LRU}")
            
            if max_LRU is None or s.LRU > max_LRU:
                max_id = s.id
                max_LRU = s.LRU
        return max_id



    def get_min_div_state(self, recent_window):
        """ Check diversity if state S is removed. Return state with highest diversity if removed."""
        max_E = None
        max_id = None

        n_samples = len(recent_window)
        n_features = len(recent_window[0].X)
        recent_X = np.array([e.X for e in recent_window]).reshape(n_samples, n_features)
        recent_y = np.array([e.y for e in recent_window])
        
        def get_diversity_entropy(preds, y):
            L = len(preds)
            N = len(preds[0])
            E_sum = 0
            for j in range(N):
                l = 0
                for state_predictions in preds:
                    if state_predictions[j] == y[j]:
                        l += 1
                E_sum += (1 / (L - math.ceil(L / 2))) * min(l, L - l)
            E = E_sum / N

            return E

        if len(self.states.values()) <= 2:
            return list(self.states.values())[0].id

        state_predictions = []
        for s_i, s in enumerate(self.states.values()):
            state_predictions.append(s.main_model.predict(recent_X))

        for s_i, s in enumerate(self.states.values()):
            if s.id == self.active_state_id:
                continue
            
            preds = []
            for c_i, c in enumerate(self.states.values()):
                if s_i == c_i:
                    continue
                preds.append(state_predictions[c_i])

            # The entropy if state s is removed.
            E = get_diversity_entropy(preds, recent_y)
            if not self.suppress: print(f"State {s.id} acc: {E}")
            if max_E is None or E > max_E:
                max_id = s.id
                max_E = E
        
        return max_id

    

    def get_min_acc_state(self, recent_window):
        min_acc = None
        min_id = None

        n_samples = len(recent_window)
        n_features = len(recent_window[0].X)
        recent_X = np.array([e.X for e in recent_window]).reshape(n_samples, n_features)
        recent_y = np.array([e.y for e in recent_window])
        
        for s in self.states.values():
            if s.id == self.active_state_id:
                continue
            _, acc = get_recent_window_accuracy(s, recent_X, recent_y, False)
            if not self.suppress: print(f"State {s.id} acc: {acc}")
            if min_acc is None or acc < min_acc:
                min_id = s.id
                min_acc = acc
        return min_id

    def cull_states(self, recent_window):
        num_states = len(self.states.values())
        
        sup_val =self.suppress
        self.suppress = False
        if len(self.states.values()) > self.concept_limit and self.concept_limit > 0:
            while len(self.states.values()) > self.concept_limit:
                cull_state = None
                if self.memory_strategy == 'rA':
                    cull_state = self.get_min_rA_state()
                if self.memory_strategy == 'rAAuc':
                    cull_state = self.get_min_rAAuc_state()
                if self.memory_strategy == 'auc':
                    cull_state = self.get_min_AAC_state()
                if self.memory_strategy == 'age':
                    cull_state = self.get_max_age_state()
                if self.memory_strategy == 'LRU':
                    cull_state = self.get_max_LRU_state()
                if self.memory_strategy == 'acc':
                    cull_state = self.get_min_acc_state(recent_window)
                if self.memory_strategy == 'score':
                    cull_state = self.get_min_score_state()
                if self.memory_strategy == 'div':
                    cull_state = self.get_min_div_state(recent_window)
                
                if not self.suppress: print(f"Culling {cull_state}")
                del self.states[cull_state]
                self.deleted_states.append(cull_state)
        post_num_states = len(self.states.values())
        self.suppress = sup_val


def get_kappa_agreement(A_preds, B_preds):
    A_counts = Counter()
    B_counts = Counter()
    similar = 0
    for pA, pB in zip(A_preds, B_preds):
        A_counts[pA] += 1
        B_counts[pB] += 1
        if pA == pB:
            similar += 1
    observed_acc = similar / len(A_preds)
    expected_sum = 0
    for cat in np.unique(pA + pB):
        expected_sum += min((A_counts[cat] * B_counts[cat]) / len(A_preds), 0.99999)
    expected_acc = expected_sum / len(A_preds)
    k_s = (observed_acc - expected_acc) / (1 - expected_acc)
    return k_s




def get_recent_window_accuracy(state, recent_X, recent_Y, fit = False):
    right = 0
    predictions = []
    test_model = state.main_model
    predictions = test_model.predict(recent_X)
    right = np.sum(predictions == recent_Y)
    
    return (state.id, right / len(predictions))

def get_stats(results):
    predictions = [x[0] for x in results]
    recent_y = [x[1] for x in results]
    # results = zip(predictions, recent_y)
    accuracy = sum(np.array(predictions) == np.array(recent_y)) / len(predictions)
    k_temporal_acc = 0
    k_majority_acc = 0
    gt_counts = Counter()
    our_counts = Counter()
    majority_guess = results[0][1]
    temporal_guess = results[0][1]
    for o in results:
        p = o[0]
        gt = o[1]
        if gt == temporal_guess:
            k_temporal_acc += 1
        if gt == majority_guess:
            k_majority_acc += 1
        gt_counts[gt] += 1
        our_counts[p] += 1

        majority_guess = gt if gt_counts[gt] > gt_counts[majority_guess] else majority_guess
        temporal_guess = gt
    k_temporal_acc = min(k_temporal_acc / len(results), 0.99999)
    k_temporal_acc = (accuracy - k_temporal_acc) / (1 - k_temporal_acc)
    k_majority_acc = min(k_majority_acc / len(results), 0.99999)

    k_majority_acc = (accuracy - k_majority_acc) / (1 - k_majority_acc)
    expected_accuracy = 0
    for cat in np.unique(predictions):
        expected_accuracy += min((gt_counts[cat] * our_counts[cat]) / len(results), 0.99999)
    expected_accuracy /= len(results)
    k_s = (accuracy - expected_accuracy) / (1 - expected_accuracy)

    return accuracy, k_temporal_acc, k_majority_acc, k_s
def get_knn_similarity(state, recent_window):
    state_window = state.main_model_stats.first_seen_examples
    pval = knnTest(state_window, recent_window).pvalue
    return pval

def get_QT_similarity(state, recent_window):
    state_window = state.main_model_stats.first_seen_examples
    different, val = quantTreeTest(state_window, recent_window)
    return different

def get_FSM_transition_prob(current_state, state, fsm):
    transition = fsm.transitions[current_state.id]
    total_support = transition.out_edges
    specific_support = transition.to_ids[state.id]
    probability = 0 if specific_support < 1 or total_support < 1 else specific_support / total_support
    return (probability, specific_support, total_support)
