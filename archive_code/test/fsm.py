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
        self.evolution = []                     # Tracks the evolution of the model.
        self.reuse_log = {
            'Q': 1e-2,
            'R': 1e-6,
            'transitions_to_model': 0,
            'transitions_not_to_model': 0,
            'future_proportion_estimate': 1,
            'error_estimate': 1,
            'fading_average': 0,
        }
        self.accuracy_log = {
            'Q': 1e-5,
            'R': 0,
            'correct_last_use': 0,
            'wrong_last_use': 0,
            'future_accuracy_estimate': 1,
            'error_estimate': 1,
            't_correct_last_use': 0,
            't_wrong_last_use': 0,
            't_last_class': 0,
        }
        self.prop_seen_log = {'total': 0}
        self.age = 0
        self.LRU = 1
        self.estimated_reuse_proportion = 1
        self.estimated_reuse_advantage = 1
        self.estimated_total_advantage = 1
        self.past_accuracy = [(0, 0)]
        self.start_ex = 0
        self.end_ex = 0
        self.cumulative_ex = 0
        self.estimated_past_advantage = 0
        self.extrapolated_advantage = 0
        self.main_counter = 0

        self.score_right = 0
        self.score_wrong = 0
        self.score_invalid_temporal = 0
        self.score_invalid_majority = 0
        self.recent_score = deque()
        self.recent_rec = deque()
        self.recent_queue = deque()
        self.recent_rec_score = 0
        self.last_props_right = deque()
        self.recent_kts = deque()
        self.last_kt = 0
        self.temporal_right = 0
        self.temporal_wrong = 0
        self.acc_right = 0
        self.acc_wrong = 0

        self.shadow_state = None
        self.shadow_model_stats = modelStats(id, 'shadow')

        self.restore_state = None
        self.signal_backtrack = False
        self.shadow_transition_point = None
        self.shadow_transition_from = None
        self.backtrack_better_count = 0
        


        
        

    def __repr__(self):
        return f"<{self.id}: eR: {self.estimated_reuse_proportion}, eA: {self.estimated_reuse_advantage}, rA: {self.estimated_total_advantage}"
    
    def add_main_prediction(self, X, y, p, is_correct, ts):
        """
        Call on making a new prediction. Adds accuracy to state logs.
        """
        self.main_model_stats.add_prediction(X, y, p, is_correct, ts)

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

        self.recent_queue.append((p, y))
        if len(self.recent_queue) > 1500:
            self.recent_queue.popleft()
        # Log accuracy
        if is_correct:
            self.accuracy_log['correct_last_use'] += 1
        else:
            self.accuracy_log['wrong_last_use'] += 1
        
        # Log accuracy of temporal base case
        t_prediction = self.accuracy_log['t_last_class']
        t_correct = t_prediction == y
        self.accuracy_log['t_last_class'] = y
        if t_correct:
            self.accuracy_log['t_correct_last_use'] += 1
        else:
            self.accuracy_log['t_wrong_last_use'] += 1

        # Log accuracy of Majority base case
        if y not in self.prop_seen_log:
            self.prop_seen_log[y] = 0
        self.prop_seen_log[y] += 1
        self.prop_seen_log['total'] += 1

        max_y = None
        max_y_prop = None
        for y_val in self.prop_seen_log:
            if y_val == 'total':
                continue
            y_prop = self.prop_seen_log[y_val] / self.prop_seen_log['total']
            if max_y_prop == None or max_y_prop < y_prop:
                max_y_prop = y_prop
                max_y = y_val

        # Log accuracy above temporal and majority base cases
        if not is_correct:
            self.score_wrong += 1
            self.recent_score.append(0)
            self.temporal_wrong += 1
            self.acc_wrong += 1
            self.recent_rec.append(0)
        elif t_correct:
            self.score_invalid_temporal += 1
            self.recent_score.append(0)
            # self.temporal_wrong += 1
            self.acc_right += 1
            self.recent_rec.append(1)
        elif y == max_y:
            self.score_invalid_majority += 1
            self.recent_score.append(0)
            self.temporal_right += 1
            self.acc_right += 1
            self.recent_rec.append(1)
        else:
            self.score_right += 1
            self.temporal_right += 1
            self.recent_score.append(1)
            self.acc_right += 1
            self.recent_rec.append(1)

        if(len(self.recent_score) > 2000):
            self.recent_score.popleft()
        
        if(len(self.recent_rec) > 500):
            self.recent_rec.popleft()
        
        self.recent_rec_score = 0 if len(self.recent_rec) == 0 else sum(self.recent_rec) / len(self.recent_rec)

        # Add an accuracy measurement every so often
        if self.main_counter > 500:
            self.add_accuracy_measurement(ex = ts)
            self.main_counter = 0
        self.main_counter += 1

    def calculate_past_score_advantage(self):
        """
        Calculate a measurement of how good reusing this state is
        compared to retraining an alternative from scratch.
        Uses the max of recent accuracy measurement scores to calculate 
        an estimated number of correct predictions if reused.
        Compares this to the number of correct predictions made while training.
        The difference gives a 'score' for reuse over retrain.
        """
        if len(self.recent_score) > 0:
            recent_prop_right = sum(self.recent_score) / len(self.recent_score)
        else:
            print("Bad, 0 rrecent score")
            recent_prop_right = 1

        self.last_props_right.append(recent_prop_right)
        prop_right = max(self.last_props_right)
        if len(self.last_props_right) > 10:
            self.last_props_right.popleft()
        past_total = self.score_right + self.score_wrong + self.score_invalid_majority + self.score_invalid_temporal

        reuse_right = prop_right * past_total

        retrain_right = self.score_right

        if past_total > 0:
            retrain_prop_right = self.score_right / past_total
        else:
            print("Bad, 0 reuse past total")
            retrain_prop_right = 1

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

    def add_accuracy_measurement(self, ex = 0):
        """Update our estimate of future proportion of the stream taken up by this state using a Kalman filter.
        Also update temporal and majority comparisons.
        Also build the 'auc' measure, of reuse vs retrain accuracy.
        """

        # Get accuracy since last called
        if (self.accuracy_log['correct_last_use'] + self.accuracy_log['wrong_last_use']) > 0:
            last_usage_accuracy = self.accuracy_log['correct_last_use'] / (self.accuracy_log['correct_last_use'] + self.accuracy_log['wrong_last_use'])
        else:
            print("Bad, 0 total acc")
            last_usage_accuracy = 1
        
        # Get temporal base comparison accuracy since last called
        if (self.accuracy_log['t_correct_last_use'] + self.accuracy_log['t_wrong_last_use']) > 0:
            temporal_accuracy = self.accuracy_log['t_correct_last_use'] / (self.accuracy_log['t_correct_last_use'] + self.accuracy_log['t_wrong_last_use'])
        else:
            print("Bad, 0 tacc")
            temporal_accuracy = 1
        
        # Get majority base comparison accuracy since last called
        if self.prop_seen_log['total'] > 0:
            majority_accuracy = max([self.prop_seen_log[k] for k in self.prop_seen_log.keys() if k != 'total']) / self.prop_seen_log['total']
        else:
            print("Bad, 0 total prop")
            majority_accuracy = 1

        # Calculate temporal kappa
        if temporal_accuracy > 0 and temporal_accuracy < 1:
            if temporal_accuracy < 1:
                k_t = (last_usage_accuracy - temporal_accuracy) / (1 - temporal_accuracy)
            else:
                print("Bad")
                k_t = 1
        else:
            print("Bad, 0 t-acc")
            k_t = 1
        
        kt_avg = 0
        if len(self.recent_kts) > 2:
            kt_avg = sum(self.recent_kts) / len(self.recent_kts)
        if k_t > kt_avg and k_t > 0.4:
            self.recent_kts.append(k_t)
        if len(self.recent_kts) > 10:
            self.recent_kts.popleft()
        # calculate majority kappa
        if majority_accuracy > 0 and majority_accuracy < 1:
            if majority_accuracy < 1:
                k_m = (last_usage_accuracy - majority_accuracy) / (1 - majority_accuracy)
            else:
                k_m = 1
        else:
            print("Bad, 0 m-acc")
            k_m = 1

        # Average recent temporal and majority kappa to get a combined accuracy score
        score = (k_t + k_m) / 2

        # Use this score to estimate 'auc' between reusing and retraining a model.
        # Basically if score is higher than max, it adds the area of the rectangle + triangle,
        # If lower by a small amount no change,
        # If lower by a large amount diminishes the whole 'auc'
        last_ex = self.past_accuracy[-1][0]
        last_accurcy = self.past_accuracy[-1][1]
        run = ex - self.start_ex
        self.start_ex = ex
        self.cumulative_ex += run

        accuracy_delta = score - last_accurcy
        score = max(score, last_accurcy)
        if accuracy_delta > 0:
            ex_delta = self.cumulative_ex - last_ex
            acc_inc = (accuracy_delta * last_ex) + ((ex_delta * abs(accuracy_delta)) / 2)
            future_potential = (1 - score) + 1
            acc_inc *= future_potential
            
            self.estimated_past_advantage += acc_inc
        else:
            if score > 0:
                if (abs(accuracy_delta) / score) > 0.05:
                    self.estimated_past_advantage = self.estimated_past_advantage * 0.99
        
        # Estimate the accuracy in the future using a kalman filter on previous measures.
        self.past_accuracy.append((self.cumulative_ex, score, self.estimated_past_advantage))
        last_p = self.accuracy_log['error_estimate'] + self.accuracy_log['Q']
        k = last_p / (last_p + self.accuracy_log['R'])
        estimate = self.accuracy_log['future_accuracy_estimate'] + k * (last_usage_accuracy - self.accuracy_log['future_accuracy_estimate'])

        self.accuracy_log['future_accuracy_estimate'] = estimate
        self.accuracy_log['error_estimate'] = (1 - k) * last_p
        self.accuracy_log['correct_last_use'] = 0
        self.accuracy_log['wrong_last_use'] = 0
        # self.accuracy_log['t_correct_last_use'] = 0
        # self.accuracy_log['t_wrong_last_use'] = 0
        self.last_kt = k_t

    def estimate_future_accuracy(self, test_acc):
        """
        Estimate the accuracy in the future using a kalman filter on previous measures.
        """
        last_p = self.accuracy_log['error_estimate'] + self.accuracy_log['Q']
        k = last_p / (last_p + self.accuracy_log['R'])
        estimate = self.accuracy_log['future_accuracy_estimate'] + k * (test_acc - self.accuracy_log['future_accuracy_estimate'])
        return estimate
    
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

class FSM:
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
        
    def transition_to(self, to_id, ex = 0, shadow = None):
        from_id = self.active_state_id
        self.add_transition(self.active_state_id, to_id)
        self.get_state().add_accuracy_measurement(ex)
        self.recent_states.append(to_id)
        if len(self.recent_states) > 100:
            self.recent_states.popleft()

        if not self.suppress: print(f"Transition from {self.active_state_id} to {to_id}. Accuracy for {self.active_state_id} was {self.states[self.active_state_id].past_accuracy[-5:]}")
        
        try:
            last_acc = self.states[self.active_state_id].acc_right / (self.states[self.active_state_id].acc_right + self.states[self.active_state_id].acc_wrong)
        except:
            last_acc = 0

        print(f"Transition from {self.active_state_id} to {to_id}. Accuracy for {self.active_state_id} was {last_acc}")
        self.active_state_id = to_id
        self.get_state().start_ex = ex
        for s_id in self.states.keys():
            s = self.states[s_id]
            s.age += 1
            if s.id == to_id:
                s.reuse_log['transitions_to_model'] += 1
                s.add_proportion_measurement()
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
        
        last_auc = 0
        last_evo = -1
        last_ex = 0
        if len(self.get_state().evolution) > 0:
            last_auc = self.get_state().evolution[-1][2]
            last_evo = self.get_state().evolution[-1][1]
            last_ex = self.get_state().evolution[-1][0]

        evo = len(self.get_state().evolution)
        auc = last_auc
        inc = 0

        inc += (last_ex) * (evo - last_evo)
        inc += ((ex - last_ex) * (evo - last_evo)) / 2

        cur_evo = min(evo, 10)
        last_evo =  min(last_evo, 10)
        gradient = (cur_evo - last_evo) / (ex - last_ex)

        if gradient == 0:

            future_potential = auc * 0.01
        else:
            future_potential = 0
            future_potential += (ex) * (10 - cur_evo)
            intersection_point = (10 - cur_evo) / gradient
            future_potential += ((intersection_point - ex) * (10 - cur_evo)) / 2

        exploration = 0.5


        auc += (inc + exploration * future_potential)

        auc /= 10000
        self.get_state().evolution.append((ex, evo, auc))
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
                        # state_accuracy = state.estimate_future_accuracy(test_accuracy)
                        ss = (state_id, return_stat, 0.79, accuracy)
                        # ss = (state_id, accuracy)
                        ss_i = next_index + 1
            if not merged:
                ss_i = next_index
                    
            
            merged_scores.append(ss)
        print(f"original: {scores}, merged: {merged_scores}")
        if scores != merged_scores:
            print("DEBUG")
            # exit()
        return merged_scores


    def get_recurring_state_or_new_prev_acc_comparison(self, recent_window, learner, system_stats):
        t = 0.85
        shadow_t = 0.95
        shadow_m = 1.05

        print("*****")
        print(f"Length of window: {len(recent_window)}")
        close_concepts = []
        n_samples = len(recent_window)
        if not self.suppress: print(n_samples)
        n_features = len(recent_window[0].X)
        recent_X = np.array([e.X for e in recent_window]).reshape(n_samples, n_features)
        recent_y = np.array([e.y for e in recent_window])
        shadow_state = self.construct_state(-1, learner)

        shadow_results = []
        
        pretrain_c = 0
        pretrain_l = 50
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

        
        for s_i, state in enumerate(self.states.values()):
            predictions = state.main_model.predict(recent_X)
            test = list(zip(list(predictions), list(recent_y)))
            accuracy, k_temporal_acc, k_majority_acc, k_s = get_stats(test)
            print(f"State {state.id}: Acc {accuracy}, kt: {k_temporal_acc}")
            recent_temp = (state.temporal_right / max((state.temporal_right + state.temporal_wrong), 1))
            # recent_temp = (state.acc_right / max((state.acc_right + state.acc_wrong), 1))
            print(f"Recent: {recent_temp * t}")
            shadow_ks = get_kappa_agreement([x[0] for x in shadow_results], predictions)
            print(f"Shadow ks: {shadow_ks}")
            active_state_penalty = 1.05 if state.id == self.active_state_id else 1
            if k_temporal_acc * active_state_penalty > recent_temp * t  or shadow_ks * active_state_penalty > shadow_t:
            # if accuracy > recent_temp * t or shadow_ks > shadow_t:
                print("added")
                close_concepts.append((state.id, k_temporal_acc, shadow_ks, accuracy))
                # close_concepts.append((state.id, accuracy, shadow_ks))

        close_concepts = self.close_state_merge_check(close_concepts, learner, system_stats, recent_X, recent_y, 'kt')
        
        # close_concepts.append((-1, test['shadow_results']['acc'] / shadow_m))        
        close_concepts.sort(key = lambda x: x[1], reverse = True)

        use_shadow = True
        print("shadow")
        print(shad_acc, shad_k_t, shad_k_m, shad_k_s)
        if len(close_concepts) > 0:
            
            print(close_concepts[0][1])
            current_accuracy_overall = system_stats.model_stats.right / (system_stats.model_stats.right + system_stats.model_stats.wrong)
            if close_concepts[0][1] > shad_k_t * shadow_m:
            # if close_concepts[0][1] > shad_acc * shadow_m or close_concepts[0][1] > min(current_accuracy_overall * 1.1, 0.9) or close_concepts[0][2] > 0.8:
                use_shadow = False
                print("Top state better score")
            
            potential = 1 - current_accuracy_overall
            if close_concepts[0][3] > max(current_accuracy_overall + potential * 0.5, 0.8):
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
                self.states[new_state_id].main_model.partial_fit(recent_X[-100::-1], recent_y[-100::-1])
            is_recurring = True
            if not self.suppress: print(f'using {new_state_id}')
        return (new_state_id, is_recurring, shadow_state.main_model)


    def get_AD_reccurence(self, recent_window, learner, system_stats):
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
                test = list(zip([x[0] for x in state.recent_queue][-len(list(predictions)):], list(recent_y)))

            accuracy, k_temporal_acc, k_majority_acc, k_s = get_stats(test)
            print(f"State {state.id}: Acc {accuracy}, kt: {k_temporal_acc}, ks: {k_s}")
            # recent_temp = (state.acc_right / max((state.acc_right + state.acc_wrong), 1))
            recent_temp = state.recent_rec_score
            if is_active_state:
                recent_temp = 0 if len([x[0] for x in state.recent_queue][:-len(list(predictions))]) == 0 else sum([x[0] == x[1] for x in state.recent_queue][:-len(list(predictions))]) / len([x[0] for x in state.recent_queue][:-len(list(predictions))])
            print(f"Recent: {recent_temp * t}")
            shadow_ks = get_kappa_agreement([x[0] for x in shadow_results], predictions)
            print(f"Shadow ks: {shadow_ks}")
            active_state_penalty = 1.05 if is_active_state else 1
            if accuracy * active_state_penalty > recent_temp * t  or shadow_ks * active_state_penalty > shadow_t:
                print("added")
                close_concepts.append((state.id, accuracy, shadow_ks, accuracy))

        close_concepts = self.close_state_merge_check(close_concepts, learner, system_stats, recent_X, recent_y, stat='accuracy')
           
        close_concepts.sort(key = lambda x: x[1], reverse = True)

        use_shadow = True
        print("shadow")
        print(shad_acc, shad_k_t, shad_k_m, shad_k_s)
        if len(close_concepts) > 0:
            
            print(close_concepts[0][1])
            current_accuracy_overall = system_stats.model_stats.right / (system_stats.model_stats.right + system_stats.model_stats.wrong)
            if close_concepts[0][1] > shad_acc * shadow_m:
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
                self.states[new_state_id].main_model.partial_fit(recent_X[-100::-1], recent_y[-100::-1])
            is_recurring = True
            if not self.suppress: print(f'using {new_state_id}')
        return (new_state_id, is_recurring, deepcopy(shadow_state.main_model))

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
                test = list(zip([x[0] for x in state.recent_queue][-len(list(predictions)):], list(recent_y)))

            accuracy, k_temporal_acc, k_majority_acc, k_s = get_stats(test)
            print(f"State {state.id}: Acc {accuracy}, kt: {k_temporal_acc}, ks: {k_s}")
            
            if len([x[0] for x in state.recent_queue][-len(list(predictions)):]) == 0:
                recent_temp = 0 
            else:
                r_a, r_kt, r_km, r_ks = get_stats(list(zip([x[0] for x in state.recent_queue][-len(list(predictions)):], [x[1] for x in state.recent_queue][-len(list(predictions)):])))
                recent_temp = r_kt
            
            if is_active_state:
                if len([x[0] for x in state.recent_queue][:-len(list(predictions))]) == 0:
                    recent_temp = 0 
                else:
                    r_a, r_kt, r_km, r_ks = get_stats(list(zip([x[0] for x in state.recent_queue][:-len(list(predictions))], [x[1] for x in state.recent_queue][:-len(list(predictions))])))
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

    def get_recurring_state_or_new(self, recent_window, learner, system_stats):
        """Checks the FSM for a possible recurring state.

        Passed in data from a detected concept drift, attempts to find a matching
        concept in the FSM repository. If found, returns the <state_id> of the matching state.
        If not found, returns the <state_id> of a newly trained state.

        Parameters
        ----------
        recent_window: list<BufferItem>
            A list of recent examples, assumed to be from the new concept.

        learner: StreamModel generator
            A function to create an adaptive learner, for use in training a baseline new model.
        Returns
        -------
        (state_id: int, is_recurring: bool)
            The state_id of the next state. False if a new state, True if recurring.
            Note state has NOT switched yet. This must be explicit.

        """
        state_similarity = []
        n_samples = len(recent_window)
        if not self.suppress: print(n_samples)
        n_features = len(recent_window[0].X)
        recent_X = np.array([e.X for e in recent_window]).reshape(n_samples, n_features)
        recent_y = np.array([e.y for e in recent_window])
        shadow_state = self.construct_state(-1, learner)
        _, shadow_state_accuracy = get_recent_window_accuracy(shadow_state, recent_X, recent_y, True)
        #state_similarity.append((shadow_state.id, shadow_state_accuracy))
        if not self.suppress: print(f"Performance on last 50 with shadow state: Acc {shadow_state_accuracy}")
        if len(self.states.keys()) > 15 and False:
            p = Pool()
            state_similarity = p.starmap(get_recent_window_accuracy, [(s, recent_X, recent_y) for s in self.states.values()])
        else:
            for s_i, state in enumerate(self.states.values()):
                active_state = self.is_current_state(state.id)
                t_i, test_accuracy = get_recent_window_accuracy(state, recent_X, recent_y)
                #probability, specific_support, total_support = get_FSM_transition_prob(self.get_state(), state, self)
                #knn_p = get_knn_similarity(state, recent_window)
                #qt_diff = get_QT_similarity(state, recent_window)
                state_similarity.append((state.id, state.estimate_future_accuracy(test_accuracy)))
                # if not self.suppress:
                if not self.suppress: print(f"State {state.id} accuracy on last 50: {test_accuracy}")
                if not self.suppress: print(f"State {state.id} accuracy estimate: {state.estimate_future_accuracy(test_accuracy)}")
                    #print(f"State {state.id} FSM probability: {probability} with support {specific_support} / {total_support}")
                    #print(f"State {state.id} KNN p-value: {knn_p}")
                    #print(f"State {state.id} QT different: {qt_diff}")
                # unique, counts = np.unique(test_predictions, return_counts=True)
                # print(list(zip(unique, counts)))
        state_similarity.sort(key = lambda x : x[1], reverse=True)

        merged_state_similarity = []
        ss_i = 0
        #print(state_similarity)
        while ss_i < len(state_similarity):
            # print(ss_i)
            ss = state_similarity[ss_i]
            state_id = ss[0]
            # print(state_id)
            state_accuracy = ss[1]
            
            # if state_id == -1:
            #     continue
            state = self.states[state_id]
            next_index = ss_i + 1
            if next_index >= len(state_similarity):
                merged_state_similarity.append(ss)
                break
            
            # if state_similarity[next_index][0] == -1:
            #     next_index = ss_i + 2
            #     if next_index >= len(state_similarity):
            #         merged_state_similarity.append(ss)
            #         break
            # print(next_index)
            next_state_id = state_similarity[next_index][0]
            next_state = self.states[next_state_id]
            nex_state_accuracy = state_similarity[next_index][1]
            
            # print(next_state_id)
            merged = False
            if abs(state_accuracy - nex_state_accuracy) < 0.05:
                merge_state, merge_similarity = self.check_for_merge(state, next_state)
                if merge_state:
                    if not self.suppress: self.suppress: print('MERGING!!')
                    merged = True
                    state = self.merge_states(state, next_state, learner, system_stats.feature_description, system_stats.seen_features, merge_similarity)
                    if state == None:
                        merged = False
                        if not self.suppress: print("merge failed")
                    
                    if merged:
                        state_id = state.id
                        t_i, test_accuracy = get_recent_window_accuracy(state, recent_X, recent_y)
                        state_accuracy = state.estimate_future_accuracy(test_accuracy)
                        ss = (state_id, state_accuracy)
                        ss_i = next_index + 1
            if not merged:
                ss_i = next_index
                    
            
            merged_state_similarity.append(ss)
        
        state_similarity = merged_state_similarity
        #print(state_similarity)
                    

            


        new_state_id = None
        is_recurring = False
        # if len(state_similarity) < 1 or (state_similarity[0][1] < (shadow_state_accuracy) or state_similarity[0][1] < 0.8):
        current_accuracy_overall = system_stats.model_stats.right / (system_stats.model_stats.right + system_stats.model_stats.wrong)
        top_state_vs_overall = state_similarity[0][1] / current_accuracy_overall
        if not self.suppress: print(f"top state is {top_state_vs_overall} of overall")
        
        use_shadow = False
        if len(state_similarity) < 1:
            use_shadow = True
        if state_similarity[0][0] == -1:
            use_shadow = True
        if not use_shadow and top_state_vs_overall < 1:
            shadow_state_evolutions = len(shadow_state.evolution) + 1
            top_state_evolutions = len(self.states[state_similarity[0][0]].evolution) + 1
            weighted_shadow_acc = shadow_state_accuracy / shadow_state_evolutions
            weighted_top_acc = state_similarity[0][1] / top_state_evolutions
            ratio = weighted_shadow_acc / weighted_top_acc
            if not self.suppress: print(f"shadow state is {ratio} of top")
            if ratio > top_state_vs_overall:
                use_shadow = True

        if use_shadow:
            new_state_id = self.make_state(learner)
            is_recurring = False
            if not self.suppress: print("using shadow")
        else:
            new_state_id = state_similarity[0][0]

            is_recurring = True
            if not self.suppress: print(f'using {new_state_id}')
        return (new_state_id, is_recurring)
    
    def check_for_merge(self, state_A, state_B):
        window = [e.X for e in state_A.main_model_stats.first_seen_examples + state_B.main_model_stats.first_seen_examples]

        n_samples = len(window)
        if n_samples < 50:
            return True
        n_features = len(window[0])
        window_np = np.array(window).reshape(n_samples, n_features)

        state_A_predictions = state_A.main_model.predict(window_np)
        state_B_predictions = state_B.main_model.predict(window_np)

        kappa = get_kappa_agreement(state_A_predictions, state_B_predictions)
        print(kappa)
        return kappa > self.merge_similarity, kappa


    def merge_states(self, state_A, state_B, learner, feature_description, k, merge_similarity):
        #print(state_A)
        #print(state_B)
        #print(self.states)
        new_state_id = self.make_state(learner)
        #delete old


        new_learner = self.construct_state(-1, learner)
        if self.merge_strategy == 'sur':
            new_learner, fitted = self.surrogate_train(state_A, state_B, new_learner, feature_description, k)
        if self.merge_strategy == 'acc':
            new_learner, fitted = self.acc_merge(state_A, state_B, new_learner, feature_description, k)
        if self.merge_strategy == 'both':
            if merge_similarity > 0.95:
                new_learner, fitted = self.acc_merge(state_A, state_B, new_learner, feature_description, k)
            else:
                new_learner, fitted = self.surrogate_train(state_A, state_B, new_learner, feature_description, k)
        
        if fitted:
            del self.states[state_A.id]
            del self.states[state_B.id]
            new_accuracy_log = None
            if self.active_state_id == state_A.id:
                self.active_state_id = new_state_id
                new_accuracy_log = state_A.accuracy_log
            if self.active_state_id == state_B.id:
                self.active_state_id = new_state_id
                new_accuracy_log = state_B.accuracy_log
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
            
            state_A.id = new_state_id
            new_state = state_A

            new_state.main_model = new_learner.main_model

            #Merge meta-information
            merge_proportion = 0.5
            def get_merge_average(A_property, B_property):
                ret_val = merge_proportion * A_property + (1 - merge_proportion) * B_property
                if isinstance(A_property, int) and isinstance(B_property, int):
                    ret_val = int(ret_val)
                return ret_val
            num_evolutions = int(get_merge_average(len(state_A.evolution), len(state_B.evolution)))
            if num_evolutions > 1:
                print(num_evolutions)
                print(state_A.evolution)
                print(state_B.evolution)
                if len(state_A.evolution) < 1 and len(state_B.evolution) >= 2:
                    state_A.evolution.append(state_B.evolution[-2])
                if len(state_A.evolution) < 2 and len(state_B.evolution) >= 2:
                    state_A.evolution.append(state_B.evolution[-1])
                if len(state_B.evolution) < 1 and len(state_A.evolution) >= 2:
                    state_B.evolution.append(state_A.evolution[-2])
                if len(state_B.evolution) < 2 and len(state_A.evolution) >= 2:
                    state_B.evolution.append(state_A.evolution[-1])
                if len(state_A.evolution) < 2 and len(state_B.evolution) < 2:
                    state_A.evolution.append((0, 1, 0))
                    state_A.evolution.append((1000, 2, 0))
                    state_B.evolution.append((0, 1, 0))
                    state_B.evolution.append((1000, 2, 0))
                print(state_A.evolution)
                print(state_B.evolution)
                final_evolution = (get_merge_average(state_A.evolution[-1][0], state_B.evolution[-1][0]),
                                    get_merge_average(state_A.evolution[-1][1], state_B.evolution[-1][1]),
                                    get_merge_average(state_A.evolution[-1][2], state_B.evolution[-1][2]),
                                    )
                semifinal_evolution = (get_merge_average(state_A.evolution[-2][0], state_B.evolution[-2][0]),
                                    get_merge_average(state_A.evolution[-2][1], state_B.evolution[-2][1]),
                                    get_merge_average(state_A.evolution[-2][2], state_B.evolution[-2][2]),
                                    )
                merged_evolutions = []
                for e in range(num_evolutions):
                    merged_evolutions.append([e, e, e])
                merged_evolutions[-2] = semifinal_evolution
                merged_evolutions[-1] = final_evolution
                new_state.evolution = merged_evolutions
            else:
                new_state.evolution = []
                if len(state_A.evolution) > 0:
                    new_state.evolution.append(state_A.evolution[-1])
                if len(state_B.evolution) > 0:
                    new_state.evolution.append(state_B.evolution[-1])

            merged_reuselog = {}
            for k in state_A.reuse_log.keys():
                merged_reuselog[k] = get_merge_average(state_A.reuse_log[k], state_B.reuse_log[k])
            new_state.reuse_log = merged_reuselog

            merged_accuracy_log= {}
            for k in state_A.accuracy_log.keys():
                merged_accuracy_log[k] = get_merge_average(state_A.accuracy_log[k], state_B.accuracy_log[k])
            new_state.accuracy_log = merged_accuracy_log

            merged_prop_seen_log= {}
            for k in state_A.prop_seen_log.keys():
                if k in state_B.prop_seen_log:
                    merged_prop_seen_log[k] = get_merge_average(state_A.prop_seen_log[k], state_B.prop_seen_log[k])
                else:
                    merged_prop_seen_log[k] = int(state_A.prop_seen_log[k] * merge_proportion)
            for k in state_B.prop_seen_log.keys():
                if k in state_A.prop_seen_log:
                    pass
                else:
                    merged_prop_seen_log[k] = int(state_B.prop_seen_log[k] * (1 - merge_proportion))
            new_state.prop_seen_log = merged_prop_seen_log

            new_state.age = get_merge_average(state_A.age, state_B.age)
            new_state.LRU = get_merge_average(state_A.LRU, state_B.LRU)
            new_state.estimated_reuse_proportion = get_merge_average(state_A.estimated_reuse_proportion, state_B.estimated_reuse_proportion)
            new_state.estimated_reuse_advantage = get_merge_average(state_A.estimated_reuse_advantage, state_B.estimated_reuse_advantage)
            new_state.estimated_total_advantage = get_merge_average(state_A.estimated_total_advantage, state_B.estimated_total_advantage)


            if len(state_A.past_accuracy) > 1 and len(state_B.past_accuracy) > 1:
                num_past_accuracys = get_merge_average(len(state_A.past_accuracy), len(state_B.past_accuracy))
                final_past_accuracy = (get_merge_average(state_A.past_accuracy[-1][0], state_B.past_accuracy[-1][0]),
                                    get_merge_average(state_A.past_accuracy[-1][1], state_B.past_accuracy[-1][1]),
                                    )
                semifinal_past_accuracy = (get_merge_average(state_A.past_accuracy[-2][0], state_B.past_accuracy[-2][0]),
                                    get_merge_average(state_A.past_accuracy[-2][1], state_B.past_accuracy[-2][1]),
                                    )
                merged_past_accuracys = []
                for e in range(num_past_accuracys):
                    merged_past_accuracys.append([e, e])
                merged_past_accuracys[-2] = semifinal_past_accuracy
                merged_past_accuracys[-1] = final_past_accuracy
                new_state.past_accuracy = merged_past_accuracys
            else:
                if len(state_A.past_accuracy) > len(state_B.past_accuracy):
                    new_state.past_accuracy = state_A.past_accuracy
                else:
                    new_state.past_accuracy = state_B.past_accuracy
                
            new_state.start_ex = get_merge_average(state_A.start_ex, state_B.start_ex)
            new_state.end_ex = get_merge_average(state_A.end_ex, state_B.end_ex)
            new_state.cumulative_ex = get_merge_average(state_A.cumulative_ex, state_B.cumulative_ex)
            new_state.estimated_past_advantage = get_merge_average(state_A.estimated_past_advantage, state_B.estimated_past_advantage)
            new_state.extrapolated_advantage = get_merge_average(state_A.extrapolated_advantage, state_B.extrapolated_advantage)
            new_state.main_counter = get_merge_average(state_A.main_counter, state_B.main_counter)
            new_state.score_right = get_merge_average(state_A.score_right, state_B.score_right)
            new_state.score_wrong = get_merge_average(state_A.score_wrong, state_B.score_wrong)
            new_state.score_invalid_temporal = get_merge_average(state_A.score_invalid_temporal, state_B.score_invalid_temporal)
            new_state.score_invalid_majority = get_merge_average(state_A.score_invalid_majority, state_B.score_invalid_majority)
            new_state.temporal_right = get_merge_average(state_A.temporal_right, state_B.temporal_right)
            new_state.temporal_wrong = get_merge_average(state_A.temporal_wrong, state_B.temporal_wrong)
            new_state.acc_right = get_merge_average(state_A.acc_right, state_B.acc_right)
            new_state.acc_wrong = get_merge_average(state_A.acc_wrong, state_B.acc_wrong)

            new_state.recent_score = deque()
            for i in range(min(len(state_A.recent_score), len(state_B.recent_score))):
                new_state.recent_score.appendleft(get_merge_average(state_A.recent_score[-1 * (i+1)], state_B.recent_score[-1 * (i+1)]))
            new_state.last_props_right = deque()
            for i in range(min(len(state_A.last_props_right), len(state_B.last_props_right))):
                new_state.last_props_right.appendleft(get_merge_average(state_A.last_props_right[-1 * (i+1)], state_B.last_props_right[-1 * (i+1)]))
            # if not new_accuracy_log is None:
            #     new_state.accuracy_log = new_accuracy_log

            # new_state.evolution = list(range(max(len(state_A.evolution), len(state_B.evolution))))


            self.states[new_state_id] = new_state
            #print(self.states)
            return new_state
        else:
            return None

    def surrogate_train(self, state_A, state_B, new_state, feature_description, k):
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
            new_state.main_model.partial_fit(np_x, y)
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
            if not self.suppress: print(s.id)
            if not self.suppress: print(s.estimated_reuse_proportion)
            if not self.suppress: print(s.estimated_reuse_advantage)
            if not self.suppress: print(f"State {s.id} reuse: {reuse}")
            if min_reuse is None or reuse < min_reuse:
                min_id = s.id
                min_reuse = reuse
        return min_id

    def get_min_rAAuc_state(self):
        min_reuse = None
        min_id = None
        for s in self.states.values():
            if s.id == self.active_state_id:
                continue
            s.calculate_total_advantage(self.recent_states)
            evol_score = 1
            if len(s.evolution) > 0:
                evol_score = s.evolution[-1][2]
            reuse = s.estimated_reuse_proportion * evol_score
            if not self.suppress: print(s.id)
            if not self.suppress: print(s.estimated_reuse_proportion)
            if not self.suppress: print(evol_score)
            if not self.suppress: print(f"State {s.id} reuseAuc: {reuse}")
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
            if not self.suppress: print(s.id)
            if not self.suppress: print(s.estimated_reuse_proportion)
            if not self.suppress: print(score)
            if not self.suppress: print(f"State {s.id} reuse: {reuse}")
            if min_reuse is None or reuse < min_reuse:
                min_id = s.id
                min_reuse = reuse
        return min_id

    def get_min_p_rA_state(self):
        min_reuse = None
        min_id = None
        for s in self.states.values():
            if s.id == self.active_state_id:
                continue
            s.calculate_total_advantage(self.recent_states)
            reuse = s.estimated_reuse_proportion * s.estimated_past_advantage
            if not self.suppress: print(s.id)
            if not self.suppress: print(s.estimated_reuse_proportion)
            if not self.suppress: print(s.estimated_past_advantage)
            if not self.suppress: print(f"State {s.id} p_reuse: {reuse}")
            if min_reuse is None or reuse < min_reuse:
                min_id = s.id
                min_reuse = reuse
        return min_id

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

    def cull_states(self, recent_window):
        # if not self.suppress: print(f"Len: {len(self.states.values())}, Limit: {self.concept_limit}")
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
                    cull_state = self.get_min_p_rA_state()
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
                
                # if not self.suppress: print(f"Deleting: {min_id} with rA of {min_reuse}")
                if not self.suppress: print(f"Culling {cull_state}")
                del self.states[cull_state]
                self.deleted_states.append(cull_state)
        post_num_states = len(self.states.values())
        self.suppress = sup_val
        # print(f"Initial: {num_states}, post: {post_num_states}, cl: {self.concept_limit}")

# def get_kappa_agreement(A_predictions, B_predictions):
#     N = len(A_predictions)
#     print(A_predictions == B_predictions)
#     n_A = Counter()
#     n_B = Counter()
#     p_o_sum = 0
#     K = {}
#     for i in range(N):
#         p_A = A_predictions[i]
#         p_B = B_predictions[i]
#         if p_A == p_B:
#             p_o_sum += 1
#         n_A[p_A] += 1
#         n_B[p_B] += 1
#         if p_A not in K:
#             K[p_A] = 0
#         if p_B not in K:
#             K[p_B] = 0
#     p_o = p_o_sum / N
#     p_e = 0
#     for k in K.keys():
#         p_e += n_A[k] * n_B[k]
#     p_e = (p_e) / (N * N)

#     if (1 - p_e) == 0:
#         if (p_o - p_e) > 0:
#             kappa = 1
#         elif (p_o - p_e) == 0:
#             kappa = 0
#         else:
#             kappa = -1
#     else:
#         kappa = (p_o - p_e) / (1 - p_e)
#     return kappa

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
