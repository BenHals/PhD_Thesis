
import math
import numpy as np
from typing import List


def check_for_merge(state_A, state_B):
    window = [e.X for e in state_A.main_model_stats.first_seen_examples + state_B.main_model_stats.first_seen_examples]

    n_samples = len(window)
    if n_samples < 50:
        return True, 1
    n_features = len(window[0])
    window_np = np.array(window).reshape(n_samples, n_features)

    state_A_predictions = state_A.main_model.predict(window_np)
    state_B_predictions = state_B.main_model.predict(window_np)

    kappa = get_kappa_agreement(state_A_predictions, state_B_predictions)
    #print(kappa)
    return kappa > self.merge_similarity, kappa


def merge_states(state_A, state_B, learner, feature_description, k, merge_similarity):
    new_state_id = self.make_state(learner)

    new_learner = self.construct_state(-1, learner)
    if merge_similarity > 0.95:
        new_learner, fitted = self.acc_merge(state_A, state_B, new_learner, feature_description, k)
    else:
        new_learner, fitted = self.surrogate_merge(state_A, state_B, new_learner, feature_description, k)

    
    if fitted:
        del states[state_A.id]
        del states[state_B.id]
        if active_state_id == state_A.id:
            active_state_id = new_state_id
        if active_state_id == state_B.id:
            active_state_id = new_state_id
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
        self.observed_merges[state_A.id] = new_state_id
        self.observed_merges[state_B.id] = new_state_id

        states[new_state_id] = new_learner
        return new_learner
    else:
        return None

def surrogate_merge(state_A, state_B, new_state, feature_description, k):
    categorical_proportions = {}
    # if not self.suppress: #print(" ")
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
    # if not self.suppress: #print(np_x.shape)
    np_x = np_x.transpose()
    # if not self.suppress: #print(np_x.shape)
    
    try:
        y = np.zeros(100000)
        y[::2] = state_A.main_model.predict(np_x[::2])
        # if not self.suppress: #print(y[:5])
        y[1::2] = state_B.main_model.predict(np_x[1::2])
        # if not self.suppress: #print(y[:5])
        
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
                new_state.evolution.append([self.ex, len(new_state.evolution), None, None, None])
        fitted = True
    except:
        # if not self.suppress: #print("fit failed")
        pass
    
    # for i in range(50000):
    #     #print(f"surrogate {i}\r", end = "")
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
    # if not self.suppress: #print("Done")
    return new_state, fitted


def acc_merge(state_A, state_B, new_state, feature_description, k):
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

def get_reuse_proportion(state):
    """ Returns the proportion of the stream a state has been active since it was first observed.
    We assume that this proportion continues in the future.
    """
    # We can sometimes get a zero, this may happen when a new state is first created, but we should avoid
    # this if it is the active state.
    # Otherwise, this can occur with backtracking, when a state is reset to when it was first created.
    # In this case, we just return the reuse proportion 0 since it is as if the state was never in use.
    if state.age == 0:
        return 0
    return state.active_age / state.age

def get_min_rA_state(states, active_state_id, recent_window):
    state_values = []
    for s in states.values():
        if s.id == active_state_id:
            continue
        reuse_estimate = get_reuse_proportion(s)
        advantage_estimate = s.current_evolution + 1
        total_benefit_estimate = reuse_estimate * advantage_estimate
        state_values.append((total_benefit_estimate, s.id))
    min_val, min_id = min(state_values, key=lambda x: x[0])
    return min_id
    
def get_min_score_state(states, active_state_id, recent_window):
    min_reuse = None
    min_id = None
    for s in states.values():
        if s.id == active_state_id:
            continue
        s.calculate_total_advantage(self.recent_states)
        score = s.calculate_past_score_advantage()
        reuse = s.estimated_reuse_proportion * score
        if min_reuse is None or reuse < min_reuse:
            min_id = s.id
            min_reuse = reuse
    return min_id

def get_min_AAC_state(states, active_state_id, recent_window):
    min_reuse = None
    min_id = None
    for s in states.values():
        if s.id == active_state_id:
            continue
        s.calculate_total_advantage(self.recent_states)
        if s.evolution[-1][4] is None:
            # #print("checking")
            s.set_evolution_stats()
        reuse = s.estimated_reuse_proportion * s.evolution[-1][4]
        if min_reuse is None or reuse < min_reuse:
            min_id = s.id
            min_reuse = reuse
    return min_id

def get_max_age_state(states, active_state_id, recent_window):
    state_values = []
    for s in states.values():
        if s.id == active_state_id:
            continue
        # Benefit here is the inverse of age
        total_benefit_estimate = s.age * -1
        state_values.append((total_benefit_estimate, s.id))
    min_val, min_id = min(state_values, key=lambda x: x[0])
    return min_id

# def get_max_age_state(states, active_state_id, recent_window):
#     max_age = None
#     max_id = None
#     for s in states.values():
#         if s.id == active_state_id:
#             continue
#         # if not self.suppress: #print(f"State {s.id} age: {s.age}")
        
#         if max_age is None or s.age > max_age:
#             max_id = s.id
#             max_age = s.age
#     return max_id

def get_max_LRU_state(states, active_state_id, recent_window):
    state_values = []
    for s in states.values():
        if s.id == active_state_id:
            continue
        # Benefit here is the inverse of age since last active, i.e., the highest value is lowest total benefit estimate
        total_benefit_estimate = s.age_since_last_active * -1
        state_values.append((total_benefit_estimate, s.id))
    min_val, min_id = min(state_values, key=lambda x: x[0])
    return min_id

# def get_max_LRU_state(states, active_state_id, recent_window):
#     max_LRU = None
#     max_id = None
#     for s in states.values():
#         if s.id == active_state_id:
#             continue
#         # if not self.suppress: #print(f"State {s.id} LRU: {s.LRU}")
        
#         if max_LRU is None or s.LRU > max_LRU:
#             max_id = s.id
#             max_LRU = s.LRU
#     return max_id



def get_min_div_state(states, active_state_id, recent_window):
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

    if len(states.values()) <= 2:
        return list(states.values())[0].id

    state_predictions = []
    for s_i, s in enumerate(states.values()):
        state_predictions.append(s.main_model.predict(recent_X))

    for s_i, s in enumerate(states.values()):
        if s.id == active_state_id:
            continue
        
        preds = []
        for c_i, c in enumerate(states.values()):
            if s_i == c_i:
                continue
            preds.append(state_predictions[c_i])

        # The entropy if state s is removed.
        E = get_diversity_entropy(preds, recent_y)
        # if not self.suppress: #print(f"State {s.id} acc: {E}")
        if max_E is None or E > max_E:
            max_id = s.id
            max_E = E
    
    return max_id



def get_min_acc_state(states, active_state_id, recent_window):
    min_acc = None
    min_id = None

    n_samples = len(recent_window)
    n_features = len(recent_window[0].X)
    recent_X = np.array([e.X for e in recent_window]).reshape(n_samples, n_features)
    recent_y = np.array([e.y for e in recent_window])
    
    for s in states.values():
        if s.id == active_state_id:
            continue
        _, acc = get_recent_window_accuracy(s, recent_X, recent_y, False)

        if min_acc is None or acc < min_acc:
            min_id = s.id
            min_acc = acc
    return min_id

def get_recent_window_accuracy(state, recent_X, recent_y, val):
    pass

def get_cull_states(states, active_state_id:int, repository_maximum:int, valuation_policy, recent_window) -> List[int]:
    """ Returns a list of state IDs to delete. Does not handle deletion.

    Parameters
    ----------
    states: Dict<Int, State>
        A dict mapping state IDs to states. Represents the current repository.
    
    active_state_id: Int
        ID of the active state
    
    repository_maximum: Int
        The maximum number of states in the repository. -1 for no limit.
    
    valuation_policy: Str
        The valuation policy name to use
    
    recent_window: List<Labelled Observation>
        A list of recent observations to compute statistics.
    """
    num_states :int = len(states.values())
    culled_ids :List[int] = []
    # Short circut if negative to perform no check
    if repository_maximum < 0:
        return culled_ids

    while (num_states - len(culled_ids)) > repository_maximum:
        cull_state = None
        if valuation_policy == 'rA':
            cull_state = get_min_rA_state(states, active_state_id, recent_window)
        # elif valuation_policy == 'auc':
        #     cull_state = get_min_AAC_state(states, active_state_id, recent_window)
        elif valuation_policy == 'age':
            cull_state = get_max_age_state(states, active_state_id, recent_window)
        elif valuation_policy == 'LRU':
            cull_state = get_max_LRU_state(states, active_state_id, recent_window)
        # elif valuation_policy == 'acc':
        #     cull_state = get_min_acc_state(states, active_state_id, recent_window)
        # elif valuation_policy == 'score':
        #     cull_state = get_min_score_state(states, active_state_id, recent_window)
        # elif valuation_policy == 'div':
        #     cull_state = get_min_div_state(states, active_state_id, recent_window)
        else:
            raise ValueError("Valuation policy not implemented yet")
        
        culled_ids.append(cull_state)
    post_num_states = len(states.values())

    return culled_ids


def repository_memory_management(states, active_state_id:int, repository_maximum:int, valuation_policy, recent_window):
    """ Handles memory management of a dict of states in place. Deletes states when the size of the repo is above repository_maximum

    Parameters
    ----------
    states: Dict<Int, State>
        A dict mapping state IDs to states. Represents the current repository.
    
    active_state_id: Int
        ID of the active state
    
    repository_maximum: Int
        The maximum number of states in the repository. -1 for no limit.
    
    valuation_policy: Str
        The valuation policy name to use
    
    recent_window: List<Labelled Observation>
        A list of recent observations to compute statistics.
    """

    culled_ids = get_cull_states(states, active_state_id, repository_maximum, valuation_policy, recent_window)
    for cull_id in culled_ids:
        del states[cull_id]






