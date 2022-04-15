import sys, os, glob
import math
import time
from collections import Counter
import pickle

from PhDCode.Classifier.advantage_fsm_o.fsm import FSM
from PhDCode.Classifier.advantage_fsm_o.systemStats import systemStats

from PhDCode.Classifier.advantage_fsm_o.tracksplit_hoeffding_tree import TS_HoeffdingTree
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from PhDCode.Classifier.advantage_fsm_o.tracksplit_HAT import TS_HAT
from PhDCode.Classifier.advantage_fsm_o.TS_ARFTREE import TS_ARFHoeffdingTree
from skmultiflow.bayes.naive_bayes import NaiveBayes
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.utils import get_dimensions, normalize_values_in_dict, check_random_state, check_weights


import numpy as np
import pandas as pd

def make_detector(warn = False, s = 1e-5):
    sensitivity = s * 10 if warn else s
    return ADWIN(delta=sensitivity)

def run_fsm(datastream, options, suppress = False,
 fsm = None, system_stats=None, detector = None, stream_examples = None, save_checkpoint = False, name = None,
 run_non_active_states = False, concept_chain = None, optimal_selection = False):
    if not suppress: print(datastream)
    if detector == None:
        detector = make_detector(s=options.sensitivity)
        warn_detector = make_detector(warn=True,s=options.sensitivity)
    else:
        detector = make_detector(s=options.sensitivity)
        warn_detector = make_detector(warn=True,s=options.sensitivity)
    in_warning = False

    if fsm == None:
        fsm = FSM(concept_limit = options.concept_limit, memory_strategy= options.memory_management)
        fsm.suppress = suppress
        # Initialize components
        starting_id = 0
        if not (concept_chain is None) and 0 in concept_chain:
            starting_id = concept_chain[0]
        fsm.make_state(options.learner, s_id = starting_id)
        fsm.active_state_id = starting_id
    if system_stats == None:
        system_stats = systemStats()
        system_stats.state_control_log.append([0, 0, None])
        system_stats.last_seen_window_length = options.window

    random_state = None
    _random_state = check_random_state(random_state)
    cancelled = False
    update_percent = int(datastream.n_remaining_samples() / 100)
    checkpoint_instances = update_percent * 10
    percent_start = time.process_time()
    if stream_examples == None:
        stream_examples = []
        # Run the main loop
        ex = -1
    else:
        ex = len(stream_examples)
    state_recurrence_checks = 0
    if optimal_selection:
        print(concept_chain)
        optimal_alias = {}
    while datastream.has_more_samples() and not cancelled:
        try:
            ex += 1

            # Basic get example, make prediction loop
            X,y = datastream.next_sample(options.batch_size)
            for b in range(options.batch_size):
                np_x = np.array(X[b]).reshape(1, len(X[b]))

                # For each state, predict and log.
                # The system logging is done only for the active state.
                
                if run_non_active_states:
                    for s_i,state in enumerate(fsm.states.values()):
                        # We want to predict the current state with the real model
                        # and the background states with their comparison model
                        predict = state.comparison_model.predict(np_x)[0]
                        is_correct = 1 if predict == y[b] else 0
                        state.add_comparison_prediction(X[b], y[b], predict, is_correct, ex)
     
                predict = fsm.get_state().main_model.predict(np_x)[0]
                is_current_state_correct = 1 if predict == y[b] else 0
                system_stats.add_prediction(X[b], y[b], predict, is_current_state_correct, ex)
                fsm.get_state().add_main_prediction(X[b], y[b], predict, is_current_state_correct, ex)   
                        
                skip_selection = concept_chain != None and optimal_selection
                found_change = False
                if not skip_selection:

                    # Add to detector, and get any alerts
                    detector.add_element(is_current_state_correct)
                    warn_detector.add_element(is_current_state_correct)
                    found_change = detector.detected_change()

                    if(warn_detector.detected_change()):
                        in_warning = True
                    if in_warning:
                        system_stats.add_warn_prediction(X[b], y[b], predict, is_current_state_correct, ex)
                else:
                    if ex in concept_chain and ex != 0:
                        found_change = True
                        switch_to_id = concept_chain[ex]
                        if switch_to_id in optimal_alias:
                            switch_to_id = optimal_alias[switch_to_id]

                found_change = False
                if(found_change):
                    
                    state_recurrence_checks += 1
                    if not suppress: print(f"change detected at {ex}")
                    
                    # Check for state to transition to.
                    recent_window = system_stats.warn_log
                    if len(recent_window) < 100 or True:
                        recent_window = system_stats.last_seen_examples
                    
                    if not skip_selection:
                        new_state_id, is_recurring = fsm.get_recurring_state_or_new(recent_window, options.learner, system_stats)
                        if new_state_id != fsm.active_state_id:
                            fsm.transition_to(new_state_id, ex)
                    else:
                        print("")
                        print(f"{ex}: Switch to {switch_to_id}")
                        to_recurring_state = switch_to_id in [s.id for s in fsm.states.values()]
                        if not to_recurring_state:
                            new_id = fsm.make_state(options.learner, s_id=switch_to_id)
                            fsm.transition_to(new_id, ex)
                            optimal_alias[switch_to_id] = new_id
                        else:
                            fsm.transition_to(switch_to_id, ex)
                    fsm.cull_states(recent_window)
                    
                    # Logging
                    if not suppress: print(f"Transitioned to {new_state_id}")
                    system_stats.log_change_detection(ex)
                    system_stats.state_control_log[-1][2] = ex
                    system_stats.state_control_log.append([fsm.active_state_id, ex, None])

                    # We reset the detector and model structure tracking, as the previous models accuracy
                    # should not affect new changes.
                    # detector = ADWIN(delta=0.2)
                    detector = make_detector(s=options.sensitivity)
                    warn_detector = make_detector(warn=True, s=options.sensitivity)
                    in_warning = False
                    system_stats.clear_warn_log()
                    if(hasattr(fsm.get_state().main_model, 'splits_since_reset')):
                        system_stats.model_update_status = fsm.get_state().main_model.splits_since_reset

                # We want to track splits in the hoeffding tree.
                # Splits effect accuracy (in either direction), which can trigger
                # a change detection even if no drift is present. So we reset.
                if( hasattr(fsm.get_state().main_model, 'splits_since_reset') and fsm.get_state().main_model.splits_since_reset > system_stats.model_update_status):
                    system_stats.log_model_update(ex, fsm.get_state().main_model.splits_since_reset)
                    # detector = ADWIN(delta=0.2)
                    detector = make_detector(s=options.sensitivity)
                    warn_detector = make_detector(warn=True, s=options.sensitivity)
                    fsm.get_state().evolution.append(len(fsm.get_state().evolution))

                # Fit model
                w = _random_state.poisson(6)
                w= 1
                fsm.get_state().main_model.partial_fit(np_x, y, sample_weight = np.asarray([w]))
                if run_non_active_states:
                    for s_i,s in enumerate(fsm.states.values()):
                        s.comparison_model.partial_fit(np_x, y)

            if ex % update_percent == 0:
                percent_end = time.process_time()
                last_percent_time = percent_end - percent_start
                percent = ex // update_percent
                remaining_percent = 100 - percent
                remaining_time = remaining_percent * last_percent_time
                print(f"Example {ex} of {ex + datastream.n_remaining_samples()}. {percent}%. {round(remaining_time / 6) / 10} minutes to complete. Sys Acc: {round(system_stats.model_stats.right / (system_stats.model_stats.right + system_stats.model_stats.wrong) * 100)/100}. Used {len(fsm.states)} States. Checks {state_recurrence_checks}.\r", end = "")
                percent_start = time.process_time()
            if save_checkpoint:
                if ex % checkpoint_instances == 0 and ex != 0:
                    print(f"Saving Checkpoint")
                    make_sys_csv(options.experiment_directory, fsm, datastream.concept_chain, system_stats, datastream, options = options, name = name, segment= ex // checkpoint_instances)
                    # Reset to save ram
                    for s in fsm.states.values():
                        s.comparison_model_stats.sliding_window_accuracy_log = []
                        s.comparison_model_stats.correct_log = []
                        s.comparison_model_stats.p_log = []
                        s.comparison_model_stats.y_log = []
                        s.main_model_stats.sliding_window_accuracy_log = []
                        s.main_model_stats.correct_log = []
                        s.main_model_stats.p_log = []
                        s.main_model_stats.y_log = []
                    system_stats.model_stats.sliding_window_accuracy_log = []
                    system_stats.model_stats.correct_log = []
                    system_stats.model_stats.p_log = []
                    system_stats.model_stats.y_log = []
        except KeyboardInterrupt:
            cancelled = True
    system_stats.state_control_log[-1][2] = ex
    if not hasattr(datastream, 'concept_chain'):
        datastream.concept_chain = None
    make_sys_csv(options.experiment_directory, fsm, datastream.concept_chain, system_stats, datastream, options = options, name = name, segment= 99)
    with open(f'{options.experiment_directory}{os.sep}{name}-merges.pickle', 'wb') as f:
        pickle.dump(fsm.merge_log, f)
    stitch_csv(options.experiment_directory, name)
    return (fsm, system_stats, datastream.concept_chain, datastream, stream_examples)

class FSMOptions:
    def __init__(self):
        self.batch_size = 1
        self.seed = 42
        self.similarity_measure = "ACC"
        self.learner = TS_ARFHoeffdingTree
        self.experiment_directory = 'datastreams'
        self.memory_management = "rA"
        self.sensitivity = 0.02
        self.window = 175

def make_sys_csv(directory, fsm, concept_chain, system_stats, datastream, options = None, name = None, segment = 0):
    if name == None:
        if options != None:
            name = '-'.join('system', str(options.concept_limit))
        else:
            name = '-'.join('system', str(-1))
    sys_csv_filename = f'{directory}{os.sep}{name}-{segment}.csv'

    if not os.path.exists(sys_csv_filename):
        sys_results = pd.DataFrame(system_stats.model_stats.sliding_window_accuracy_log, columns=['example', 'sliding_window_accuracy'])
        num_results = sys_results.shape[0]
        #print(fsm.states)
        if False:
            for s in fsm.states.values():
                # Get the list of (ts, acc) values for when the state was in control
                # Put in a dict where ts is the key.
                m_xs = dict(s.main_model_stats.sliding_window_accuracy_log)

                # xs = list of timesteps
                xs = list(map(lambda a: a[0], s.comparison_model_stats.sliding_window_accuracy_log))

                #ys = list of accuracies. Either the comparison model if not in control, or the main model if it
                # was in contol
                ys = list(map(lambda a: a[1] if a[0] not in m_xs else m_xs[a[0]],
                    s.comparison_model_stats.sliding_window_accuracy_log))
                
                ys = np.pad(ys, (num_results-len(ys), 0), 'constant', constant_values=np.nan)
            sys_results[f'state_{s.id}_acc'] = ys

        sys_results['is_correct'] = np.array([x[1] for x in system_stats.model_stats.correct_log])
        sys_results['p'] = np.array([x[1] for x in system_stats.model_stats.p_log])
        sys_results['y'] = np.array([x[1] for x in system_stats.model_stats.y_log])
        correct_counts = Counter((x[1] for x in system_stats.model_stats.correct_log))
        num_right = correct_counts[1]
        num_wrong = correct_counts[0]
        total_predictions = num_right + num_wrong
        initial_right = system_stats.model_stats.right - num_right
        initial_wrong = system_stats.model_stats.wrong - num_wrong
        if initial_right + initial_wrong > 0:
            initial_accuracy = initial_right / (initial_right + initial_wrong)
        else:
            initial_accuracy = 0
        cum_right = (np.cumsum(sys_results['is_correct'].tolist()) + initial_right)
        total_predictions = (np.array(range(initial_right + initial_wrong + 1, initial_right + initial_wrong + sys_results.shape[0] + 1))  )
        sys_results['overall_accuracy'] =  (cum_right/total_predictions)  * 100
        sys_results['sliding_window_accuracy'] = sys_results['sliding_window_accuracy'] * 100
        start = system_stats.model_stats.sliding_window_accuracy_log[0][0]
        end = system_stats.model_stats.sliding_window_accuracy_log[-1][0] + 1
        syscs = get_system_concepts(system_stats, end - start, start, end)
        sc = np.array(syscs)
        sys_results['system_concept'] = sc
        sys_results['change_detected'] = get_model_change_detections(num_results, system_stats, start, end)
        sys_results.to_csv(sys_csv_filename, index = False)

def get_example_system_concept(system_concepts, ex_index, start, end):
    """ Given [(gt_concept, start_i, end_i)...] , [(sys_concept, start_i, end_i)...]
        Return the ground truth and system concepts occuring at a given index."""
    system_concept = None
    for system_c, s_i, e_i in system_concepts:
        if e_i != None:
            if s_i <= ex_index < e_i:
                system_concept = system_c
                break
        else:
            if s_i <= ex_index:
                system_concept = system_c
                break
    if system_concept == None:
        system_concept = system_c
    return (system_concept)

def get_system_concepts_by_example(system_concepts, ns, start, end):
    sysc_by_ex = []
    print(start)
    print(end)
    for ex in range(start, end):
        sample_sys_concept = get_example_system_concept(system_concepts, ex, start, end)
        sysc_by_ex.append(sample_sys_concept)
    #print(sysc_by_ex)
    return sysc_by_ex

def get_system_concepts(system_stats, ns, start, end):
    system_concepts = system_stats.state_control_log
    return get_system_concepts_by_example(system_concepts, ns, start, end)

def get_model_change_detections(num_samples, system_stats, start, end):
    detections = np.zeros(num_samples)
    for d_i, d in enumerate(system_stats.change_detection_log):
        if d < start or d > end:
            continue
        detections[d - start] = 1
    return detections

def stitch_csv(directory, name):
    fns = glob.glob(os.sep.join([directory, f"{name}*.csv"]))
    fns.sort()
    print(fns)
    drift_fn = f'{directory}{os.sep}drift_info.csv'
    sys_csv_filename = f'{directory}{os.sep}{name}.csv'
    data = None
    set_data = False
    for fn in fns:
        if fn != drift_fn:
            if not set_data:
                data = pd.read_csv(fn)
                set_data = True
            else:
                data = data.append(pd.read_csv(fn), ignore_index=True)
    if os.path.exists(drift_fn):
        drift_data = pd.read_csv(drift_fn)
        drift_data.columns = ['example', 'ground_truth_concept', 'drift_occured']
        columns = data.columns

        full_data = data.merge(drift_data, on = 'example', how = 'left', sort = False)
        full_data.to_csv(sys_csv_filename)
    else:
        data.to_csv(sys_csv_filename)
    
    for fn in fns:
        os.remove(fn)


if __name__ == "__main__":

    args, options = makeExperiment.parse_options()
    # Setup system components
    if args['gradual']:
        datastream = RecurringConceptGradualStream(options.stream_type, options.ds_length, 0, options.concept_chain, window_size = options.window_size, seed = options.seed)
    else:
        datastream = RecurringConceptStream(options.stream_type, options.ds_length, 0, options.concept_chain, seed = options.seed)

    run_fsm(datastream, options)
