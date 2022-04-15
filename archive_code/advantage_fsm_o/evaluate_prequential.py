import numpy as np
import pandas as pd
import os, glob, psutil, sys
import time
import pickle

from PhDCode.Classifier.advantage_fsm_o.systemStats import systemStats
from PhDCode.Classifier.advantage_fsm_o.fsm import FSM
from collections import Counter
from pympler import classtracker

def get_size(obj, seen=None, level = 0, print_sum = False, hide_evolution = False, fo = None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        for k in obj.keys():
            # Exclude summary statistics only used for display
            if k in ["state_log", "correct_log", "p_log", "y_log", "sliding_window_accuracy_log", "reuse_log", "accuracy_log", "prop_seen_log", "past_accuracy", "recent_score", "recent_rec", "recent_queue", "recent_kts"]:
                continue
            if hide_evolution and k in ["evolution", "reuse_log", "accuracy_log", "prop_seen_log", "past_accuracy", "recent_score", "recent_rec", "recent_queue", "recent_kts"]:
                continue
            val_size = get_size(obj[k], seen, level= level + 1, print_sum = print_sum, hide_evolution = hide_evolution, fo = fo)
            key_size = get_size(k, seen, level= level + 1, print_sum = print_sum, hide_evolution = hide_evolution, fo = fo)
            if print_sum:
                if key_size + val_size > 10000:
                    print(f"{'-'.join([' ' for x in range(level)])}{k} size: {key_size + val_size}")
                    if not fo is None:
                        fo.write(f"{'-'.join([' ' for x in range(level)])}{k} size: {key_size + val_size}\n")
            size += (key_size + val_size)
        # size += sum([get_size(v, seen) for v in obj.values()])
        # size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen, level= level + 1, print_sum = print_sum, hide_evolution = hide_evolution, fo = fo)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen, level= level + 1, print_sum = print_sum, hide_evolution = hide_evolution, fo = fo) for i in obj])
    return size


def evaluate_prequential(datastream = None, classifier = None, save_checkpoint = True, directory = "", name = "Test", noise = 0, seed = 1):
    """
    Evaluate a classifier on a datastream prequentially (Evaluate each instance before training).

    Parameters
    ----------
    datastream: Object
        A scikit-multiflow datastream object
    
    classifier: Object
        A scikit-multiflow classifier object

    save_checkpoint: Bool
        True if should save results to csv file

    directory: Str
        The path to the base directory, to read and write files

    name: Str
        The name of the experiment, to name files and store config info
    """

    if datastream is None or classifier is None:
        raise ValueError("Missing datastream or classifier")
    cancelled = False

    max_memory = 0
    average_memory_sum = 0
    memory_checks = 0

    # How many examples between each info log to screen
    update_percent = int(datastream.n_remaining_samples() / 100)

    # How many examples between intermediary csv writes
    checkpoint_instances = update_percent * 10
    percent_start = time.process_time()

    # Holds evaluation statistics
    system_stats = systemStats()

    ex = -1

    rand_state = np.random.RandomState(seed)
    seen_y_values = {}
    
    tracker = classtracker.ClassTracker()
    if hasattr(classifier, 'fsm'):
        tracker.track_object(classifier.fsm)
    else:
        tracker.track_object(classifier)
    
    while datastream.has_more_samples() and not cancelled:
        ex += 1
        X,y = datastream.next_sample()

        
        


        if y[0] not in seen_y_values:
            seen_y_values[y[0]] = 1
        
        noise_rn = rand_state.random_sample()
        if noise_rn < noise:
            noise_y = rand_state.choice(list(seen_y_values.keys()))
            y = [noise_y]

        try:
            prediction = classifier.predict(X)[0]
        except Exception as e:
            print(e)
            print(X)
            print(ex)
            exit()
        correctly_classifies = prediction == y[0]

        system_stats.add_prediction(X[0], y[0], prediction, correctly_classifies, ex)

        classifier.partial_fit(X, y)

        # Log to screen
        if ex % update_percent == 0:

            # if hasattr(classifier, 'fsm'):
            #     with open("mem_check", "w") as f:
            #         memory_usage = get_size(classifier.fsm, hide_evolution = classifier.memory_management in ["LRU", "age", "acc", "div"], print_sum = True, fo = f)
            #         f.write(f"{memory_usage}\n")

            percent_end = time.process_time()
            last_percent_time = percent_end - percent_start
            percent = ex // update_percent
            remaining_percent = 100 - percent
            remaining_time = remaining_percent * last_percent_time
            state_str = ""
            if hasattr(classifier, 'fsm'):
                state_str = f"Using {len(classifier.fsm.states)} states."
            
            system_acc = round(system_stats.model_stats.right / (system_stats.model_stats.right + system_stats.model_stats.wrong) * 10000)/10000
            classifier_acc = 0
            if hasattr(classifier, 'system_stats'):
                classifier_acc = round(classifier.system_stats.model_stats.right / (classifier.system_stats.model_stats.right + classifier.system_stats.model_stats.wrong) * 10000)/10000
            print(f"Example {ex} of {ex + datastream.n_remaining_samples()}. {percent}%. {round(remaining_time / 6) / 10} minutes to complete. Sys Acc: {system_acc}, FSM Acc: {classifier_acc}. {state_str}\r", end = "")
            percent_start = time.process_time()

        # Create intermediary csv file
        if save_checkpoint:
            if ex % checkpoint_instances == 0 and ex != 0:
                print(" ")
                print(f"Saving Checkpoint")
                # memory_usage = psutil.Process().memory_full_info().uss
                if hasattr(classifier, 'fsm'):
                    # with open("mem_check", "w") as f:
                    memory_usage = get_size(classifier.fsm, hide_evolution = classifier.memory_management in ["LRU", "age", "acc", "div"], print_sum = False, fo = None)
                        # f.write(f"{memory_usage}\n")
                else:
                    memory_usage = get_size(classifier)

                average_memory_sum += memory_usage
                memory_checks += 1
                if memory_usage > max_memory:
                    max_memory = memory_usage
                tracker.create_snapshot(f'{ex % update_percent}')

                if hasattr(classifier, 'system_stats'):
                    make_sys_csv(directory, system_stats, name = name, segment= ex // checkpoint_instances, system_concepts= True, system_concepts_log = classifier.system_stats)
                else:
                    make_sys_csv(directory, system_stats, name = name, segment= ex // checkpoint_instances)
                # Reset to save ram
                system_stats.model_stats.sliding_window_accuracy_log = []
                system_stats.model_stats.correct_log = []
                system_stats.model_stats.p_log = []
                system_stats.model_stats.y_log = []
                if hasattr(classifier, 'reset_stats'):
                    classifier.reset_stats()
    
    # Write final results
    if not hasattr(datastream, 'concept_chain'):
        datastream.concept_chain = None
    if hasattr(classifier, 'system_stats'):
        make_sys_csv(directory, system_stats, name = name, segment= 99, system_concepts= True, system_concepts_log = classifier.system_stats)
    else:
        make_sys_csv(directory, system_stats, name = name, segment= 99)
    
    if hasattr(classifier, 'finish_up'):
        with open(f'{directory}{os.sep}{name}-merges.pickle', 'wb') as f:
            pickle.dump(classifier.finish_up(ex), f)
    stitch_csv(directory, name) 

    tracker.stats.print_summary()

    if hasattr(classifier, 'fsm'):
        with open(f'{directory}{os.sep}{name}-memorytree.txt', "w") as f:
            m = get_size(classifier.fsm, hide_evolution = classifier.memory_management in ["LRU", "age", "acc", "div"], print_sum= True, fo = f)
            f.write(f"{m}\n")
    else:
        get_size(classifier, print_sum= True)
    
    return average_memory_sum / memory_checks, max_memory


def make_sys_csv(directory, system_stats, name = None, segment = 0, system_concepts = False, system_concepts_log = None):
    if segment > 10:
        segment = 90 + (segment - 9)
    sys_csv_filename = f'{directory}{os.sep}{name}-{segment}.csv'

    if not os.path.exists(sys_csv_filename):
        if len(system_stats.model_stats.sliding_window_accuracy_log) < 10:
            return
        sys_results = pd.DataFrame(system_stats.model_stats.sliding_window_accuracy_log, columns=['example', 'sliding_window_accuracy'])
        num_results = sys_results.shape[0]

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

        if system_concepts:
            start = system_concepts_log.model_stats.sliding_window_accuracy_log[0][0]
            end = system_concepts_log.model_stats.sliding_window_accuracy_log[-1][0] + 1
            syscs = get_system_concepts(system_concepts_log, end - start, start, end)
            sc = np.array(syscs)
            sys_results['system_concept'] = sc
            alt_syscs = get_altered_system_concepts(system_concepts_log, end - start, start, end)
            alt_sc = np.array(alt_syscs)
            sys_results['alt_system_concept'] = alt_sc
            sys_results['change_detected'] = get_model_change_detections(num_results, system_concepts_log, start, end)
        else:
            df_len = len(sys_results['example'])
            sys_results['system_concept'] = pd.Series(np.zeros(df_len), index = sys_results.index)
            sys_results['change_detected'] = pd.Series(np.zeros(df_len), index = sys_results.index)
        sys_results.to_csv(sys_csv_filename, index = False)

def stitch_csv(directory, name):
    fns = glob.glob(os.sep.join([directory, f"{name}*.csv"]))
    fns.sort(key = lambda x: int(x.split('-')[-1].split('.')[0]))
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

def get_altered_system_concepts(system_stats, ns, start, end):
    if hasattr(system_stats, 'state_control_log_altered'):
        system_concepts = system_stats.state_control_log_altered
    else:
        system_concepts = system_stats.state_control_log
    return get_system_concepts_by_example(system_concepts, ns, start, end)

def get_model_change_detections(num_samples, system_stats, start, end):
    detections = np.zeros(num_samples)
    for d_i, d in enumerate(system_stats.change_detection_log):
        if d < start or d > end:
            continue
        detections[d - start] = 1
    return detections