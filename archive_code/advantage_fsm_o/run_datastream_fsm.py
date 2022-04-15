import sys, os, psutil
import glob
import argparse
import random
import pickle
import json
import subprocess
import fsmsys
import PhDCode.Classifier.advantage_fsm_o.evaluate_prequential
from fsm_classifier import FSMClassifier
from skmultiflow.data import DataStream
from PhDCode.Classifier.advantage_fsm_o.tracksplit_hoeffding_tree import TS_HoeffdingTree
from PhDCode.Classifier.advantage_fsm_o.tracksplit_HAT import TS_HAT
from PhDCode.Classifier.advantage_fsm_o.TS_ARFTREE import TS_ARFHoeffdingTree
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from skmultiflow.drift_detection.adwin import ADWIN
import numpy as np
from scipy.io import arff
from time import process_time

import pandas as pd
import numpy as np




def start_run(options):
    if not os.path.exists(options.experiment_directory):
        print('No Directory')
        return
    datastream_filename = None
    datastream_pickle_filename = None
    fns = glob.glob(os.sep.join([options.experiment_directory, "*.ARFF"]))
    print(fns)
    # mm_options = ['rA', 'age', 'LRU', 'acc'] if options.memory_management == 'all' else [options.memory_management]
    for fn in fns:
        save_mm = options.memory_management
        mm_options = [options.memory_management] if options.memory_management != 'all' else ["score", "rA", 'auc', "age", "LRU", 'acc', 'div']
        mm_options = mm_options if options.memory_management != 'mine' else ['auc', "score", "rA"]
        for mm in mm_options:
            print(mm)
            options.memory_management = mm
            name = '-'.join(['system', str(options.noise), str(options.concept_limit), str(options.memory_management), str(options.sensitivity), str(options.window), str(options.optimal_selection), str(options.learner_str), str(options.poisson), str(options.seed), str(options.optimal_drift), str(options.similarity_measure), str(options.merge_strategy)])
            name_no_seed = '-'.join(['system', str(options.noise), str(options.concept_limit), str(options.memory_management), str(options.sensitivity), str(options.window), str(options.optimal_selection), str(options.learner_str), str(options.poisson), "*", str(options.optimal_drift), str(options.similarity_measure), str(options.merge_strategy)])
            print(name)
            if fn.split('.')[-1] == 'ARFF':
                actual_fn = fn.split(os.sep)[-1]
                fn_path = os.sep.join(fn.split(os.sep)[:-1])
                print(actual_fn)
                print(fn_path)
                pickle_fn = f"{actual_fn.split('.')[0]}_concept_chain.pickle"
                pickle_full_fn = os.sep.join([fn_path, pickle_fn])
                csv_fn = f"{name}.csv"
                csv_full_fn = os.sep.join([fn_path, csv_fn])
                print(f"checking {csv_full_fn}")

                concept_chain_exists = os.path.exists(pickle_full_fn)

                if not options.no_chain and not concept_chain_exists:
                    print("No concept chain pickle file")
                    continue

                skip_file = False

                existing_matches = glob.glob(os.sep.join([fn_path, f"{name_no_seed}.csv"]))
                if len(existing_matches):
                    if any([os.path.getsize(x) > 2000 for x in existing_matches]):
                        skip_file = True
                if not skip_file:
                    datastream_filename = fn
                    datastream_pickle_filename = pickle_full_fn
                else:
                    print(f'{csv_full_fn} exists')

            if datastream_filename == None:
                print('Not datastream file')
                continue
            print(datastream_filename)

            if not options.no_chain:
                with open(f'{datastream_pickle_filename}', 'rb') as f:
                    concept_chain = pickle.load(f)
            else:
                concept_chain = None
            
            with open(f"{options.experiment_directory}{os.sep}{name}_info.txt", "w") as f:
                f.write(json.dumps(options.__dict__, default=lambda o: '<not serializable>'))
                f.write(f"\n Git Commit: {subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()}")
            
            try:
                data = arff.loadarff(datastream_filename)
                df = pd.DataFrame(data[0])
            except Exception as e:
                print(e)
                print("trying csv")
                df = pd.read_csv(datastream_filename, header=None)

            for c_i,c in enumerate(df.columns):
                
                if pd.api.types.is_string_dtype(df[c]):
                    print(f"Factoizing {c}")
                    print(pd.factorize(df[c])[0].shape)
                    df[c] = pd.factorize(df[c])[0]
                
                # print(f"{c_i}: {len(df.columns) - 1}")
                # if c_i == len(df.columns) - 1:
                #     print(f"converting {c}")
                #     df[c] = df[c].astype('category')
                
            
            print(df.info())

            datastream = DataStream(df)
            datastream.concept_chain = concept_chain
            print(concept_chain)
            datastream.prepare_for_use()
            t_start = process_time()
            print(options.__dict__)
            classifier = FSMClassifier(
                                        concept_limit = options.concept_limit, 
                                        memory_management = options.memory_management, 
                                        learner = options.learner, 
                                        window = options.window,
                                        sensitivity = options.sensitivity,
                                        concept_chain= concept_chain,
                                        optimal_selection = options.optimal_selection,
                                        optimal_drift = options.optimal_drift,
                                        rand_weights= options.rand_weights,
                                        poisson= options.poisson,
                                        similarity_measure = options.similarity_measure,
                                        merge_strategy= options.merge_strategy,
                                        use_clean= options.use_clean)
            # fsm, system_stats, concept_chain, ds, stream_examples =  fsmsys.run_fsm(datastream, options, suppress = True, name = name, save_checkpoint=True, concept_chain= concept_chain, optimal_selection= options.optimal_selection)
            avg_memory, max_memory = evaluate_prequential.evaluate_prequential(datastream, classifier, directory=options.experiment_directory, name = name, noise = options.noise, seed = options.seed)
            t_stop = process_time()
            print("")
            print("Elapsed time during the whole program in seconds:", 
                                                t_stop-t_start)
            with open(f"{options.experiment_directory}{os.sep}{name}_timer.txt", "w") as f:
                f.write(f"Elapsed time during the whole program in seconds: {t_stop-t_start}")
            with open(f"{options.experiment_directory}{os.sep}{name}_memory.txt", "w") as f:
                f.write(f"Average: {avg_memory}\n")
                f.write(f"Max: {max_memory}")
            options.memory_management = save_mm
        options.memory_management = save_mm

def subdir_run(options, base_directory):
    list_of_directories = []
    for (dirpath, dirnames, filenames) in os.walk(base_directory):
        for filename in filenames:
            if filename.endswith('.ARFF') or filename.endswith('.arff'): 
                list_of_directories.append(dirpath)
    list_of_directories.sort()
    for subdir in list_of_directories:
        print(subdir)
        options.experiment_directory = subdir
        sensitivities = [options.sensitivity]
        windows = [options.window]
        if options.multiple_sensitivities:
            sensitivities = [1e-2, 1e-3, 1e-4, 1e-5]
        if options.multiple_windows:
            windows = [30, 50, 80, 100, 150]
        for s in sensitivities:
            for w in windows:
                options.sensitivity = s
                options.window = w
                start_run(options)
            #start_run(options)
def get_ARF_HAT():
    max_features=3
    disable_weighted_vote=False
    lambda_value=6
    performance_metric='acc'
    drift_detection_method = ADWIN(0.001)
    warning_detection_method = ADWIN(0.01)
    max_byte_size=33554432
    memory_estimate_period=2000000
    grace_period=50
    split_criterion='info_gain'
    split_confidence=0.01
    tie_threshold=0.05
    binary_split=False
    stop_mem_management=False
    remove_poor_atts=False
    no_preprune=False
    leaf_prediction='nba'
    nb_threshold=0
    nominal_attributes=None
    random_state=None

    classifier = TS_ARFHoeffdingTree(max_byte_size= max_byte_size,
                                    memory_estimate_period= memory_estimate_period,
                                                                        grace_period= grace_period,
                                                                        split_criterion= split_criterion,
                                                                        split_confidence= split_confidence,
                                                                        tie_threshold= tie_threshold,
                                                                        binary_split= binary_split,
                                                                        stop_mem_management= stop_mem_management,
                                                                        remove_poor_atts= remove_poor_atts,
                                                                        no_preprune= no_preprune,
                                                                        leaf_prediction= leaf_prediction,
                                                                        nb_threshold= nb_threshold,
                                                                        nominal_attributes= nominal_attributes,
                                                                        max_features= max_features,
                                                                        random_state= random_state)
    return classifier


if __name__ == "__main__":
    # Set config params, get commandline params
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--seed", type=int,
        help="Random seed", default=None)
    ap.add_argument("-n", "--noise", type=float,
        help="Noise", default=0)
    ap.add_argument("-cl", "--conceptlimit", type=int,
        help="Concept limit", default=-1)
    ap.add_argument("-clr", "--conceptlimitrange", type=int,
        help="Concept limit", default=-1)
    ap.add_argument("-cls", "--conceptlimitstart", type=int,
        help="Concept limit", default=1)
    ap.add_argument("-p", "--poisson", type=int,
        help="Bagging poisson", default=10)
    ap.add_argument("-ds", "--sensitivity", type=float,
        help="Bagging poisson", default=0.05)
    ap.add_argument("-pr", "--poissonreps", type=int,
        help="Repititions for positive poisson levels", default=1)
    ap.add_argument("-ms", "--maxsize", type=int,
        help="Repititions for positive poisson levels", default=33554432)
    ap.add_argument("-d", "--directory",
        help="tdata generator for stream", default="datastreams")
    ap.add_argument("-al", "--adaptivelearner",
        help="learner", default="HN", choices=["HN", "NBN", "ARFN", "HATN", "ARF_HATN", "HC"])
    ap.add_argument("-mg", "--mergestrat",
        help="learner", default="sur", choices=["sur", "both", "acc"])
    ap.add_argument("-sm", "--similarity",
        help="learner", default="KT", choices=["ACC", "KNN", "QT", 'PREVACC', 'EVOACC', 'AD', 'KT'])
    ap.add_argument("-mm", "--memmanagement",
        help="learner", default="rA", choices=["rA", "age", "LRU", 'acc', 'auc', 'score', 'rAAuc', 'all', 'div', 'mine'])
    # ap.add_argument("-ms", "--msense", action="store_true",
    #     help="multiple sensitivities")
    # ap.add_argument("-mw", "--mwind", action="store_true",
    #     help="multiple windows")
    ap.add_argument("-os", "--optsel", action="store_true",
        help="optimal selection")
    ap.add_argument("-od", "--optdrift", action="store_true",
        help="optimal drifting")
    ap.add_argument("-uc", "--useclean", action="store_true",
        help="use clean fsm class")
    ap.add_argument("-nc", "--nochain", action="store_true",
        help="Don't look for a concept chain")
    args = vars(ap.parse_args())
    options = fsmsys.FSMOptions()
    options.max_size = args['maxsize']
    print(f"Max byte size: {options.max_size}")
    options.learner_str = args['adaptivelearner']
    options.learner = lambda: TS_HoeffdingTree(max_byte_size = options.max_size, memory_estimate_period = 1000)
    if args['adaptivelearner'] == 'NBN':
        options.learner = lambda: NaiveBayes()
    if args['adaptivelearner'] == 'ARFN':
        options.learner = lambda: AdaptiveRandomForest(n_estimators= 5)
    if args['adaptivelearner'] == 'HATN':
        options.learner = lambda: TS_HAT()
    if args['adaptivelearner'] == 'ARF_HATN':
        options.learner = lambda: get_ARF_HAT()
    if args['adaptivelearner'] == 'HC':
        w = args['poisson']
        options.learner = lambda: TS_HoeffdingTree(max_byte_size = options.max_size, memory_estimate_period = 1000, split_confidence= pow(0.0000001,  1 / w), grace_period= int(200 / w))

    
    options.similarity_measure = args['similarity']
    options.merge_strategy = args['mergestrat']
    options.noise = args['noise']
    options.concept_limit = args['conceptlimit']
    options.memory_management = args['memmanagement']
    # options.multiple_sensitivities = args['msense']
    options.multiple_sensitivities = False
    # options.multiple_windows = args['mwind'] 
    options.multiple_windows = False
    options.optimal_selection = args['optsel']
    options.optimal_drift = args['optdrift']
    options.poisson = args['poisson']
    options.rand_weights = options.poisson > 1 and args['adaptivelearner'] != 'HC'
    options.no_chain = args['nochain']
    options.poisson_reps = args['poissonreps']
    options.sensitivity = args['sensitivity']
    options.use_clean = args['useclean']
    
    # options.experiment_directory = args['directory']
    base_directory = args['directory']
    options.batch_size = 1
    seed = args['seed']


    run_all_mem = True if args['memmanagement'] == 'all' else False
    run_mine_mem = True if args['memmanagement'] == 'mine' else False
    if args['conceptlimitrange'] > 0:
        for cl in range(max(args['conceptlimitstart'], 1), args['conceptlimitrange'], max(args['conceptlimit'], 1)):
            options.concept_limit = cl
            if run_all_mem:
                options.memory_management = 'all'
            if run_mine_mem:
                options.memory_management = 'mine'
            print(options.poisson)
            print(options.poisson_reps)
            num_reps = 1 if options.poisson < 1 or options.poisson_reps < 1 else options.poisson_reps
            print(num_reps)
            for pr in range(num_reps):
                if seed == None:
                    seed = random.randint(0, 10000)
                options.seed = seed
                options.concept_limit = cl
                if run_all_mem:
                    options.memory_management = 'all'
                if run_mine_mem:
                    options.memory_management = 'mine'
                subdir_run(options, base_directory)
                seed = None
    else:
        print(options.poisson)
        print(options.poisson_reps)
        num_reps = 1 if options.poisson < 1 or options.poisson_reps < 1 else options.poisson_reps
        print(num_reps)
        for pr in range(num_reps):
            if seed == None:
                seed = random.randint(0, 10000)
            options.seed = seed
            if run_all_mem:
                options.memory_management = 'all'
            if run_mine_mem:
                options.memory_management = 'mine'
            subdir_run(options, base_directory)
            seed = None


    # run_fsm(datastream, options, suppress = False, display = True, save_stream = False,
    #     fsm = None, system_stats=None, detector = None, stream_examples = None):

