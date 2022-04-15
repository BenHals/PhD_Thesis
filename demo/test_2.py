
#%%
from PhDCode.Classifier.advantage_classifier import FSMClassifier
# from PhDCode.Classifier.advantage_fsm_o.tracksplit_hoeffding_tree import TS_HoeffdingTree
from PhDCode.Classifier.advantage_fsm.evaluate_prequential import evaluate_prequential
# from PhDCode.Classifier.hoeffding_tree_evolution import HoeffdingTreeEvoClassifier as TS_HoeffdingTree
# from PhDCode.Classifier.advantage_fsm_o.ts_hoeffding_new_skm import TS_HoeffdingTree as TS_HoeffdingTree
from PhDCode.Classifier.hoeffding_tree_evolution import HoeffdingTreeEvoClassifier as TS_HoeffdingTree


from skmultiflow.data.data_stream import DataStream

from scipy.io import arff
import pandas as pd
import pathlib
import json
import pickle
from collections import Counter
import numpy as np
import tqdm

class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        elif isinstance(obj, pathlib.Path):
            return str(obj)

        return json.JSONEncoder.default(self, obj)



#%%
datastream_filename = pathlib.Path(r"S:\PhD\testing\MemManagementTest\2199.ARFF")
info_filename = pathlib.Path(r"S:\PhD\testing\MemManagementTest\2199_dsinfo.txt")
cc_filename = pathlib.Path(r"S:\PhD\testing\MemManagementTest\2199_concept_chain.pickle")
dsinfo_str = info_filename.open('r').readlines()
print(dsinfo_str[0][:-1])
dsinfo = json.loads(dsinfo_str[0][:-1])
concept_chain = dsinfo['concept_chain']
concept_chain_2 = pickle.load(cc_filename.open('rb'))
print(concept_chain_2)

def get_gt_id(i, cc):
    found_val = 0
    for start, c_id in cc.items():
        if start <= i:
            found_val = c_id
        else:
            break
    return found_val

#%%
print(get_gt_id(0, concept_chain_2))
print(get_gt_id(1420100, concept_chain_2))
print(get_gt_id(1426000, concept_chain_2))

#%%
data = arff.loadarff(datastream_filename)
df = pd.DataFrame(data[0])
print(df.head())
print(df.info())
nominal_attributes = []
for c_i,c in enumerate(df.columns):
    init_vals = df[c].values[:20]
    n_u = len(np.unique(init_vals))
    if n_u < 10:
        # print(f"Factoizing {c}")
        # print(pd.factorize(df[c])[0].shape)
        # df[c] = pd.factorize(df[c])[0]
        nominal_attributes.append(c_i)
print(df.head())
print(df.info())
# try:
#     data = arff.loadarff(datastream_filename)
#     df = pd.DataFrame(data[0])
# except Exception as e:
#     print(e)
#     print("trying csv")
#     df = pd.read_csv(datastream_filename, header=None)

# for c_i,c in enumerate(df.columns):
    
#     if pd.api.types.is_string_dtype(df[c]):
#         print(f"Factoizing {c}")
#         print(pd.factorize(df[c])[0].shape)
#         df[c] = pd.factorize(df[c])[0]
    
    # print(f"{c_i}: {len(df.columns) - 1}")
    # if c_i == len(df.columns) - 1:
    #     print(f"converting {c}")
    #     df[c] = df[c].astype('category')
    

print(df.info())

datastream = DataStream(df)
datastream.concept_chain = concept_chain
print(concept_chain)
datastream.prepare_for_use()
#%%
# df.head()
# stream = DataStream(df)
# stream.concept_chain = concept_chain_2
# stream.prepare_for_use()

# learner = HoeffdingTreeEvoClassifier
learner = lambda : TS_HoeffdingTree(max_byte_size = 33554432, memory_estimate_period = 1000, nominal_attributes=nominal_attributes)

classifier = FSMClassifier(
    concept_limit=35,
    memory_management="rA",
    learner=learner,
    window=175,
    sensitivity=0.05,
    concept_chain=concept_chain_2,
    use_clean=True,
    merge_strategy="sur",
    poisson=10,
)

# classifier = FSMClassifier(
#                             concept_limit = 35, 
#                             memory_management = "rA", 
#                             learner = learner, 
#                             window = 175,
#                             sensitivity = 0.05,
#                             concept_chain= concept_chain_2,
#                             optimal_selection = False,
#                             optimal_drift = False,
#                             rand_weights= True,
#                             poisson= 10,
#                             similarity_measure = 'KT',
#                             merge_strategy= 'sur',
#                             use_clean= True)

# print(classifier_o.__dict__)
# print(classifier.__dict__)

# for k,v in classifier_o.__dict__.items():
#     print(f"{k}: {v == classifier.__dict__[k]}")

#%%
output_path = pathlib.Path(r"S:\PhD\testing\MemManagementTest")
evaluate_prequential(datastream, classifier, directory=str(output_path), name = "PhDCode-test", noise = 0.0, seed = 8423)