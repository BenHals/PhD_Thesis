#%%
import pandas as pd
from PhDCode.utils.eval_utils import (
    make_stream,
    get_unique_stream_names,
    get_ground_truth_concept_idx,
)
import pathlib


options = {
    'concept_max': 6,
    'data_type': 'Real',
    'conceptdifficulty': 0,
    'data_name': 'cmc',
    'seed': 1,
    'raw_data_path': pathlib.Path(__file__).absolute().parents[1] / 'RawData',
    'max_rows': 75000,
    'concept_length': 5000,
    'repeats': 3,
    'repeatproportion': 1.0,
    'shuffleconcepts': False,
    'TMdropoff': 1.0,
    'TMforward': 1,
    'TMnoise': 0.0,
    'drift_width': 0,
    'GT_context_location': None,
}
stream, stream_concepts, length, classes = make_stream(options)

#%%
stream_names = [c[3] for c in stream_concepts]
unique_stream_names = get_unique_stream_names(stream_concepts)
gt_context_values = None
rows = []
contexts = []
for i in range(length):
    X, y = stream.next_sample()
    gt = get_ground_truth_concept_idx(i, stream_concepts, unique_stream_names, gt_context_values)
    rows.append([*X[0], y[0]])
    contexts.append(gt)
#%%

data_df = pd.DataFrame(rows)
contexts_df = pd.DataFrame(contexts)
data_df.head()
contexts_df.head()
# %%

data_df.to_csv(f"{options['data_name']}_0.csv")
contexts_df.to_csv('context.csv', index=False)