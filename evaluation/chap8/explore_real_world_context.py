#%%
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from collections import Counter
import json

from sklearn import tree
from sklearn.model_selection import train_test_split

class BasicNB:
    """
    Basic Naive Bayes classifier for 1D X and y.
    If an X value is seen which is not in training set, mark as a failed prediction.
    """
    def __init__(self):
        self.p_X = {}
        self.p_y = {}
        self.p_Xy = {}
    
    def get_key(self, x_val, y_val):
        return str(int(x_val)) +':' + str(int(y_val))

    def fit(self, X_train, y_train):
        count_X = Counter()
        count_y = Counter()
        count_Xy= Counter()

        for x_val, y_val in zip(X_train, y_train):
            count_X[x_val] += 1
            count_y[y_val] += 1
            count_Xy[self.get_key(x_val, y_val)] += 1
        
        total_count = X_train.shape[0]

        for k, v in count_X.items():
            self.p_X[k] = v / total_count
        for k, v in count_y.items():
            self.p_y[k] = v / total_count
        for k, v in count_Xy.items():
            x_val, y_val = map(lambda x: int(float(x)), k.split(':'))
            total_count_for_y = count_y[y_val]
            self.p_Xy[k] = v / total_count_for_y
        
    def predict(self, X_test):
        predictions = []
        for x_val in X_test:
            class_predictions = {}
            for y_val in self.p_y:
                key = self.get_key(x_val, y_val)
                if key not in self.p_Xy:
                    class_predictions[y_val] = -1
                    continue
                class_predictions[y_val] = self.p_Xy[key] * self.p_y[y_val]
            
            prediction, prob = max(class_predictions.items(), key = lambda x: x[1])
            if prob == -1:
                prediction = -1
            predictions.append(prediction)
        return predictions
    
    def get_accuracy(self, predictions, y_test):
        right = 0
        wrong = 0
        for p, y in zip(predictions, y_test):
            right += p == y
            wrong += p != y
        
        return right / (right + wrong)
sns.set_context('talk')

base_path = pathlib.Path(__file__).absolute().parents[0] / 'datasets'
print(pathlib.Path(__file__))

# data_name = "covtype"
data_name = "poker-lsn"

dataset_path = base_path / f"{data_name}.csv"
dataset_info_path = base_path / f"{data_name}_info.json"

#%%

column_info = json.load(dataset_info_path.open('r'))
# Some columns names can be associated with multiple columns (i.e., one hot encoded.)
# We first sort these out so each column gets a unique name
column_name_index = []
feature_column_map = {}
for cn, cis in column_info.items():
    feature_column_map[cn] = []
    for j, ci in enumerate(cis):
        column_name = cn if len(cis) == 1 else f"{cn}_{j}"
        column_name_index.append((column_name, ci))
        feature_column_map[cn].append((column_name, ci))
column_name_index.sort(key= lambda x: x[1])
column_names = [x[0] for x in column_name_index]
full_df = pd.read_csv(str(dataset_path), header=None, names=column_names)
#%%
#covtype
# context_feature = "Wilderness_Area"
# context_feature = "Soil_Type"
# context_feature = "Elevation"
# context_feature = "Slope"
# context_features = ["Elevation", "Slope"]
# context_features = ["Elevation"]
# context_features = ["Slope"]
# context_features = ["Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology"]
# context_features = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
#poker
context_features = ["Suit_5", "Rank_5"]
context_columns = [x[0] for context_feature in context_features for x in feature_column_map[context_feature]]
print(context_columns)
c_col = full_df[context_columns[0]]
if len(c_col.unique()) > 5:
    c_col = pd.qcut(c_col, 5, labels=False)
    print(c_col)
context_column = c_col.astype(str)
for col in context_columns[1:]:
    c_col = full_df[col]
    if len(c_col.unique()) > 5:
        c_col = pd.qcut(c_col, 5, labels=False, duplicates='drop')
    context_column = context_column + '-' + c_col.astype(str)
print(context_column)
context_df = full_df.copy()
context_df['context'] = context_column
context_df = context_df[[*context_df.columns[:-2], context_df.columns[-1], context_df.columns[-2]]]
context_df = context_df.drop(context_columns, axis=1)
context_df.head()

unique_contexts = list(context_column.unique())
context_map = {k:unique_contexts.index(k) for k in unique_contexts}
context_df['context'] = context_df['context'].replace(context_map)
context_df.head()

X = context_df.drop(context_df.columns[-1], axis=1).to_numpy()
y = context_df[context_df.columns[-1]].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
p = clf.predict(X_test)
print(sum(p == y_test) / p.shape[0])

X = context_df.drop(context_df.columns[-2:], axis=1).to_numpy()
y = context_df[context_df.columns[-1]].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
p = clf.predict(X_test)
print(sum(p == y_test) / p.shape[0])




# %%
no_context_df = context_df.drop(['context'], axis=1)
dataset_name = f"{data_name}-{'_'.join(context_features)}"
save_dir = base_path / dataset_name
save_dir.mkdir(parents=True, exist_ok=True)
total = 0
for c_name in unique_contexts:
    one_context_df = no_context_df[context_df['context'] == context_map[c_name]]
    print(one_context_df.head())
    save_path = save_dir / f"{dataset_name}_{c_name}.csv"
    one_context_df.to_csv(save_path)
    total += one_context_df.shape[0]

print(total)


#%%
full_df.head()

#%%
s = full_df.sample(100)
sns.lineplot(data=s, x=s.index, y='Slope')