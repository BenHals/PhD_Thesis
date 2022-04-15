import copy
import statistics
import numpy as np
from scipy.stats import ttest_1samp
import PhDCode.Classifier.advantage_fsm_o.libquanttree as qt
from scipy import stats as st
import sys
from PhDCode.Classifier.advantage_fsm_o.modelStats import BufferItem
import pickle

def getDistance(e, t):
    axis_distance = []
    for i,feature in enumerate(e.X):
        compare_feature = t.X[i]
        try:
            ff = float(feature)
            fcf = float(compare_feature)

            # If they both are numbers
            axis_distance.append(ff - fcf)
        except ValueError:
            print("FEATURES NEED TO BE NUMBERS! Use ints for categorical.")
    label = e.y
    compare_label = t.y
    try:
        ff = float(label)
        fcf = float(compare_label)

        # If they both are numbers
        axis_distance.append((ff - fcf))
    except ValueError:
        print("FEATURES NEED TO BE NUMBERS! Use ints for categorical.")
    return np.sqrt(np.sum(np.power(axis_distance, 2)))



def knnTest(buffer_test, buffer_shadow):
    """Computes the statistical significance of the hypothesis both buffers came from the same distribution.

    This test is based on the idea that a recurring concept will produce examples 
    drawn from a similar distribution. This may not be true if noise distorts the
    beginning of one of the buffers.
    Both buffers should be the same length.

    Parameters
    ----------
    buffer_test: list<BufferItem>
        The first items seen by the concept being tested.
    
    buffer_shadow: list<BufferItem>
        The first items after a concept change.
    """
    combined_buffer = []
    k = 13
    for example in buffer_test:
        labeled_item = copy.copy(example)
        labeled_item.instance = 0
        combined_buffer.append(labeled_item)
    for example in buffer_shadow:
        labeled_item = copy.copy(example)
        labeled_item.instance = 1
        combined_buffer.append(labeled_item)
    
    distance_matrix = {}
    num_similar_neighbor_list = []

    # For each example, get the k nearest neighbors.
    for example_index, example in enumerate(combined_buffer):
        distance_list = []

        # Look at the distance to each item.
        for test_index, test_example in enumerate(combined_buffer):

            # Don't need to get distance to ourselves.
            if test_index == example_index:
                continue
            
            # Distance is symetric, So we look to see if the distance has been calculated.
            lookup_key = (min(example_index, test_index), max(example_index, test_index))
            if lookup_key in distance_matrix:
                distance_list.append((distance_matrix[lookup_key], test_index))
            else:
                distance = getDistance(example, test_example)
                # print(distance)
                distance_list.append((distance, test_index))
                distance_matrix[lookup_key] = distance
        
        distance_list.sort(key = lambda x: x[0])
        
        k_neighbors = list(map(lambda t: 1 if combined_buffer[t[1]].instance == example.instance else -1, distance_list[:k]))
        
        num_similar_neighbor_list.append(sum(k_neighbors) - 1)
    # print(num_similar_neighbor_list)
    return ttest_1samp(num_similar_neighbor_list, 0)

PRECOMPUTED_QUANTTREE_THRESH = None
def quantTreeTest(buffer_test, buffer_shadow):
    global PRECOMPUTED_QUANTTREE_THRESH
    num_bins = 8
    alpha = 0.05
    K = num_bins
    nu = len(buffer_shadow)
    N = len(buffer_test)



    train_data = np.array(list(map(BufferItem.getExample, buffer_test)))
    test_data = np.array(list(map(BufferItem.getExample, buffer_shadow)))
    qtree = qt.QuantTree(num_bins)
    qtree.build_histogram(train_data, True)

    test = qt.ChangeDetectionTest(qtree, nu, qt.pearson_statistic)
    if PRECOMPUTED_QUANTTREE_THRESH == None:
        try:
            with open('PRECOMPUTED_QUANTTREE_THRESH.pickle', 'rb') as f:
                PRECOMPUTED_QUANTTREE_THRESH = pickle.load(f)
        except:
            PRECOMPUTED_QUANTTREE_THRESH = {}
    # threshold = test.get_precomputed_quanttree_threshold('pearson', alpha, K, N, nu)
    threshold = PRECOMPUTED_QUANTTREE_THRESH.get(('pearson', alpha, K, N, nu))
    if threshold is None:
        print("estimating thresh")
        threshold = test.estimate_quanttree_threshold('pearson', alpha, K, N, nu, 10000)
        PRECOMPUTED_QUANTTREE_THRESH[('pearson', alpha, K, N, nu)] = threshold
        with open('PRECOMPUTED_QUANTTREE_THRESH.pickle', 'wb') as f:
            pickle.dump(PRECOMPUTED_QUANTTREE_THRESH, f)
    test.set_threshold(alpha, threshold)

    hp, _ = test.reject_null_hypothesis(test_data, alpha)

    return (hp, _)






