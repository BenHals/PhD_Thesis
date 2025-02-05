import math
import numpy as np


def fisher_score(group_values, group_counts, overall_stdev, group_stdev=None):
    """ Calculate the fisher score
    as the between group variance divided by 
    overall variance.

    Parameters
    ----------
    group_values: np.array
        - The ordered means of each group

    group_counts: np.array
        - The ordered counts of each group

    overall_stdev: Float
        - The standard deviation over all groups

    group_stdev: np.array, optional
        - If set, uses the average intra-group stdev to compare
        against instead of the overall_stdev.
        Fingerprints often have very small intra-group stdev,
        and while this is used in some fisher score implementations
        it causes unstable results here, with some features
        showing very high relative weights and dominating.
    """
    group_proportions = group_counts / group_counts.sum()
    overall_mean = np.dot(group_values, group_proportions)
    group_deviance = group_values - overall_mean
    squared_deviance = np.square(group_deviance)
    sum_concept_deviations = np.dot(squared_deviance, group_proportions)
    # sum_concept_deviations = np.average(squared_deviance, weights=group_proportions)
    # sum_concept_deviations = 0
    # for concept_i, concept_mu in enumerate(group_values):
    #     sum_concept_deviations += group_proportions[concept_i] * \
    #         math.pow((concept_mu - overall_mean), 2)
    sum_concept_variance = 0
    if group_stdev is not None:
        sum_concept_variance = 0
        for concept_i, concept_sigma in enumerate(group_stdev):
            sum_concept_variance += group_proportions[concept_i] * \
                math.pow((concept_sigma), 2)
    if sum_concept_variance > 0:
        fisher_score = sum_concept_deviations / sum_concept_variance
    else:
        fisher_score = sum_concept_deviations / (math.pow(overall_stdev, 2))
    return fisher_score
