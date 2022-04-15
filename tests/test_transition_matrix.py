import warnings
from PhDCode.Classifier.select_classifier import SELeCTClassifier
from PhDCode.Classifier.hoeffding_tree_shap import HoeffdingTreeSHAPClassifier
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.sine_generator import SineGenerator


def test_transition_matrix():
    learner = HoeffdingTreeSHAPClassifier
    cd_classifier = SELeCTClassifier(learner=learner, ignore_sources=["features", "labels", "predictions", "error_distances"], ignore_features=["IMF", "MI", "FI", "kurtosis", "skew"], window_size=75, similarity_gap=10, fingerprint_update_gap=10, non_active_fingerprint_update_gap=10, observation_gap=10, feature_selection_method="fisher_overall")

    stream = SineGenerator(classification_function=0, random_state=1)
    X,y = stream.next_sample()
    cd_classifier.partial_fit(X, y, classes = [0, 1])

    # Test result is the same before any drift, then 
    # the same as a fresh classifier after drift.
    counter = 0
    n_since_drift = -10000
    for i in range(24000):
        if i % 4000 == 0 and i != 0:
            # stream = STAGGERGenerator(classification_function=0 if counter in [0, 2] else 1, random_state=1)
            stream = SineGenerator(classification_function=counter, random_state=1)
            counter += 1
            counter %= 3
            n_since_drift = 0

        # cd_classifier.manual_control = False
        # cd_classifier.force_transition = False
        # if n_since_drift == 75:
        #     cd_classifier.manual_control = True
        #     cd_classifier.force_transition = True
        
        n_since_drift += 1
        X,y = stream.next_sample()


        p = cd_classifier.predict(X)
        cd_classifier.partial_fit(X, y, classes = [0, 1])
    
    print(cd_classifier.concept_transitions_standard)
    assert str(cd_classifier.concept_transitions_standard) == str({0: {'total': 11840, 0: 11837, 1: 1, 2: 2}, 1: {'total': 99, 1: 98, 0: 1}, 2: {'total': 7966, 2: 7965, 3: 1}, 3: {'total': 4096, 3: 4095, 0: 1}})
    print(cd_classifier.concept_transitions_warning)
    assert str(cd_classifier.concept_transitions_warning) == str({0: {'total': 3269, 0: 3266, 1: 1, 2: 2}, 1: {'total': 0}, 2: {'total': 802, 2: 801, 3: 1}, 3: {'total': 69, 3: 68, 0: 1}})
    print(cd_classifier.concept_transitions_drift)
    assert str(cd_classifier.concept_transitions_drift) == str({0: {'total': 5, 0: 2, 1: 1, 2: 2}, 1: {'total': 0}, 2: {'total': 3, 2: 2, 3: 1}, 3: {'total': 5, 3: 4, 0: 1}})
    print(cd_classifier.get_transition_probabilities_smoothed(0, 3))
    assert str(cd_classifier.get_transition_probabilities_smoothed(0, 3)) == str((8.443093549476529e-05, 0.00030553009471432935, 0.1111111111111111))
    cd_classifier.in_warning = False
    print(cd_classifier.get_transition_prior(0, 3, False))
    assert cd_classifier.get_transition_prior(0, 3, False) == 8.443093549476529e-05
    cd_classifier.in_warning = True
    print(cd_classifier.get_transition_prior(0, 3, False))
    assert cd_classifier.get_transition_prior(0, 3, False) == 0.00030553009471432935
    print(cd_classifier.get_transition_prior(0, 3, True))
    assert cd_classifier.get_transition_prior(0, 3, True) == 0.1111111111111111
    cd_classifier.in_warning = False
    assert cd_classifier.get_transition_prior(0, 3, True) == 0.1111111111111111
    print(cd_classifier.get_transition_probabilities_smoothed(0, 2))
    assert str(cd_classifier.get_transition_probabilities_smoothed(0, 2)) == str((0.00025329280648429586, 0.0009165902841429881, 0.3333333333333333))
    cd_classifier.in_warning = False
    print(cd_classifier.get_transition_prior(0, 2, False))
    assert cd_classifier.get_transition_prior(0, 2, False) == 0.00025329280648429586
    cd_classifier.in_warning = True
    print(cd_classifier.get_transition_prior(0, 2, False))
    assert cd_classifier.get_transition_prior(0, 2, False) == 0.0009165902841429881
    print(cd_classifier.get_transition_prior(0, 2, True))
    assert cd_classifier.get_transition_prior(0, 2, True) == 0.3333333333333333
    cd_classifier.in_warning = False
    assert cd_classifier.get_transition_prior(0, 2, True) == 0.3333333333333333

def test_multihop_prior():
    learner = HoeffdingTreeSHAPClassifier
    cd_classifier = SELeCTClassifier(learner=learner, ignore_sources=["features", "labels", "predictions", "error_distances"], ignore_features=["IMF", "MI", "FI", "kurtosis", "skew"], window_size=75, similarity_gap=10, fingerprint_update_gap=10, non_active_fingerprint_update_gap=10, observation_gap=10, feature_selection_method="fisher_overall")

    stream = SineGenerator(classification_function=0, random_state=1)
    X,y = stream.next_sample()
    cd_classifier.partial_fit(X, y, classes = [0, 1])

    # Test result is the same before any drift, then 
    # the same as a fresh classifier after drift.
    counter = 0
    n_since_drift = -10000
    for i in range(24000):
        if i % 4000 == 0 and i != 0:
            # stream = STAGGERGenerator(classification_function=0 if counter in [0, 2] else 1, random_state=1)
            stream = SineGenerator(classification_function=counter, random_state=1)
            counter += 1
            counter %= 3
            n_since_drift = 0

        # cd_classifier.manual_control = False
        # cd_classifier.force_transition = False
        # if n_since_drift == 75:
        #     cd_classifier.manual_control = True
        #     cd_classifier.force_transition = True
        
        n_since_drift += 1
        X,y = stream.next_sample()


        p = cd_classifier.predict(X)
        cd_classifier.partial_fit(X, y, classes = [0, 1])
    
    print(cd_classifier.concept_transitions_standard)
    assert str(cd_classifier.concept_transitions_standard) == str({0: {'total': 11840, 0: 11837, 1: 1, 2: 2}, 1: {'total': 99, 1: 98, 0: 1}, 2: {'total': 7966, 2: 7965, 3: 1}, 3: {'total': 4096, 3: 4095, 0: 1}})
    print(cd_classifier.concept_transitions_warning)
    assert str(cd_classifier.concept_transitions_warning) == str({0: {'total': 3269, 0: 3266, 1: 1, 2: 2}, 1: {'total': 0}, 2: {'total': 802, 2: 801, 3: 1}, 3: {'total': 69, 3: 68, 0: 1}})
    print(cd_classifier.concept_transitions_drift)
    assert str(cd_classifier.concept_transitions_drift) == str({0: {'total': 5, 0: 2, 1: 1, 2: 2}, 1: {'total': 0}, 2: {'total': 3, 2: 2, 3: 1}, 3: {'total': 5, 3: 4, 0: 1}})
    print(cd_classifier.get_transition_probabilities_smoothed(0, 3))
    assert str(cd_classifier.get_transition_probabilities_smoothed(0, 3)) == str((8.443093549476529e-05, 0.00030553009471432935, 0.1111111111111111))
    
    # Test that both single hop methods give the same result
    assert cd_classifier.get_transition_probabilities_smoothed(0, 3) == cd_classifier.get_multihop_transition_probs(0, 3, smoothing_factor=1, n_hops=1)

    calculated_2hop = sum([
        cd_classifier.get_multihop_transition_probs(0, 0, smoothing_factor=1, n_hops=1)[0]*cd_classifier.get_multihop_transition_probs(0, 3, smoothing_factor=1, n_hops=1)[0],
        cd_classifier.get_multihop_transition_probs(0, 1, smoothing_factor=1, n_hops=1)[0]*cd_classifier.get_multihop_transition_probs(1, 3, smoothing_factor=1, n_hops=1)[0],
        cd_classifier.get_multihop_transition_probs(0, 2, smoothing_factor=1, n_hops=1)[0]*cd_classifier.get_multihop_transition_probs(2, 3, smoothing_factor=1, n_hops=1)[0],
        cd_classifier.get_multihop_transition_probs(0, 3, smoothing_factor=1, n_hops=1)[0]*cd_classifier.get_multihop_transition_probs(3, 3, smoothing_factor=1, n_hops=1)[0],
    ])

    returned_2hop = cd_classifier.get_multihop_transition_probs(0, 3, smoothing_factor=1, n_hops=2)[0]
    assert calculated_2hop == returned_2hop
    print(cd_classifier.get_multihop_transition_probs(0, 3, smoothing_factor=1, n_hops=3))
# test_transition_matrix()
test_multihop_prior()