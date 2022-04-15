from collections import deque
import numpy as np
import PhDCode.Classifier.advantage_fsm_o.config as config

class BufferItem:
    """Holds an item representing one datastream example.

    Parameters
    ----------
    X: List
        A list representing a feature vector.
    
    y: int
        A class representing the real label of the feature.
    
    p: int
        The predicted label of the feature.
    """
    def __init__(self, X, y, p):
        self.X = X
        self.y = y
        self.p = p
        self.instance = None
    
    def __str__(self):
        return f"X: {self.X}, y: {self.y}, p: {self.p}"
    
    def getExample(self):
        return np.append(self.X, self.y)

class modelStats:
    """Holds the statistics for a model
    
    Parameters
    ----------
    id: int
        The ID for the owner state.
    
    type: string
        The type of the owning model.
    """
    def __init__(self, id, model_type):
        self.id = id                            # The ID of the state.
        self.type = model_type                  # The type of the controlling model.
        self.sliding_window_accuracy_log = []   # The accuracy of the state on the sliding window.
        self.sliding_window = deque()           # A sliding window of the last predictions.
        self.first_seen_examples = []           # A list of the first seen examples for the state.
        self.right = 0                          # The number of right predictions
        self.wrong = 0                          # The number of wrong predictions
        self.correct_log = []
        self.p_log = []
        self.y_log = []
    
    def add_prediction(self, X, y, p, is_correct, ts):
        example = BufferItem(X, y, p)
        if(len(self.first_seen_examples) < config.example_window_length):
            self.first_seen_examples.append(example)
            
        # Sliding window tracks recent accuracy
        self.sliding_window.append(1 if is_correct else 0)
        if(len(self.sliding_window) > config.report_window_length):
            self.sliding_window.popleft()

        self.sliding_window_accuracy_log.append((ts, sum(self.sliding_window) / len(self.sliding_window)))
        self.correct_log.append((ts, 1 if is_correct else 0))
        self.p_log.append((ts, p))
        self.y_log.append((ts, y))
        if is_correct:
            self.right += 1
        else:
            self.wrong += 1
