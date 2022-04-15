import copy
import itertools
import numpy as np
import logging
from operator import attrgetter, itemgetter
import gc, sys
import pathlib
import yaml

import torch

from PhDCode.Classifier.CNDPM.models import NdpmModel
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from PhDCode.Classifier.CNDPM.ndpm.expert import Expert

from skmultiflow.utils import get_dimensions, normalize_values_in_dict, \
    calculate_object_size
from skmultiflow.core import BaseSKMObject, ClassifierMixin

from skmultiflow.trees.split_criterion import GiniSplitCriterion
from skmultiflow.trees.split_criterion import InfoGainSplitCriterion
from skmultiflow.trees.split_criterion import HellingerDistanceCriterion

from skmultiflow.trees.attribute_test import NominalAttributeMultiwayTest

from skmultiflow.trees.nodes import Node
from skmultiflow.trees.nodes import ActiveLearningNode
from skmultiflow.trees.nodes import InactiveLearningNode
from skmultiflow.trees.nodes import LearningNode
from skmultiflow.trees.nodes import LearningNodeNB
from skmultiflow.trees.nodes import LearningNodeNBAdaptive
from skmultiflow.trees.nodes import SplitNode
from skmultiflow.trees.nodes import FoundNode

from skmultiflow.rules.base_rule import Rule

import warnings
import shap

def actualsize(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size

class EmptyShap:
    def __init__(self, n_features):
        self.n_features = n_features
    
    def shap_values(self, X, check_additivity, approximate):
        return [np.zeros((2, self.n_features))]

class CNDPMMLPClassifier(BaseSKMObject, ClassifierMixin):
    """ Base classifier for CNDPM.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

   
    """

    def __init__(self,
                 max_byte_size=33554432,
                 nominal_attributes=None,
                 train_weight_required_for_evolution=100,
                 batch_learning=False,
                 cndpm_use_large=False):
        super().__init__()
        self.max_byte_size = max_byte_size
        self.nominal_attributes = nominal_attributes

        self._train_weight_seen_by_model = 0.0
        self.classes = None
        self.shap_model = None
        self.evolution = 0
        self.train_weight_seen_since_evolution = 0
        self.train_weight_required_for_evolution = train_weight_required_for_evolution
        self.batch_learning = batch_learning
        if not self.batch_learning:
            print("WARNING: The CNDPM base classifier doesn't work so well without the batch learning option")
            input("Arr you sure you want to continue running?")
        self.stm_x = []
        self.stm_y = []

        self.cndpm_use_large = cndpm_use_large

        # self.config_path = pathlib.Path(__file__).parent / "configs" / "cndpm-ADL-small.yaml"
        # self.config_path = pathlib.Path(__file__).parent / "configs" / "cndpm-ADL-small-nosleep-batch.yaml"
        # self.config_path = pathlib.Path(__file__).parent / "configs" / "cndpm-ADL-small-nosleep.yaml"
        # self.config = yaml.load(open(self.config_path), Loader=yaml.FullLoader)
        if not self.cndpm_use_large:
            self.config_path = pathlib.Path(__file__).parent / "configs" / "cndpm-ADL-small-nosleep-batch.yaml"
        else:
            self.config_path = pathlib.Path(__file__).parent / "configs" / "cndpm-ADL-norm-nosleep-batch.yaml"
        with open(self.config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.writer = SummaryWriter(".\\experimentlog")
        self.device = self.config['device'] if 'device' in self.config else 'cuda'
        self.classifier = None
        self.experts = [self.classifier]
        self.batch_size = 1

    def make_shap_model(self, X):
        return EmptyShap(X.shape[0])

    @property
    def max_byte_size(self):
        return self._max_byte_size

    @max_byte_size.setter
    def max_byte_size(self, max_byte_size):
        self._max_byte_size = max_byte_size

    @property
    def nominal_attributes(self):
        return self._nominal_attributes

    @nominal_attributes.setter
    def nominal_attributes(self, nominal_attributes):
        self._nominal_attributes = nominal_attributes

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        self._classes = value

    def measure_byte_size(self):
        """ Calculate the size of the tree.

        Returns
        -------
        int
            Size of the tree in bytes.

        """
        return calculate_object_size(self)

    def reset(self):
        """ Reset the Hoeffding Tree to default values."""
        self._train_weight_seen_by_model = 0.0
        self.train_weight_seen_since_evolution = 0

        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Incrementally trains the model. Train samples (instances) are
        composed of X attributes and their corresponding targets y.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: numpy.array
            Contains the class values in the stream. If defined, will be used
            to define the length of the arrays returned by `predict_proba`
        sample_weight: float or array-like
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
            self

        Notes
        -----
        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for
          the instance and update the leaf node statistics.
        * If growth is allowed and the number of instances that the leaf has
          observed between split attempts exceed the grace period then attempt
          to split.

        """
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.
                                 format(row_cnt, len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    
                    self._partial_fit(X[i], y[i], sample_weight[i])
        if self.shap_model is None:
            self.shap_model = self.make_shap_model(X[0])
        return self

    def _partial_fit(self, X, y, sample_weight):
        """ Trains the model on samples X and corresponding targets y.

        Private function where actual training is carried on.

        Parameters
        ----------
        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.
        y: int
            Class label for sample X.
        sample_weight: float
            Sample weight.

        """
        if self.classifier is None:
            self.config['x_h'] = X.shape[0]
            self.config['y_c'] = len(self.classes)
            self.config['batch_size'] = self.batch_size
            base_expert = Expert(self.config)
            experts = (base_expert, )
            self.classifier = Expert(self.config, experts)
            self.classifier.eval()
            self.experts = [base_expert, self.classifier]

        y_idx = self.classes.index(y)
        x = torch.from_numpy(X).view(self.batch_size, 1, -1, 1).float()
        y = torch.from_numpy(np.array([y_idx])).to(torch.int64)

        if not self.batch_learning:
            x, y = x.to(self.device), y.to(self.device)
            nll, summaries = self.classifier.collect_nll(x, y)  # [B, 1+K]
            nl_joint = nll
            destination = torch.argmin(nl_joint, dim=1).to(self.device)
            # to_stm = destination == 0
            to_stm = torch.zeros_like(destination)

            with torch.no_grad():
                min_joint = nl_joint.min(dim=1)[0].view(-1, 1)
                to_expert = torch.exp(-nl_joint + min_joint)  # [B, 1+K]
                to_expert[:, 0] = 0.  # [B, 1+K]
                to_expert = \
                    to_expert / (to_expert.sum(dim=1).view(-1, 1) + 1e-7)

                to_expert = torch.from_numpy(np.array([[0.0, 1.0]])).to(self.device)

            nll_for_train = nll * (1. - to_stm.float()).unsqueeze(1)  # [B,1+K]
            # nll_for_train = nll
            losses = (nll_for_train * to_expert).sum(0)  # [1+K]
            expert_usage = to_expert.sum(dim=0)  # [K+1]
            loss = losses.sum()
            if loss.requires_grad:
                if 'update_min_usage' in self.config:
                    update_threshold = self.config['update_min_usage']
                else:
                    update_threshold = 0
                for k, usage in enumerate(expert_usage):
                    if usage > update_threshold:
                        self.experts[k].zero_grad()
                loss.backward()
                for k, usage in enumerate(expert_usage):
                    if usage > update_threshold:
                        self.experts[k].clip_grad()
                        self.experts[k].optimizer_step()
                        self.experts[k].lr_scheduler_step()

            self.train_weight_seen_since_evolution += sample_weight
            if self.train_weight_seen_since_evolution > self.train_weight_required_for_evolution:
                self.evolution += 1
                self.train_weight_seen_since_evolution = 0

        else:
            self.stm_x.extend(torch.unbind(x))
            self.stm_y.extend(torch.unbind(y))
            self.train_weight_seen_since_evolution += 1
            if self.train_weight_seen_since_evolution > self.train_weight_required_for_evolution:
                self.evolution += 1
                self.train_weight_seen_since_evolution = 0
                self.sleep()
                self.stm_x = []
                self.stm_y = []

    def sleep(self):
        print('\nGoing to sleep...')
        # Add new expert and optimizer
        expert = self.classifier
        expert.train()
        stacked_stm_x = torch.stack(self.stm_x)
        stacked_stm_y = torch.stack(self.stm_y)
        dream_dataset = TensorDataset(
            stacked_stm_x,
            stacked_stm_y)
        dream_val_x = stacked_stm_x
        dream_val_y = stacked_stm_y

        # Prepare data iterator
        dream_iterator = iter(DataLoader(
            dream_dataset,
            batch_size=self.config['sleep_batch_size'],
            num_workers=self.config['sleep_num_workers'],
            sampler=RandomSampler(
                dream_dataset,
                replacement=True,
                num_samples=(
                        self.config['sleep_step_g'] *
                        self.config['sleep_batch_size']
                ))
            # sampler=SequentialSampler(
            #     dream_dataset)
        ))

        # Train generative component
        for step, (x, y) in enumerate(dream_iterator):
            step += 1
            x, y = x.to(self.device), y.to(self.device)
            g_loss, g_summary = expert.g.nll(x, y, step=step)
            g_loss = (g_loss + self.config['weight_decay']
                      * expert.g.weight_decay_loss())
            expert.g.zero_grad()
            g_loss.mean().backward()
            expert.g.clip_grad()
            expert.g.optimizer.step()

            if step % self.config['sleep_summary_step'] == 0:
                print('\r   [Sleep-G %6d] loss: %5.1f' % (
                    step, g_loss.mean()
                ), end='')
        print()

        dream_iterator = iter(DataLoader(
            dream_dataset,
            batch_size=self.config['sleep_batch_size'],
            num_workers=self.config['sleep_num_workers'],
            sampler=RandomSampler(
                dream_dataset,
                replacement=True,
                num_samples=(
                        self.config['sleep_step_d'] *
                        self.config['sleep_batch_size'])
            )
            # sampler=SequentialSampler(
            #     dream_dataset)
        ))

        # Train discriminative component
        if not self.config['disable_d']:
            for step, (x, y) in enumerate(dream_iterator):
                step += 1
                x, y = x.to(self.device), y.to(self.device)
                d_loss, d_summary = expert.d.nll(x, y, step=step)
                d_loss = (d_loss + self.config['weight_decay']
                          * expert.d.weight_decay_loss())
                expert.d.zero_grad()
                d_loss.mean().backward()
                expert.d.clip_grad()
                expert.d.optimizer.step()

                if step % self.config['sleep_summary_step'] == 0:
                    print('\r   [Sleep-D %6d] loss: %5.1f' % (
                        step, d_loss.mean()
                    ), end='')

        expert.lr_scheduler_step()
        expert.lr_scheduler_step()
        expert.eval()
        print()                


            

    def get_votes_for_instance(self, X):
        """ Get class votes for a single instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)

        """
        pass

    def predict(self, X):
        """
        Predict using the model of the currently active state.
        """
        if self.classes is None:
            return [0]
        if self.classifier is None:
            return [self.classes[0]]
        # print(f"n Experts: {len(self.classifier.ndpm.experts)}")
        logits = self.classifier(
                            torch.from_numpy(np.concatenate([X], axis=None)).view([1, 1, -1, 1]).float())[0]

        top_vals, top_idxs = logits.topk(1, dim=0)
        return [self.classes[top_idxs.tolist()[0]]]

    def predict_proba(self, X):
        """ Predicts probabilities of all label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted the probabilities of all the labels for all instances in X.

        """
        return self.predict(X)


    def measure_tree_depth(self):
        """ Calculate the depth of the tree.

        Returns
        -------
        int
            Depth of the tree.
        """
        pass

    def estimate_model_byte_size(self):
        """ Calculate the size of the model and trigger tracker function if the actual model size exceeds the max size
        in the configuration."""
        pass

