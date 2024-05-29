import json
import time

import numpy as np
from pytorch_tabnet.callbacks import Callback

from ALP.evaluation.experimenter.LogTableObserver import LogTableObserver, SparseLogTableObserver
from ALP.util.common import fullname


class TimeLimitCallback(Callback):
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.start_time is None:
            self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit:
            print(f"Stopping training as the time limit of {self.time_limit} seconds has been reached.")
            return True  # This will stop training


class ActiveLearningPipeline:

    def __init__(
        self,
        learner,
        sampling_strategy,
        initializer=None,
        observer_list: list() = None,
        init_budget: int = None,
        num_iterations=10,
        num_samples_per_iteration=10,
    ):
        self.initializer = initializer
        self.learner = learner
        self.sampling_strategy = sampling_strategy
        self.observer_list = observer_list

        # the budget for sampling data points with the initialization strategy
        self.init_budget = init_budget
        # the number of active learning rounds to carry out alternating between learning and querying the oracle
        self.num_iterations = num_iterations
        # the number of data points to select in every active learning iteration to be labeled by the oracle
        self.num_samples_per_iteration = num_samples_per_iteration

    def active_fit(self, X_l, y_l, X_u, oracle):
        # select data points from X_u to sample additional data points for initialization (i.e., uninformed) and remove
        # the sampled data points from the unlabeled dataset
        idx_available = np.arange(0, len(X_u))
        idx_queried = np.array([])

        X_u_red = X_u
        X_l_aug = X_l
        y_l_aug = y_l

        observer_data = {}
        observer_model = {}

        all_data_used = False

        if self.initializer is not None and self.init_budget is not None:
            idx_init = self.initializer.sample(X_u, self.init_budget)
            # find the index of the sampled indices
            idx_mapped = np.array([np.where(idx_available == value)[0][0] for value in idx_init])
            # update the list of already queried indices
            idx_queried = np.concatenate((idx_queried, idx_init))
            # remove the queried indices from the list of available indices
            np.delete(idx_available, idx_mapped)

            X_u_red = X_u[idx_available]
            X_u_sel = X_u[idx_mapped]

            # label data points via the oracle
            y_u_sel = oracle.query(idx_mapped)

            # augment the given labeled data set by the data points selected for initialization
            X_l_aug = np.concatenate((X_l_aug, X_u_sel))
            y_l_aug = np.concatenate((y_l_aug, y_u_sel))

            if self.observer_list is not None:
                for o in self.observer_list:
                    if isinstance(o, LogTableObserver):
                        o.observe_data(0, X_u_sel, y_u_sel, X_l_aug, y_l_aug, X_u_red)
                    elif isinstance(o, SparseLogTableObserver):
                        observer_data[0] = o.observe_data(0, X_u_sel, y_u_sel, X_l_aug, y_l_aug, X_u_red)

        elif self.observer_list is not None:
            for o in self.observer_list:
                if isinstance(o, LogTableObserver):
                    o.observe_data(0, X_l, y_l, X_l_aug, y_l_aug, X_u_red)
                elif isinstance(o, SparseLogTableObserver):
                    observer_data[0] = o.observe_data(0, X_l, y_l, X_l_aug, y_l_aug, X_u_red)

        # fit the initial model
        learner_fqn = fullname(self.learner)
        if learner_fqn == "tabpfn.scripts.transformer_prediction_interface.TabPFNClassifier" and len(y_l_aug) > 1000:
            ids = np.random.choice(len(y_l_aug), 1000)
            self.learner.fit(X_l_aug[ids], y_l_aug[ids])
        else:
            self.learner.fit(X_l_aug, y_l_aug)

        assert len(np.unique(y_l_aug)) == len(np.unique(y_l)), "Not all classes are represented in the labeled data"

        # let the observer know about the learned model
        if self.observer_list is not None:
            for o in self.observer_list:
                if isinstance(o, LogTableObserver):
                    o.observe_model(0, self.learner)
                elif isinstance(o, SparseLogTableObserver):
                    observer_model[0] = o.observe_model(iteration=0, model=self.learner)

        for i in range(1, self.num_iterations + 1):

            if self.num_samples_per_iteration > len(idx_available):
                idx_query_orig = idx_available
                all_data_used = True

            else:
                # ask query strategy for samples
                idx_query = self.sampling_strategy.sample(
                    learner=self.learner,
                    X_l=X_l_aug,
                    y_l=y_l_aug,
                    X_u=X_u_red,
                    num_samples=self.num_samples_per_iteration,
                )
                # get the original indices for X_u
                idx_query_orig = idx_available[idx_query]

            idx_mapping = np.array([np.where(idx_available == v)[0][0] for v in idx_query_orig])

            # delete the selected indices from the available list of indices
            idx_available = np.delete(idx_available, idx_mapping)

            # if dimension > 1
            if len(idx_query_orig.shape) > 1:
                idx_query_orig = idx_query_orig.flatten()

            idx_queried = np.concatenate((idx_queried, idx_query_orig))

            X_u_red = X_u[idx_available]
            X_u_sel = X_u[idx_query_orig]

            # query oracle for ground truth labels
            y_u_sel = oracle.query(idx_query_orig)

            # augment the labeled dataset
            X_l_aug = np.concatenate([X_l_aug, X_u_sel])
            y_l_aug = np.concatenate([y_l_aug, y_u_sel])

            # let the observer see the change in the data for this iteration
            if self.observer_list is not None:
                for o in self.observer_list:
                    if isinstance(o, LogTableObserver):
                        o.observe_data(i, X_u_sel, y_u_sel, X_l_aug, y_l_aug, X_u_red)
                    elif isinstance(o, SparseLogTableObserver):
                        observer_data[i] = o.observe_data(i, X_u_sel, y_u_sel, X_l_aug, y_l_aug, X_u_red)

            # fit the initial model
            learner_fqn = fullname(self.learner)
            if (
                learner_fqn == "tabpfn.scripts.transformer_prediction_interface.TabPFNClassifier"
                and len(y_l_aug) > 1000
            ):
                ids = np.random.choice(len(y_l_aug), 1000)
                self.learner.fit(X_l_aug[ids], y_l_aug[ids])
            elif learner_fqn == "pytorch_tabnet.tab_model.TabNetClassifier":
                self.learner.fit(X_l_aug, y_l_aug, callbacks=[TimeLimitCallback(180)])
            else:
                self.learner.fit(X_l_aug, y_l_aug)

            # let the observer know about the learned model
            if self.observer_list is not None:
                for o in self.observer_list:
                    if isinstance(o, LogTableObserver):
                        o.observe_model(i, self.learner)
                    elif isinstance(o, SparseLogTableObserver):
                        observer_model[i] = o.observe_model(i, self.learner)

            if all_data_used:
                break

        # finales logging
        if self.observer_list is not None:
            for o in self.observer_list:
                if isinstance(o, SparseLogTableObserver):
                    o.log_data({"data_dict": json.dumps(observer_data)})
                    o.log_model({"model_dict": json.dumps(observer_model)})

    def predict(self, X):
        return self.learner.predict(X)
