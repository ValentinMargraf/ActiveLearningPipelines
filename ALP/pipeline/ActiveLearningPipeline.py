import numpy as np

from ALP.benchmark.Observer import Observer


class ActiveLearningPipeline:

    def __init__(self, learner, sampling_strategy, initializer=None, observer: Observer = None, init_budget=10,
                 num_iterations=10, num_samples_per_iteration=10):
        self.initializer = initializer
        self.learner = learner
        self.sampling_strategy = sampling_strategy
        self.observer = observer

        # the budget for sampling data points with the initialization strategy
        self.init_budget = init_budget
        # the number of active learning rounds to carry out alternating between learning and querying the oracle
        self.num_iterations = num_iterations
        # the number of data points to select in every active learning iteration to be labeled by the oracle
        self.num_samples_per_iteration = num_samples_per_iteration

    def active_fit(self, X_l, y_l, X_u, oracle):
        # X_u_idx = np.arange(len(X_u))

        # select data points from X_u to sample additional data points for initialization (i.e., uninformed) and remove
        # the sampled data points from the unlabeled dataset

        idx_available = np.arange(0, len(X_u))
        idx_queried = np.array([])

        X_u_red = X_u
        X_l_aug = X_l
        y_l_aug = y_l

        if self.initializer is not None:
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

            if self.observer is not None:
                self.observer.observe_data(0, X_u_sel, y_u_sel, X_l_aug, y_l_aug)

        # fit the initial model
        self.learner.fit(X_l_aug, y_l_aug)

        # let the observer know about the learned model
        if self.observer is not None:
            self.observer.observe_model(0, self.learner)

        for i in range(1, self.num_iterations + 1):
            # ask query strategy for samples
            idx_query = self.sampling_strategy.sample(learner=self.learner, X_l=X_l_aug, y_l=y_l_aug, X_u=X_u_red,
                                                      num_samples=self.num_samples_per_iteration)
            # get the original indices for X_u
            idx_query_orig = idx_available[idx_query]
            idx_mapping = np.array([np.where(idx_available == v)[0][0] for v in idx_query_orig])

            # delete the selected indices from the available list of indices
            idx_available = np.delete(idx_available, idx_mapping)
            idx_queried = np.concatenate((idx_queried, idx_query_orig))

            X_u_red = X_u[idx_available]
            X_u_sel = X_u[idx_query_orig]

            # query oracle for ground truth labels
            y_u_sel = oracle.query(idx_query_orig)

            # augment the labeled dataset
            X_l_aug = np.concatenate([X_l_aug, X_u_sel])
            y_l_aug = np.concatenate([y_l_aug, y_u_sel])

            # let the observer see the change in the data for this iteration
            if self.observer is not None:
                self.observer.observe_data(i, X_u_sel, y_u_sel, X_l_aug, y_l_aug)

            # fit the initial model
            self.learner.fit(X_l_aug, y_l_aug)

            # let the observer know about the learned model
            if self.observer is not None:
                self.observer.observe_model(i, self.learner)

    def predict(self, X_test):
        return self.learner.predict(X_test)
