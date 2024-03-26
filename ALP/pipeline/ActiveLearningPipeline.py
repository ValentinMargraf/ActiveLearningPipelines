import numpy as np

from ALP.benchmark.Observer import Observer


class ActiveLearningPipeline:

    def __init__(self, initializer, learner, sampling_strategy, observer: Observer = None, init_budget=10,
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
        if self.initializer is not None:
            init_ids = self.initializer.sample(X_u, self.init_budget)


            X_u_selected = X_u[init_ids]
            # label data points via the oracle
            y_u_selected = oracle.query(X_u_selected)

            # augment the given labeled data set by the data points selected for initialization
            X_l_aug = np.concatenate([X_l, X_u_selected])
            y_l_aug = np.concatenate([y_l, y_u_selected])
            X_u_red = np.delete(X_u, init_ids, axis=0)


            if self.observer is not None:
                self.observer.observe_data(0, X_u_selected, y_u_selected, X_l_aug, y_l_aug)
        else:
            X_l_aug = X_l
            y_l_aug = y_l
            X_u_red = X_u

        # fit the initial model
        self.learner.fit(X_l_aug, y_l_aug)

        # let the observer know about the learned model
        if self.observer is not None:
            self.observer.observe_model(self.learner)

        for i in range(1, self.num_iterations + 1):
            # ask query strategy for samples
            queried_ids = self.sampling_strategy.sample(self.learner, X_l_aug, y_l_aug, X_u_red, self.num_samples_per_it)

            X_u_selected = X_u_red[queried_ids]
            # query oracle for ground truth labels
            y_u_selected = oracle.query(X_u_selected)

            # add to augmented labeled dataset
            X_l_aug = np.concatenate([X_l_aug, X_u_selected])
            y_l_aug = np.concatenate([y_l_aug, y_u_selected])
            X_u_red = np.delete(X_u_red, queried_ids, axis=0)

            # let the observer see the change in the data for this iteration
            if self.observer is not None:
                self.observer.observe_data(i, X_u_selected, y_u_selected, X_l_aug, y_l_aug)

            # fit the initial model
            self.learner.fit(X_l_aug, y_l_aug)

            # let the observer know about the learned model
            if self.observer is not None:
                self.observer.observe_model(self.learner)

    def predict(self, X_test):
        return self.learner.predict(X_test)
