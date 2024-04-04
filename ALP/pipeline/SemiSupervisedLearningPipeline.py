import numpy as np


class SemiSupervisedLearningPipeline:

    def __init__(self, learner, observer_list: list() = None, num_iterations=10, num_samples_per_iteration=10):
        self.learner = learner
        self.observer_list = observer_list

        # the number of active learning rounds to carry out alternating between learning and querying the oracle
        self.num_iterations = num_iterations
        # the number of data points to select in every active learning iteration to be labeled by the oracle
        self.num_samples_per_iteration = num_samples_per_iteration

    def semi_supervised_fit(self, X_l, y_l, X_u, labeler):
        # select data points from X_u to sample additional data points for initialization (i.e., uninformed) and remove
        # the sampled data points from the unlabeled dataset
        idx_available = np.arange(0, len(X_u))
        idx_already_labeled = np.array([])

        X_u_red = X_u
        X_l_aug = X_l
        y_l_aug = y_l

        # fit the initial model
        self.learner.fit(X_l_aug, y_l_aug)

        # let the observer know about the learned model
        if self.observer_list is not None:
            for o in self.observer_list:
                o.observe_model(iteration=0, model=self.learner)

        for i in range(1, self.num_iterations + 1):
            # query labeler for more labels
            idx_labeled, y_u_labeled = labeler.label(learner=self.learner, X_l=X_l_aug, y_l=y_l_aug, X_u=X_u_red,
                                                     num_samples=self.num_samples_per_iteration)

            # get the original indices for X_u
            idx_labeled_orig = idx_available[idx_labeled]
            idx_mapping = np.array([np.where(idx_available == v)[0][0] for v in idx_labeled_orig])

            # delete the selected indices from the available list of indices
            idx_available = np.delete(idx_available, idx_mapping)
            idx_already_labeled = np.concatenate((idx_already_labeled, idx_labeled_orig))

            X_u_red = X_u[idx_available]
            X_u_sel = X_u[idx_labeled_orig]

            # augment the labeled dataset
            X_l_aug = np.concatenate([X_l_aug, X_u_sel])
            y_l_aug = np.concatenate([y_l_aug, y_u_labeled])

            # let the observer see the change in the data for this iteration
            if self.observer_list is not None:
                for o in self.observer_list:
                    o.observe_data(i, X_u_sel, y_u_labeled, idx_labeled_orig, X_l_aug, y_l_aug, X_u_red)

            # fit the initial model
            self.learner.fit(X_l_aug, y_l_aug)

            # let the observer know about the learned model
            if self.observer_list is not None:
                for o in self.observer_list:
                    o.observe_model(i, self.learner)

    def predict(self, X):
        return self.learner.predict(X)
