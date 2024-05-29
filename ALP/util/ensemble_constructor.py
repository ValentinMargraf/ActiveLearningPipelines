import time

import numpy as np
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ALP.transformer_prediction_interface_ens import TabPFNClassifierEns as TabPFNEns
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


class Ensemble:
    """
    An ensemble random forest classifiers that can be used for uncertainty quantification,
    both aleatoric and epistemic, based on entropy.
    """

    def __init__(self, estimator, num_estimators, max_neighbors):
        self.estimator = estimator
        self.num_estimators = num_estimators
        self.max_neighbors = max_neighbors
        self.random_states = [np.random.randint(0, 1000) for _ in range(num_estimators)]
        self.estimators_ = None
        self.learner_fqn = fullname(self.estimator)
        self.init()

    def init(self):
        self.estimators_ = []

        if self.learner_fqn == "tabpfn.scripts.transformer_prediction_interface.TabPFNClassifier":
            self.estimator = TabPFNEns(N_ensemble_configurations=self.num_estimators)
        if self.learner_fqn == "pytorch_tabnet.tab_model.TabNetClassifier":
            for seed in self.random_states:
                self.estimators_.append(TabNetClassifier(seed=seed, verbose=0))
        if self.learner_fqn == "catboost.core.CatBoostClassifier":
            self.num_estimators = self.estimator.tree_count_
        if self.learner_fqn == "xgboost.sklearn.XGBClassifier":
            self.num_estimators = self.estimator.n_estimators
        if self.learner_fqn == "sklearn.ensemble._forest.RandomForestClassifier":
            self.estimators_ = self.estimator.estimators_
        if self.learner_fqn == "sklearn.svm._classes.SVC":
            for seed in np.linspace(1, 20, self.num_estimators):
                self.estimators_.append(SVC(kernel="rbf", probability=True, C=seed))
        if self.learner_fqn == "sklearn.neighbors._classification.KNeighborsClassifier":
            num_neighbors = np.random.choice(
                np.linspace(1, self.max_neighbors, self.max_neighbors, dtype=int), self.num_estimators
            )
            for seed in num_neighbors:
                self.estimators_.append(KNeighborsClassifier(n_neighbors=seed))
        else:
            self.estimators_ = [self.estimator for _ in range(self.num_estimators)]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the ensemble and sets the attributes of the class.

        Args
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
        The target values (class labels) as integers.

        Returns
        -------
        None
        """
        if self.learner_fqn in [
            "sklearn.ensemble._forest.RandomForestClassifier",
            "xgboost.sklearn.XGBClassifier",
            "catboost.core.CatBoostClassifier",
            "tabpfn.scripts.transformer_prediction_interface.TabPFNClassifier",
        ]:
            if self.learner_fqn == "tabpfn.scripts.transformer_prediction_interface.TabPFNClassifier" and len(y) > 1000:
                ids = np.random.choice(len(y), 1000)
                self.estimator.fit(X[ids], y[ids])
            else:
                self.estimator.fit(X, y)
        else:
            for estimator in self.estimators_:
                if self.learner_fqn == "pytorch_tabnet.tab_model.TabNetClassifier":
                    estimator.fit(X, y, callbacks=[TimeLimitCallback(60)])
                else:
                    estimator.fit(X, y)
        self.n_classes_ = len(np.unique(y))

    def predict_proba(self, X: np.ndarray, alpha: float = None) -> np.ndarray:
        """
        Predicts the probabilities of the ensemble members.

        Args
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
            alpha : float, optional (default=None)
            The threshold for the normalized likelihoods of the ensemble members.

        Returns
        -------
        preds : predicted probabilities, array-like of shape (n_samples, n_classes, n_estimators)

        """
        if self.learner_fqn == "tabpfn.scripts.transformer_prediction_interface.TabPFNClassifier":
            return self.estimator.predict_proba(X).transpose(0, 2, 1)
        else:
            preds = np.empty((X.shape[0], self.n_classes_, self.num_estimators))
            for i in range(self.num_estimators):
                if self.learner_fqn == "xgboost.sklearn.XGBClassifier":
                    preds[:, :, i] = self.estimator.predict_proba(X, iteration_range=(i, i + 1))
                elif self.learner_fqn == "catboost.core.CatBoostClassifier":
                    preds[:, :, i] = self.estimator.predict_proba(X, ntree_start=i, ntree_end=i + 1)
                else:
                    preds[:, :, i] = self.estimators_[i].predict_proba(X)

            return preds

    def predict(self, X: np.ndarray, alpha: float = None) -> np.ndarray:
        """
        Predicts the classes of the ensemble members.

        Args
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
            alpha : float, optional (default=None)
            The threshold for the normalized likelihoods of the ensemble members.

        Returns
        -------
        preds : predicted classes, array-like of shape (n_samples, n_estimators)
        """

        probas = self.predict_proba(X, alpha)
        return np.argmax(probas.mean(axis=-1), axis=1)
