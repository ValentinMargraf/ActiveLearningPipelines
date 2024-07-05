import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from alpbench.util.common import fullname


class Ensemble:
    """Ensemble

    This class is used to create an ensemble of estimators. The ensemble can be used to predict the probabilities of the
    ensemble members and the classes of the ensemble members.

    Args:
        estimator: object
        num_estimators: int
        max_neighbors: int (for k nearest neighbors) else None

    Attributes:
        estimator: object (the estimator to construct the ensemble of)
        num_estimators: int (the number of estimators in the ensemble)
        max_neighbors: int  (for k nearest neighbors)
        random_states: list (random states for the ensemble members)
        estimators_: list   (list containing the ensemble members)
        learner_fqn: str    (fully qualified name of the estimator)
    """

    def __init__(self, estimator, num_estimators, max_neighbors=None):
        self.estimator = estimator
        self.num_estimators = num_estimators
        self.max_neighbors = max_neighbors
        self.random_states = [np.random.randint(0, 1000) for _ in range(num_estimators)]
        self.estimators_ = None
        self.learner_fqn = fullname(self.estimator)
        self.init()

    def init(self):
        """Initializes the ensemble members.

        Returns:
            None
        """
        self.estimators_ = []
        if self.learner_fqn == "tabpfn.scripts.transformer_prediction_interface.TabPFNClassifier":
            from alpbench.util.transformer_prediction_interface_ens import TabPFNClassifierEns as TabPFNEns
            self.estimator = TabPFNEns(N_ensemble_configurations=self.num_estimators)
        if self.learner_fqn == "pytorch_tabnet.tab_model.TabNetClassifier":
            for seed in self.random_states:
                from alpbench.util.pytorch_tabnet.tab_model import TabNetClassifier
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
                    from alpbench.util.TorchUtil import TimeLimitCallback
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
                elif self.learner_fqn == "sklearn.ensemble._forest.RandomForestClassifier":
                    preds[:, :, i] = self.estimator.estimators_[i].predict_proba(X)
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
