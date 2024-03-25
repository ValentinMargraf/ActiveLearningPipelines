import sklearn.ensemble as ens
import numpy as np
import pandas as pd
import scipy
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold

class RandomForestEns(ens.RandomForestClassifier):
    """
    An ensemble random forest classifiers that can be used for uncertainty quantification, both aleatoric and epistemic, based on entropy.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_liks = np.zeros(self.n_estimators)
        self.names = None
        self._initialize_params()

    def fit_ensemble(self, X: np.ndarray, y: np.ndarray):
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
        self.fit(X,y)
        self.members = self.estimators_
        self.num_members = len(self.members)
        self.ids = np.arange(self.num_members)
        self.num_classes = self.members[0].n_classes_
        self.names = ["rf"+ str(i) for i in range(len(self.estimators_))]


    def _initialize_params(self):
        """
        Initializes the parameters of the ensemble.

        Args
        ----------
        None

        Returns
        -------
        None
        """
        self.n_estimators=250
        self.max_depth=10
        self.bootstrap=True
        self.criterion="entropy"


    def set_norm_liks(self, X: np.ndarray, y: np.ndarray):
        """
        Sets the normalized likelihoods of the ensemble members.

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
        for i in range(self.n_estimators):
            y_pred = self.estimators_[i].predict_proba(X)
            lik = np.sum(np.log(y_pred[np.arange(y.shape[0]), y] + 1e-10))
            lik = -1 / lik
            self.norm_liks[i] = lik
        self.norm_liks /= self.norm_liks.max()

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
        preds = np.empty((X.shape[0], self.n_classes_, self.n_estimators))
        for i in range(self.n_estimators):
            if alpha is None or self.norm_liks[i] >= alpha:
                preds[:, :, i] = self.estimators_[i].predict_proba(X)
        #print("PREEEEEEEEEDS ", preds)

        return preds

    def predict(self, X: np.ndarray, alpha: float=None) -> np.ndarray:
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
        return np.argmax(probas.mean(axis=-1), axis = 1)

    def _sort_by_confidence(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Sorts the instances by the ensemble's confidence (asscending order).

        Args
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        ids : sorted indices, array-like of shape (n_samples,)
        preds : predicted classes, array-like of shape (n_samples,)
        """
        preds = self.predict_proba(X)
        mean_prediction = preds.mean(axis=-1)
        max_class = np.max(mean_prediction, axis=-1)
        ids = np.argsort(max_class)
        return ids, preds[ids], max_class[ids]


    def _sort_by_eu(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Sorts the instances by epistemic uncertainty (ascending order).

        Args
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        ids : sorted indices, array-like of shape (n_samples,)
        preds : predicted classes, array-like of shape (n_samples,)
        eu : epistemic uncertainties, array-like of shape (n_samples,)
        """
        probs = self.predict_proba(X)
        #print("PROOOOOOOOOBS ", probs)
        eu = self._epistemic_uncertainty_entropy(probs)
        ids = np.argsort(eu)
        return ids, probs.mean(axis=-1).max(axis=-1)[ids], eu[ids]

    def _sort_by_au(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Sorts the instances by aleatoric uncertainty (ascending order).

        Args
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        ids : sorted indices, array-like of shape (n_samples,)
        preds : predicted classes, array-like of shape (n_samples,)
        au : aleatoric uncertainties, array-like of shape (n_samples,)
        """

        probs = self.predict_proba(X)
        au = self._aleatoric_uncertainty_entropy(probs)
        #au = self.general_aleatoric_uncertainty(probs)
        ids = np.argsort(au)
        #print(au[ids[:5]])
        return ids, probs.mean(axis=-1).max(axis=-1)[ids], au[ids]

    def _sort_by_tu(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Sorts the instances by total uncertainty (ascending order).

        Args
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        ids : sorted indices, array-like of shape (n_samples,)
        """

        probs = self.predict_proba(X)
        tu = self._total_uncertainty_entropy(probs)
        ids = np.argsort(tu)
        return ids, probs.mean(axis=-1).max(axis=-1)[ids], tu[ids]

    def _epistemic_uncertainty_entropy(self, probs: np.ndarray) -> np.ndarray:
        """
        Calculates the epistemic uncertainty based on entropy.

        Args
        ----------
        probs : predicted probabilities, array-like of shape (n_samples, n_classes, n_estimators)

        Returns
        -------
        e_u : epistemic uncertainty, array-like of shape (n_samples,)
        """

        mean_probs = np.mean(probs, axis=2)
        mean_probs = np.repeat(np.expand_dims(mean_probs, 2), repeats=probs.shape[2], axis=2)
        mean_probs = np.clip(mean_probs, 1e-25, 1)
        probs = np.clip(probs, 1e-25, 1)
        e_u = entropy(probs, mean_probs, axis=1, base=2) / np.log2(probs.shape[1])
        e_u = np.mean(e_u, axis=1)
        return e_u


    def _aleatoric_uncertainty_entropy(self, probs: np.ndarray) -> np.ndarray:
        """
        Calculates the aleatoric uncertainty based on entropy.

        Args
        ----------
        probs : predicted probabilities, array-like of shape (n_samples, n_classes, n_estimators)

        Returns
        -------
        a_u : aleatoric uncertainty, array-like of shape (n_samples,)
        """

        a_u = entropy(probs, axis=1, base=2) / np.log2(probs.shape[1])
        a_u = np.mean(a_u, axis=1)
        return a_u

    def _total_uncertainty_entropy(self, probs: np.ndarray) -> np.ndarray:
        """
        Calculates the total uncertainty based on entropy.

        Args
        ----------
        probs : predicted probabilities, array-like of shape (n_samples, n_classes, n_estimators)

        Returns
        -------
        t_u : total uncertainty, array-like of shape (n_samples,)
        """

        t_u = entropy(np.mean(probs, axis=2), axis=1, base=2) / np.log2(probs.shape[1])
        return t_u
