from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score


class Observer(ABC):
    """Observer

    Abstract class for the observer of the active learning process. The observer is responsible for keeping track of
    the indices of the selected data, the labeled data, and the unlabeled data, as well as the model performance on the
    test data in each iteration..
    """

    def __init__(self):
        return

    @abstractmethod
    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red):
        """
        Abstract method to observe the data in each iteration.
        """
        pass

    @abstractmethod
    def observe_model(self, iteration, model):
        """
        Abstract method to observe the model performances in each iteration.
        """
        pass


class StatisticalPerformanceObserver(Observer, ABC):
    """StatisticalPerformanceObserver

    Observer class for statistical performance evaluation of the active learning process such as the distribution of
    the labeled and selected data, and the model performance on the test data.

    Args:
        X_test (np.array): test data
        y_test (np.array): test labels

    Attributes:
        X_test (np.array): test data
        y_test (np.array): test labels
        precision (int): precision of the floating point numbers
    """

    precision = 8

    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test

    def compute_labeling_statistics(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red):
        """
        Computes the distribution of the data, which involves the ids of the selected data in each iteration, the
        overall labeled and unlabeled data.

        Parameters:
            iteration (int): iteration number
            X_u_selected (np.array): selected data
            y_u_selected (np.array): selected labels
            X_l_aug (np.array): labeled data
            y_l_aug (np.array): labeled labels
            X_u_red (np.array): unlabeled data

        Returns:
            eval_scores (dict): dictionary with the distribution of the labeled and selected data

        """
        if len(y_u_selected) > 1:
            unique_values, counts = np.unique(np.array(y_u_selected), return_counts=True)
            selected_dist = dict(zip(unique_values, (counts / len(y_u_selected))))
            for k in selected_dist.keys():
                selected_dist[k] = round(selected_dist[k], StatisticalPerformanceObserver.precision)
        else:
            selected_dist = {str(y_u_selected): 1}

        unique_values, counts = np.unique(y_l_aug, return_counts=True)
        labeled_dist = dict(zip(unique_values, counts / len(y_l_aug)))
        for k in labeled_dist.keys():
            labeled_dist[k] = round(labeled_dist[k], StatisticalPerformanceObserver.precision)

        eval_scores = {
            "iteration": iteration,
            "len_X_sel": len(X_u_selected),
            "len_X_l": len(X_l_aug),
            "len_X_u": len(X_u_red),
            "y_sel_dist": str(selected_dist),
            "y_l_dist": str(labeled_dist),
        }
        return eval_scores

    def compute_model_performances(self, iteration, model):
        """
        Computes the model performance on the test data, which involves the f1, precision, recall, log loss, accuracy,
        and AUC scores.

        Parameters:
            iteration (int): iteration number
            model (object): trained model

        Returns:
            eval_scores (dict): dictionary with the model performance on the test data
        """
        eval_scores = {"iteration": iteration}
        y_hat = model.predict(self.X_test)
        y_hat_proba = model.predict_proba(self.X_test)

        eval_scores = {
            "iteration": iteration,
            "test_f1": round(
                f1_score(self.y_test, y_hat, average="weighted"), StatisticalPerformanceObserver.precision
            ),
            "test_precision": round(
                precision_score(self.y_test, y_hat, average="weighted"), StatisticalPerformanceObserver.precision
            ),
            "test_recall": round(
                recall_score(self.y_test, y_hat, average="weighted"), StatisticalPerformanceObserver.precision
            ),
            "test_log_loss": round(log_loss(self.y_test, y_hat_proba), StatisticalPerformanceObserver.precision),
            "test_accuracy": round(accuracy_score(self.y_test, y_hat), StatisticalPerformanceObserver.precision),
        }

        if len(np.unique(self.y_test)) == 2:
            eval_scores["test_auc"] = round(
                roc_auc_score(self.y_test, y_hat_proba[:, 1]), StatisticalPerformanceObserver.precision
            )
        else:
            eval_scores["test_auc"] = round(
                roc_auc_score(self.y_test, y_hat_proba, multi_class="ovr"), StatisticalPerformanceObserver.precision
            )
        return eval_scores


class PrintObserver(StatisticalPerformanceObserver):
    """PrintObserver

    Observer class to get the statistical performance evaluation of the active learning process such as the
    distribution of the labeled and selected data, and the model performance on the test data.

    Args:
        X_test (np.array): test data
        y_test (np.array): test labels

    Attributes:
        X_test (np.array): test data
        y_test (np.array): test labels
    """

    def __init__(self, X_test, y_test):
        super().__init__(X_test, y_test)
        self.X_test = X_test
        self.y_test = y_test

    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red):
        """
        Observes the data in each iteration.

        Parameters:
            iteration (int): iteration number
            X_u_selected (np.array): selected data
            y_u_selected (np.array): selected labels
            X_l_aug (np.array): labeled data
            y_l_aug (np.array): labels of labeled data
            X_u_red (np.array): unlabeled data
        """
        super().compute_labeling_statistics(iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red)

    def observe_model(self, iteration, model):
        """
        Observes the model performances in each iteration.

        Parameters:
            iteration (int): iteration number
            model (object): trained model
        """
        super().compute_model_performances(iteration, model)
