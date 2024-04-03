import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ALP.benchmark.Observer import Observer


class LogTableObserver(Observer):
    precision = 8
    model_performance_tbl = "accuracy_log"
    labeling_log_tbl = "labeling_log"

    def __init__(self, result_processor, X_test, y_test):
        self.result_processor = result_processor
        self.X_test = X_test
        self.y_test = y_test

    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red):
        if len(y_u_selected) > 1:
            unique_values, counts = np.unique(np.array(y_u_selected), return_counts=True)
            selected_dist = dict(zip(unique_values, (counts / len(y_u_selected))))
            for k in selected_dist.keys():
                selected_dist[k] = round(selected_dist[k], LogTableObserver.precision)
        else:
            selected_dist = {str(y_u_selected): 1}

        unique_values, counts = np.unique(y_l_aug, return_counts=True)
        labeled_dist = dict(zip(unique_values, counts / len(y_l_aug)))
        for k in labeled_dist.keys():
            labeled_dist[k] = round(labeled_dist[k], LogTableObserver.precision)

        eval_scores = {
            'iteration': iteration,
            'len_X_sel': len(X_u_selected),
            'len_X_l': len(X_l_aug),
            'len_X_u': len(X_u_red),
            'y_sel_dist': str(selected_dist),
            'y_l_dist': str(labeled_dist)
        }

        self.result_processor.process_logs(LogTableObserver.labeling_log_tbl, eval_scores)

    def observe_model(self, iteration, model):
        eval_scores = {"iteration": iteration}
        y_hat = model.predict(self.X_test)
        y_hat_proba = model.predict_proba(self.X_test)

        eval_scores["test_accuracy"] = round(accuracy_score(self.y_test, y_hat), LogTableObserver.precision)

        if len(np.unique(self.y_test)) == 2:
            eval_scores["test_auc"] = round(roc_auc_score(self.y_test, y_hat_proba[:, 1]), LogTableObserver.precision)
        else:
            eval_scores["test_auc"] = round(roc_auc_score(self.y_test, y_hat_proba, multi_class='ovr'),
                                            LogTableObserver.precision)

        eval_scores = {
            'test_f1': round(f1_score(self.y_test, y_hat, average="weighted"), LogTableObserver.precision),
            'test_precision': round(precision_score(self.y_test, y_hat, average="weighted"),
                                    LogTableObserver.precision),
            'test_recall': round(recall_score(self.y_test, y_hat, average="weighted"), LogTableObserver.precision),
            'test_log_loss': round(log_loss(self.y_test, y_hat_proba), LogTableObserver.precision)
        }

        self.result_processor.process_logs(LogTableObserver.model_performance_tbl, eval_scores)
