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
    model_performance_tbl = "accuracy_log"
    labeling_log_tbl = "labeling_log"

    def __init__(self, result_processor, X_test, y_test):
        self.result_processor = result_processor
        self.X_test = X_test
        self.y_test = y_test

    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red):
        unique_values, counts = np.unique(y_u_selected)
        selected_dist = dict(zip(unique_values, counts / len(y_u_selected)))

        unique_values, counts = np.unique(y_l_aug)
        labeled_dist = dict(zip(unique_values, counts / len(y_l_aug)))

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
        y_hat = model.predict(X_test=self.X_test)
        y_hat_proba = model.predict_proba(X_test=self.X_test)

        eval_scores = {
            'iteration': iteration,
            'test_accuracy': accuracy_score(self.y_test, y_hat),
            'test_f1': f1_score(self.y_test, y_hat),
            'test_precision': precision_score(self.y_test, y_hat),
            'test_recall': recall_score(self.y_test, y_hat),
            'test_auc': roc_auc_score(self.y_true, y_hat_proba),
            'test_log_loss': log_loss(self.y_true, y_hat_proba)
        }

        self.result_processor.process_logs(eval_scores)
