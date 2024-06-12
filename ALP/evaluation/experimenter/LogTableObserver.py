from ALP.benchmark.Observer import StatisticalPerformanceObserver


class LogTableObserver(StatisticalPerformanceObserver):
    """LogTableObserver

    This class observes the performance of the model and the labeling process and logs the results in the database.

    Args:
        StatisticalPerformanceObserver (class): The parent class of the LogTableObserver.
        result_processor (class): The result processor class.
        X_test (array): The test data.
        y_test (array): The test labels.

    Attributes:
        model_performance_tbl (str): The name of the table where the model performance is logged.
        labeling_log_tbl (str): The name of the table where the labeling process is logged.

    """

    model_performance_tbl = "accuracy_log"
    labeling_log_tbl = "labeling_log"

    def __init__(self, result_processor, X_test, y_test):
        super().__init__(X_test, y_test)
        self.result_processor = result_processor

    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red):
        """
        Computes labeling statistics and log the results in the database.

        Parameters:
            iteration (int): The current iteration.
            X_u_selected (array): The selected unlabeled data.
            y_u_selected (array): The selected unlabeled labels.
            X_l_aug (array): The augmented labeled data.
            y_l_aug (array): The augmented labeled labels.
            X_u_red (array): The reduced unlabeled data.

        Returns:
            None
        """
        eval_scores = super().compute_labeling_statistics(
            iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red
        )
        self.result_processor.process_logs({LogTableObserver.labeling_log_tbl: eval_scores})

    def observe_model(self, iteration, model):
        """
        Computes model performances and log the results in the database.

        Parameters:
            iteration (int): The current iteration.
            model (class): The model

        Returns:
            None
        """
        eval_scores = super().compute_model_performances(iteration, model)
        self.result_processor.process_logs({LogTableObserver.model_performance_tbl: eval_scores})


class SparseLogTableObserver(StatisticalPerformanceObserver):
    """SparseLogTableObserver

    This class observes the performance of the model and the labeling process and logs the results in the database.
    To reduce the number of logs, the results are only logged after a whole active learning procedure is finished.

    Args:
        StatisticalPerformanceObserver (class): The parent class of the LogTableObserver.
        result_processor (class): The result processor class.
        X_test (array): The test data.
        y_test (array): The test labels.

    Attributes:
        model_performance_tbl (str): The name of the table where the model performance is logged.
        labeling_log_tbl (str): The name of the table where the labeling process is logged.
    """
    model_performance_tbl = "accuracy_log"
    labeling_log_tbl = "labeling_log"

    def __init__(self, result_processor, X_test, y_test):
        super().__init__(X_test, y_test)
        self.result_processor = result_processor

    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red):
        """
        Computes labeling statistics per iteration.

        Parameters:
            iteration (int): The current iteration.
            X_u_selected (array): The selected unlabeled data.
            y_u_selected (array): The selected unlabeled labels.
            X_l_aug (array): The augmented labeled data.
            y_l_aug (array): The augmented labels.
            X_u_red (array): The reduced unlabeled data.

        Returns:
            eval_scores (dict): The evaluation scores.
        """
        eval_scores = super().compute_labeling_statistics(
            iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red
        )
        return eval_scores

    def observe_model(self, iteration, model):
        """
        Computes model performances per iteration.

        Parameters:
            iteration (int): The current iteration.
            model (class): The model

        Returns:
            eval_scores (dict): The evaluation scores.
        """
        eval_scores = super().compute_model_performances(iteration, model)
        return eval_scores

    def log_data(self, dict):
        """
        Logs the labeling statistics in the database after the active learning procedure is finished.

        Parameters:
            dict (dict): The evaluation scores.

        Returns:
            None
        """
        self.result_processor.process_logs({LogTableObserver.labeling_log_tbl: dict})

    def log_model(self, dict):
        """
        Logs the model performances in the database after the active learning procedure is finished.

        Parameters:
            dict (dict): The evaluation scores.

        Returns:
            None
        """
        self.result_processor.process_logs({LogTableObserver.model_performance_tbl: dict})
