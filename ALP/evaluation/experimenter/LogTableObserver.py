from ALP.benchmark.Observer import StatisticalPerformanceObserver


class LogTableObserver(StatisticalPerformanceObserver):
    model_performance_tbl = "accuracy_log"
    labeling_log_tbl = "labeling_log"

    def __init__(self, result_processor, X_test, y_test):
        super().__init(X_test, y_test)
        self.result_processor = result_processor

    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red):
        eval_scores = super().compute_labeling_statistics(iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug,
                                                          X_u_red)
        self.result_processor.process_logs(
            {LogTableObserver.labeling_log_tbl: eval_scores})

    def observe_model(self, iteration, model):
        eval_scores = super().compute_model_performances(iteration, model)
        self.result_processor.process_logs(
            {LogTableObserver.model_performance_tbl: eval_scores})
