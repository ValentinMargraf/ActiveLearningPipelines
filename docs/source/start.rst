Quickstart Guide
================

You can use `ALPBench` in different ways. There already exist quite some learners and query strategies that can be
run through accessing them with their name, as can be seen in the minimal example below. In the `ALP.pipeline` module, you
can also implement your own (new) query strategies.

Fit an Active Learning Pipeline
-------------------------------

Fit an ALP on a dataset with `openmlid` 31, using a random forest and margin sampling. You can find similar example code snippets in
**examples/**.

.. code-block:: python

    from sklearn.metrics import accuracy_score

    from ALP.benchmark.BenchmarkConnector import DataFileBenchmarkConnector
    from ALP.evaluation.experimenter.DefaultSetup import ensure_default_setup
    from ALP.pipeline.ALPEvaluator import ALPEvaluator

    # create benchmark connector and establish database connection
    benchmark_connector = DataFileBenchmarkConnector()

    # load some default settings and algorithm choices
    ensure_default_setup(benchmark_connector)

    evaluator = ALPEvaluator(benchmark_connector=benchmark_connector,
                             setting_name="small", openml_id=31, sampling_strategy_name="margin", learner_name="rf_gini")
    alp = evaluator.fit()

    # fit / predict and evaluate predictions
    X_test, y_test = evaluator.get_test_data()
    y_hat = alp.predict(X=X_test)
    print("final test acc", accuracy_score(y_test, y_hat))

    >> final test acc 0.7181818181818181
