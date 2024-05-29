# ALPBench: A Benchmark for Active Learning Pipelines on Tabular Data



## Installation
```
git clone https://github.com/ValentinMargraf/ActiveLearningPipelines.git
cd ActiveLearningPipelines
conda create --name ALP python=3.10
conda activate ALP

# Install for usage
pip install -r req.txt

# Install for development
make install-dev
```

Documentation at https://ValentinMargraf.github.io/ActiveLearningPipelines/main

## Minimal Example

```
from sklearn.metrics import accuracy_score

from ALP.benchmark.BenchmarkConnector import DataFileBenchmarkConnector
from ALP.evaluation.experimenter.DefaultSetup import ensure_default_setup
from ALP.pipeline.ALTEvaluator import ALTEvaluator

# create benchmark connector and establish database connection
benchmark_connector = DataFileBenchmarkConnector()

# load some default settings and algorithm choices
ensure_default_setup(benchmark_connector)

evaluator = ALTEvaluator(benchmark_connector=benchmark_connector,
                     setting_name="small", openml_id=31, sampling_strategy_name="margin", learner_name="rf_gini")
alp = evaluator.fit()

# fit / predict and evaluate predictions
X_test, y_test = evaluator.get_test_data()
y_hat = alp.predict(X=X_test)
print("final test acc", accuracy_score(y_test, y_hat))

```
