[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/ValentinMargraf/ActiveLearningPipelines/badge.svg)](https://coveralls.io/github/ValentinMargraf/ActiveLearningPipelines)
[![Tests](https://github.com/ValentinMargraf/ActiveLearningPipelines/actions/workflows/pytest.yml/badge.svg)](https://github.com/ValentinMargraf/ActiveLearningPipelines/actions/workflows/pytest.yml)
[![Read the Docs](https://readthedocs.org/projects/shapiq/badge/?version=latest)](https://activelearningpipelines.readthedocs.io/en/latest/?badge=latest)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ALPBench: A Benchmark for Active Learning Pipelines on Tabular Data
`ALPBench` is a Python package for the specification, execution, and performance monitoring of **active learning pipelines (ALP)** consisting of a **learning algorithm** and a **query strategy** for real-world tabular classification tasks. It has built-in measures to ensure evaluations are done reproducibly, saving exact dataset splits and hyperparameter settings of used algorithms. In total, ALPBench consists of 86 real-world tabular classification datasets and 5 active learning settings, yielding 430 active learning problems. However, the benchmark allows for easy extension such as implementing your own learning algorithm and/or query strategy and benchmark it against existing approaches.


# 🛠️ Install
`ALPBench` is intended to work with **Python 3.10 and above**.
```
git clone https://github.com/ValentinMargraf/ActiveLearningPipelines.git
cd ActiveLearningPipelines
conda create --name ALP python=3.10
conda activate ALP

# Install for usage (without TabNet and TabPFN)
pip install -r requirements.txt

# OR
# Install for usage (with TabNet and TabPFN)
pip install -r requirements_full.txt

# Install for development
make install-dev
```

Documentation at https://activelearningpipelines.readthedocs.io/en/latest/


# ⭐ Quickstart
You can use `ALPBench` in different ways. There already exist quite some learners and query strategies that can be
run through accessing them with their name, as can be seen in the minimal example below. In the ALP.pipeline module you
can also implement your own (new) query strategies.


## 📈 Fit an Active Learning Pipeline

Fit an ALP on dataset with openmlid 31, using a random forest and margin sampling. You can find similar example code snippets in
**examples/**.

```python
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

>> final
test
acc
0.7181818181818181

```
