The ``ALPBench`` Python package
================================

The Benchmark for Active Learning Pipelines on Tabular Data ``ALPBench`` is a Python package for the specification, execution, and performance monitoring of **active learning pipelines (ALP)** consisting of a **learning algorithm** and a **query strategy** for real-world tabular classification tasks. It has built-in measures to ensure evaluations are done reproducibly, saving exact dataset splits and hyperparameter settings of used algorithms. In total, ALPBench consists of 86 real-world tabular classification datasets and 5 active learning settings, yielding 430 active learning problems. However, the benchmark allows for easy extension such as implementing your own learning algorithm and/or query strategy and benchmark it against existing approaches.


Contents
~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: OVERVIEW

   installation
   start

.. toctree::
   :maxdepth: 1
   :caption: TUTORIALS

   notebooks/TODO

.. toctree::
   :maxdepth: 2
   :caption: API REFERENCE

   api

.. toctree::
   :maxdepth: 1
   :caption: BIBLIOGRAPHY

   references
