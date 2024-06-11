Installation Guide
==================

.. note::
    `ALPBench` is intended to work with **Python 3.10 and above**.

Installation Steps
------------------

1. **Clone the Repository**:

    .. code-block:: sh

        git clone https://github.com/ValentinMargraf/ActiveLearningPipelines.git
        cd ActiveLearningPipelines

2. **Create and Activate a Conda Environment**:

    .. code-block:: sh

        conda create --name ALP python=3.10
        conda activate ALP

3. **Install Dependencies**:

    - **For usage (without TabNet and TabPFN)**:

        .. code-block:: sh

            pip install -r requirements.txt

    - **OR**

    - **For usage (with TabNet and TabPFN)**:

        .. code-block:: sh

            pip install -r requirements_full.txt

4. **Install for Development**:

    .. code-block:: sh

        make install-dev
