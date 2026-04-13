# kvbiii-plots

## Overview
`kvbiii-plots` is a modular Python package for advanced data visualization, focusing on exploratory data analysis (EDA), machine learning (ML), and model evaluation.

## Folder Structure
```
┣━ 📁 .github
┃  ┗━ 📁 workflows
┃     ┗━ 📜 python-publish.yml
┣━ 📁 kvbiii_plots
┃  ┣━ 📁 eda
┃  ┃  ┣━ 📁 examples
┃  ┃  ┃  ┣━ 📓 categorical_plots_examples.ipynb
┃  ┃  ┃  ┣━ 📓 continuous_plots_examples.ipynb
┃  ┃  ┃  ┣━ 📓 multivariate_plots_examples.ipynb
┃  ┃  ┃  ┗━ 📓 time_series_plots_examples.ipynb
┃  ┃  ┣━ 🐍 __init__.py
┃  ┃  ┣━ 🐍 categorical_plots.py
┃  ┃  ┣━ 🐍 continuous_plots.py
┃  ┃  ┣━ 🐍 multivariate_plots.py
┃  ┃  ┗━ 🐍 time_series_plots.py
┃  ┣━ 📁 evaluation
┃  ┃  ┣━ 📁 examples
┃  ┃  ┃  ┣━ 📓 classification_plots_examples.ipynb
┃  ┃  ┃  ┣━ 📓 regression_plots_examples.ipynb
┃  ┃  ┃  ┗━ 📓 shap_plots_examples.ipynb
┃  ┃  ┣━ 🐍 __init__.py
┃  ┃  ┣━ 🐍 classification_plots.py
┃  ┃  ┣━ 🐍 regression_plots.py
┃  ┃  ┗━ 🐍 shap_plots.py
┃  ┣━ 📁 ml
┃  ┃  ┣━ 📁 examples
┃  ┃  ┃  ┣━ 📓 optuna_plots_examples.ipynb
┃  ┃  ┃  ┗━ 📓 other_plots_examples.ipynb
┃  ┃  ┣━ 🐍 __init__.py
┃  ┃  ┣━ 🐍 optuna_plots.py
┃  ┃  ┗━ 🐍 other_plots.py
┃  ┣━ 🐍 __init__.py
┃  ┗━ 🐍 base_plots.py
┣━ 📁 tests
┃  ┣━ 📁 eda
┃  ┃  ┣━ 🐍 test_categorical_plots.py
┃  ┃  ┣━ 🐍 test_continuous_plots.py
┃  ┃  ┣━ 🐍 test_eda_init.py
┃  ┃  ┣━ 🐍 test_multivariate_plots.py
┃  ┃  ┗━ 🐍 test_time_series_plots.py
┃  ┣━ 📁 evaluation
┃  ┃  ┣━ 🐍 test_classification_plots.py
┃  ┃  ┣━ 🐍 test_evaluation_init.py
┃  ┃  ┣━ 🐍 test_regression_plots.py
┃  ┃  ┗━ 🐍 test_shap_plots.py
┃  ┣━ 📁 ml
┃  ┃  ┣━ 🐍 test_ml_init.py
┃  ┃  ┣━ 🐍 test_optuna_plots.py
┃  ┃  ┗━ 🐍 test_other_plots.py
┃  ┣━ 🐍 conftest.py
┃  ┣━ 🐍 test_base_plots.py
┃  ┗━ 🐍 test_main_init.py
┣━ 👻 .gitignore
┣━ ⚙️ pyproject.toml
┣━ 📖 README.md
┣━ 📃 requirements.txt
┗━ 🐚 setup_venv.sh
```

## Files Description
* 📁 `.github`: GitHub configuration files.
	- 📁 `workflows`: CI/CD workflow definitions.
		- 📜 `python-publish.yml`: Python package publishing workflow.
* 📁 `kvbiii_plots`: Main package source code.
	- 📁 `eda`: Exploratory Data Analysis plotting modules.
		- 📁 `examples`: Example Jupyter notebooks for EDA plots.
			- 📓 `categorical_plots_examples.ipynb`: Categorical plot examples.
			- 📓 `continuous_plots_examples.ipynb`: Continuous plot examples.
			- 📓 `multivariate_plots_examples.ipynb`: Multivariate plot examples.
			- 📓 `time_series_plots_examples.ipynb`: Time series plot examples.
		- 🐍 `__init__.py`: EDA module initializer.
		- 🐍 `categorical_plots.py`: Functions for categorical plots.
		- 🐍 `continuous_plots.py`: Functions for continuous plots.
		- 🐍 `multivariate_plots.py`: Functions for multivariate plots.
		- 🐍 `time_series_plots.py`: Functions for time series plots.
	- 📁 `evaluation`: Model evaluation plotting modules.
		- 📁 `examples`: Example notebooks for evaluation plots.
			- 📓 `classification_plots_examples.ipynb`: Classification plot examples.
			- 📓 `regression_plots_examples.ipynb`: Regression plot examples.
			- 📓 `shap_plots_examples.ipynb`: SHAP plot examples.
		- 🐍 `__init__.py`: Evaluation module initializer.
		- 🐍 `classification_plots.py`: Functions for classification plots.
		- 🐍 `regression_plots.py`: Functions for regression plots.
		- 🐍 `shap_plots.py`: Functions for SHAP plots.
	- 📁 `ml`: Machine learning plotting modules.
		- 📁 `examples`: Example notebooks for ML plots.
			- 📓 `optuna_plots_examples.ipynb`: Optuna optimization plot examples.
			- 📓 `other_plots_examples.ipynb`: Miscellaneous ML plot examples.
		- 🐍 `__init__.py`: ML module initializer.
		- 🐍 `optuna_plots.py`: Functions for Optuna plots.
		- 🐍 `other_plots.py`: Other ML-related plots.
	- 🐍 `__init__.py`: Main package initializer.
	- 🐍 `base_plots.py`: Base plotting utilities.
* 📁 `tests`: Unit tests for all modules.
	- 📁 `eda`: Tests for EDA plotting functions.
		- 🐍 `test_categorical_plots.py`: Tests for categorical plots.
		- 🐍 `test_continuous_plots.py`: Tests for continuous plots.
		- 🐍 `test_eda_init.py`: Tests for EDA module initialization.
		- 🐍 `test_multivariate_plots.py`: Tests for multivariate plots.
		- 🐍 `test_time_series_plots.py`: Tests for time series plots.
	- 📁 `evaluation`: Tests for evaluation plotting functions.
		- 🐍 `test_classification_plots.py`: Tests for classification plots.
		- 🐍 `test_evaluation_init.py`: Tests for evaluation module initialization.
		- 🐍 `test_regression_plots.py`: Tests for regression plots.
		- 🐍 `test_shap_plots.py`: Tests for SHAP plots.
	- 📁 `ml`: Tests for ML plotting functions.
		- 🐍 `test_ml_init.py`: Tests for ML module initialization.
		- 🐍 `test_optuna_plots.py`: Tests for Optuna plots.
		- 🐍 `test_other_plots.py`: Tests for other ML plots.
	- 🐍 `conftest.py`: Pytest configuration.
	- 🐍 `test_base_plots.py`: Tests for base plotting utilities.
	- 🐍 `test_main_init.py`: Tests for main package initialization.
* 👻 `.gitignore`: Git ignored files specification.
* ⚙️ `pyproject.toml`: Project metadata and build configuration.
* 📖 `README.md`: Main project documentation.
* 📃 `requirements.txt`: Python dependencies list.
* 🐚 `setup_venv.sh`: Shell script for setting up a virtual environment.

## Installation
To install the repository, follow these steps:
1. Clone the repository:
```bash
git clone <repo_url>
```

2. Navigate to the repository directory:
```bash
cd kvbiii-plots
```

3. Create a virtual environment (optional but recommended):
```bash
bash setup_venv.sh
```

4. Activate the virtual environment:
```bash
source <venv_name>/bin/activate
```

## Usage
You can check the example Jupyter notebooks in the `examples` folders within each module for usage examples.

For instance, see [categorical_plots_examples.ipynb](https://github.com/kvbiii/kvbiii-plots/blob/main/kvbiii_plots/eda/examples/categorical_plots_examples.ipynb).

-------------------------------------------
**Last updated on 2026-04-13 20:25:10**