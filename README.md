# Diagnostic Score Comparison Project

## Overview

This repository contains code for analyzing treatment-covariate interactions in continuous treatment settings using a semiparametric regression model. The project implements repeated Nadaraya-Watson regression, cross-validation strategies, and model selection methodologies to estimate prognostic and treatment-interaction scores. This work supports the study described in the paper *Learning Interactions Between Continuous Treatments and Covariates with a Semiparametric Model*, submitted to the Conference on Health, Inference, and Learning (CHIL) 2025.

## Repository Structure

```
📂 Project Root
│-- functions.py                 # Core functions for kernel smoothing, optimization, model training, and evaluation
│-- diag_score_comparison.ipynb  # Jupyter notebook for diagnostic score comparisons in the IWPC study
│-- simulation.ipynb             # Notebook for simulating data and evaluating methods
│-- beta_xi_conf.ipynb           # Notebook for confidence interval estimation of model parameters with IWPC dataset
│-- iwpc_prob.csv                # Dataset used for IWPC probability calculations in the analysis
│-- data.json                    # Precomputed results including mean and confidence intervals for optimized parameters
```

## Requirements

To ensure smooth execution, install the following dependencies:

```sh
pip install numpy pandas matplotlib plotly scipy scikit-learn imbalanced-learn optuna cma hyperopt numdifftools
```

## Key Features

- 📌 **Kernel Smoothing:** Implements various kernel smoothing functions, including the proposed repeated Nadaraya-Watson estimators.
- 🏆 **Optimization Methods:** Utilizes gradient-based and evolutionary algorithms (Hyperopt, CMA-ES, Differential Evolution) for parameter tuning.
- 🔍 **Cross-validation:** Implements cross-validation techniques for model selection and hyperparameter tuning.
- 📊 **Evaluation Metrics:** Computes classification and regression metrics, including ROC curves, precision-recall analysis, and C-index calculations.

## License

This repository is intended for anonymous review and is provided without an explicit license. Contact the authors for further details.

