# Standard libraries
import time
import warnings
# Data handling and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import plotly.graph_objects as go
# Optimization modules
from scipy.stats import norm, binom
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import optuna
import cma
# Machine learning and evaluation
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (mean_squared_error, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score)
# Additional tools
import numdifftools as nd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, SparkTrials
from hyperopt import fmin as hyperopt_fmin
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

### Model Training Functions ###

def nw(x, X, Y, h, K=norm.pdf):
    """
    Kernel-weighted smoothing for a given target x.

    Args:
        x (np.ndarray): 1D array representing the target point.
        X (np.ndarray): 2D array of sample points.
        Y (np.ndarray): 1D or 2D array of responses corresponding to X.
        h (float): Bandwidth parameter.
        K (callable, optional): Kernel function to compute weights (default: norm.pdf).

    Returns:
        np.ndarray: The weighted sum of Y computed via kernel smoothing.
    """
    x = x.reshape(-1, 1) # Ensure x has two dimensions
    X_diff = (X - x.T) / h
    Kx = K(X_diff) / h
    W = Kx / Kx.sum(axis=1, keepdims=True)
    return np.dot(W, Y)

def high_dim_nw(ZZ, Y, h, K=norm.pdf):
    """
    High-dimensional kernel smoothing using broadcasting.

    Args:
        ZZ (np.ndarray): 1D array of values to compute pairwise differences.
        Y (np.ndarray): 1D array of response values corresponding to ZZ.
        h (float): Bandwidth parameter.
        K (callable, optional): Kernel function to compute weights (default: norm.pdf).

    Returns:
        np.ndarray: The kernel-smoothed response computed as the weighted sum of Y.
    """
    # Use broadcasting instead of np.tile for pairwise differences
    diff = (ZZ[:, None] - ZZ[None, :]) / h
    M = K(diff) / h
    np.fill_diagonal(M, 0)
    normalized_M = M / M.sum(axis=1, keepdims=True)
    return normalized_M @ Y

def stable_high_dim_nw(ZZ, Y, h, K=norm.pdf, eps=1e-10):
    """
    Compute normalized kernel weights for input data using the log-sum-exp trick for numerical stability.

    Args:
        ZZ (np.ndarray): 1D array of shape (n,) for computing pairwise differences.
        Y (np.ndarray): 1D array of responses of shape (n,).
        h (float): Bandwidth parameter.
        K (callable, optional): Kernel function, defaults to the Gaussian pdf.
        eps (float, optional): Small constant for numerical stability, defaults to 1e-10.

    Returns:
        np.ndarray: Weighted sum computed as the dot product of normalized weights and Y.
    """
    # Compute the pairwise differences in a vectorized way.
    diff = (ZZ[:, None] - ZZ[None, :]) / h
    
    # Compute log kernel values: note that for the Gaussian,
    # log(K(u)) = -0.5 * u**2 - 0.5*log(2*pi)
    log_K = -0.5 * diff**2 - 0.5 * np.log(2*np.pi)
    # Adjust for the scaling by h (since K(diff)/h is used)
    log_K = log_K - np.log(h)
    
    # Set the diagonal to -inf so it does not contribute to the sum.
    np.fill_diagonal(log_K, -np.inf)
    
    # Compute the log-sum-exp for each row.
    max_log_K = np.max(log_K, axis=1, keepdims=True)
    # Use max subtraction to stabilize the sum.
    sum_exp = np.sum(np.exp(log_K - max_log_K), axis=1, keepdims=True)
    log_denom = max_log_K + np.log(sum_exp + eps)  # add eps inside log for extra stability
    
    # Now compute the normalized weights
    log_weights = log_K - log_denom
    weights = np.exp(log_weights)
    
    return weights @ Y

def objective(XI, X, Y, T, h, K=norm.pdf):
    """
    Compute the kernel-smoothed objective value for parameter estimation.

    Args:
        XI (np.ndarray): Parameter vector for transforming features.
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Outcome vector.
        T (np.ndarray): Treatment vector.
        h (float): Bandwidth parameter for kernel smoothing.
        K (callable, optional): Kernel function (default: norm.pdf).

    Returns:
        float: Average squared error from the kernel-smoothed residuals.
    """
    n = X.shape[0]
    exi = X - high_dim_nw(np.dot(X, XI) - T, X, h, K)
    eyi = Y - high_dim_nw(np.dot(X, XI) - T, Y, h, K)
    beta = exi.T @ eyi / np.sum(exi ** 2, axis=0)
    ObjVal = np.sum((eyi - np.dot(exi, beta)) ** 2)
    return 1/n * ObjVal

def objective_lasso(XI, X, Y, T, h, lam, K=norm.pdf, weights = None):
    """
    Compute the Lasso-regularized kernel-smoothed objective value.

    Args:
        XI (np.ndarray): Parameter vector for feature transformation.
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Outcome vector.
        T (np.ndarray): Treatment vector.
        h (float): Bandwidth parameter for kernel smoothing.
        lam (float): Regularization parameter for Lasso.
        K (callable, optional): Kernel function (default: norm.pdf).
        weights (np.ndarray, optional): Weights for each observation; defaults to ones if not provided.

    Returns:
        float: Regularized objective value combining the weighted squared error and the L1 penalty.
    """
    n = X.shape[0]
    
    # Set default weights to ones if not provided
    if weights is None:
        weights = np.ones_like(Y)
    exi = X - high_dim_nw(np.dot(X, XI) - T, X, h, K)
    eyi = Y - high_dim_nw(np.dot(X, XI) - T, Y, h, K)
    beta = (exi.T @ eyi) / np.sum(exi ** 2, axis=0)
    
    # Incorporate weights into the objective value calculation
    weighted_errors = weights * (eyi - np.dot(exi, beta))
    ObjVal = np.sum(weighted_errors ** 2)
    
    # L1 penalty term for Lasso
    lasso_penalty = lam * np.sum(np.abs(XI))
    
    return 1/n * (ObjVal) + lasso_penalty

def cross_validate(X_train, T_train, Y_train, h_values, n_cv=5, K=norm.pdf):
    """
    Perform cross-validation over candidate bandwidth values and return the one that minimizes the mean squared error.

    Args:
        X_train (np.ndarray): Training feature matrix.
        T_train (np.ndarray): Training treatment vector.
        Y_train (np.ndarray): Training outcome vector.
        h_values (iterable): List or array of candidate bandwidth values.
        n_cv (int, optional): Number of cross-validation folds (default: 5).
        K (callable, optional): Kernel function to compute weights (default: norm.pdf).

    Returns:
        float: The optimal bandwidth (h) that minimizes the average mean squared error.
    """

    # Initialize the KFold cross-validator
    kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
    h_mses = {}
    # Loop over each h value
    for h in h_values:
        mses = []
        print(f"Looking at h={h} now...\n")
        # Loop over each fold
        for train_index, val_index in kf.split(X_train):
            # Split the data into training and validation for the current fold
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            T_train_fold, T_val_fold = T_train[train_index], T_train[val_index]
            Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[val_index]
            
            # Train the model on the current training set
            _, XI_opt_fold, BETA_opt_fold = train(X_train_fold, Y_train_fold, T_train_fold, h, K)
            
            # Define y function for the current training set and trained parameters
            def y_fold(x, t):
                x_array = np.array(x).reshape(1, -1)  
                nw_part = nw(np.dot(x_array, XI_opt_fold) - t, np.dot(X_train_fold, XI_opt_fold) - T_train_fold, Y_train_fold - np.dot(BETA_opt_fold, X_train_fold.T), h)
                return np.dot(BETA_opt_fold, x_array.T) + nw_part

            # Make predictions on the validation set
            Y_pred_fold = np.array([y_fold(x, t) for x, t in zip(X_val_fold, T_val_fold)])
            
            # #### To deal with NaN by masking ###
            # # Count and print the number of NaNs
            # nan_count = np.sum(np.isnan(Y_pred_fold))
            # print(f"Current h: {h}\n")
            # print(f"Number of NaNs masked in Y_pred_fold of: {nan_count} / {Y_pred_fold.shape[0]} \n")

            # # Mask the NaN values
            # Y_pred_fold[np.isnan(Y_pred_fold)] = 0
            # # Calculate the mean squared error for the current fold
            # mse = mean_squared_error(Y_val_fold, Y_pred_fold)
            #######################################


            #### To deal with NaN by ignoring ###
            ##Create a mask of non-NaN values in Y_pred_fold
            non_nan_mask = ~np.isnan(Y_pred_fold)
            Y_val_fold = Y_val_fold.reshape(-1)
            Y_pred_fold = Y_pred_fold.reshape(-1)
            non_nan_mask = non_nan_mask.reshape(-1)
            # Filter both Y_val_fold and Y_pred_fold using the non-NaN mask
            Y_val_fold_filtered = Y_val_fold[non_nan_mask]
            Y_pred_fold_filtered = Y_pred_fold[non_nan_mask]
            # Calculate the mean squared error using the filtered value            
            mse = mean_squared_error(Y_val_fold_filtered, Y_pred_fold_filtered)
            ####################################

            mses.append(mse)
        
        # Store the average MSE for this h value
        h_mses[h] = np.mean(mses)
    
    # Find the h with the smallest average MSE
    best_h = min(h_mses, key=h_mses.get)
    
    return best_h

def hyperopt_objective(args, X, Y, T, h, objective_func, lam, K, weights = None):
    """
    Compute the hyperopt objective by converting spherical coordinates to the XI parameter
    and evaluating the given objective function.

    Returns:
        dict: Dictionary with keys 'loss' (the computed loss), 'status' (STATUS_OK flag),
            'eval_time' (timestamp), 'XI' (transformed parameters as a list), and 'ObjVal' (loss value).
    """
    XI = np.array(spherical_to_cartesian(*args))
    loss = objective_func(XI, X, Y, T, h, lam, K, weights)
    return {'loss': loss, 'status': STATUS_OK, 'eval_time': time.time(), 'XI': XI.tolist(), 'ObjVal': loss}

def hyperopt_train(X_train, Y_train, T_train, h, objective_func, lam, K=norm.pdf, max_evals = 100, weights = None):
    """
    Perform hyperparameter optimization using hyperopt with spherical coordinates to determine
    the optimal XI and corresponding beta parameters.

    Args:
        X_train (np.ndarray): Training feature matrix.
        Y_train (np.ndarray): Training outcome vector.
        T_train (np.ndarray): Training treatment vector.
        h (float): Bandwidth parameter.
        objective_func (callable): Objective function to optimize.
        lam (float): Regularization parameter for Lasso.
        K (callable, optional): Kernel function (default: norm.pdf).
        max_evals (int, optional): Maximum number of evaluations for hyperopt (default: 100).
        weights (np.ndarray, optional): Observation weights; defaults to ones if None.

    Returns:
        tuple: A tuple (XI_best, beta, trials, best_loss) where:
            XI_best (np.ndarray): Optimized XI parameter vector.
            beta (np.ndarray): Corresponding beta parameter vector.
            trials (hyperopt.Trials): Hyperopt trials object containing optimization details.
            best_loss (float): Best loss value achieved.
    """

    # Define the search space for hyperopt using spherical coordinates
    dim = X_train.shape[1]
    # Restricting the range of the first angle to ensure positive first coordinate in XI
    angles_space = [hp.uniform(f'angle_0', 0, np.pi/2)]
    angles_space += [hp.uniform(f'angle_{i}', 0, np.pi) for i in range(1, dim-2)]
    angles_space += [hp.uniform('angle_last', 0, 2*np.pi)]
    # Create a Trials object
    trials = Trials()
    # spark_trials = SparkTrials(parallelism=4)
    # Use hyperopt to find the best XI values using the angles_space
    best = hyperopt_fmin(
            lambda args: hyperopt_objective(args, X_train, Y_train, T_train, h, objective_func, lam, K, weights),
            space=angles_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials)
    # Convert the result dictionary to a numpy array
    angles_best = [best[f'angle_{i}'] for i in range(dim-2)] + [best['angle_last']]
    XI_best = spherical_to_cartesian(*angles_best)
    exi = X_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, X_train, h, K)
    eyi = Y_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, Y_train, h, K)
    beta = exi.T @ eyi / np.sum(exi ** 2, axis=0)

    return np.array(XI_best), np.array(beta), trials, trials.best_trial['result']['loss']

def y(x, t, X_train, T_train, Y_train, XI_opt,BETA_opt, best_h):
    """
    Compute a single prediction via kernel smoothing using optimal parameters.

    Args:
        x (np.ndarray): Feature vector.
        t (float): Treatment value for the observation.
        X_train (np.ndarray): Training feature matrix.
        T_train (np.ndarray): Training treatment vector.
        Y_train (np.ndarray): Training outcome vector.
        XI_opt (np.ndarray): Optimized XI parameter vector.
        BETA_opt (np.ndarray): Optimized beta parameter vector.
        best_h (float): Bandwidth parameter.

    Returns:
        float: Predicted outcome.
    """

    x_array = np.array(x).reshape(1, -1)  # Ensuring x has shape (1, p), where p is the number of features
    nw_part = nw(np.dot(x_array, XI_opt) - t, np.dot(X_train, XI_opt) - T_train, Y_train - np.dot(X_train, BETA_opt), best_h)
    return np.dot(x_array, BETA_opt) + nw_part

def predict(XX, TT, X_train, T_train, Y_train, XI_opt,BETA_opt, best_h):
    """
    Predict outcomes for multiple observations using kernel smoothing.

    Args:
        XX (np.ndarray): Matrix of feature vectors for prediction.
        TT (np.ndarray): Vector of treatment values corresponding to XX.
        X_train (np.ndarray): Training feature matrix.
        T_train (np.ndarray): Training treatment vector.
        Y_train (np.ndarray): Training outcome vector.
        XI_opt (np.ndarray): Optimized XI parameter vector.
        BETA_opt (np.ndarray): Optimized beta parameter vector.
        best_h (float): Bandwidth parameter.

    Returns:
        np.ndarray: Array of predicted outcomes.
    """

    return np.array([y(x, t, X_train, T_train, Y_train, XI_opt,BETA_opt, best_h) for x, t in zip(XX, TT)])

def perform_cv(X_train, Y_train, T_train, weights = None, lambdas = np.logspace(-2, 1, 5), hs = np.linspace(0.2, 1.5, 5), n_splits = 3):
    """
    Perform cross-validation to select the optimal bandwidth and Lasso regularization parameters by minimizing the mean squared error.

    Args:
        X_train (np.ndarray): Training feature matrix.
        Y_train (np.ndarray): Training outcome vector.
        T_train (np.ndarray): Training treatment vector.
        weights (np.ndarray, optional): Observation weights (default: None).
        lambdas (iterable, optional): Candidate Lasso regularization parameters (default: np.logspace(-2, 1, 5)).
        hs (iterable, optional): Candidate bandwidth values (default: np.linspace(0.2, 1.5, 5)).
        n_splits (int, optional): Number of CV splits (default: 3).

    Returns:
        tuple: (best_h, best_lam, best_avg_score) where best_h is the optimal bandwidth, best_lam the optimal regularization parameter, and best_avg_score the corresponding average MSE.
    """

    best_lam = None
    best_h = None
    best_avg_score = float('inf')

    cv = KFold(n_splits=n_splits)

    # Dictionary to store the MSE scores for each (lam, h) pair
    mse_scores_dict = {}

    for train_idx, test_idx in cv.split(X_train):
        X_train_fold, X_test_fold = X_train[train_idx], X_train[test_idx]
        Y_train_fold, Y_test_fold = Y_train[train_idx], Y_train[test_idx]
        T_train_fold, T_test_fold = T_train[train_idx], T_train[test_idx]

        for lam in lambdas:
            for h in hs:
                print(f"looking at lam: {lam}, h: {h}")

                # Train model
                XI_opt, BETA_opt, _, _ = hyperopt_train(X_train_fold, Y_train_fold, T_train_fold, h, objective_func=objective_lasso, lam=lam, weights = weights)

                # Predict and calculate MSE on the test fold
                Y_pred = predict(X_test_fold, T_test_fold, X_train_fold, T_train_fold, Y_train_fold, XI_opt, BETA_opt, h)
                Y_pred[np.isnan(Y_pred)] = 0  # handle NaN predictions
                mse = mean_squared_error(Y_test_fold, Y_pred)
                
                # Record the MSE in the dictionary
                if (lam, h) not in mse_scores_dict:
                    mse_scores_dict[(lam, h)] = []
                mse_scores_dict[(lam, h)].append(mse)

    # Determine the (lam, h) pair with the lowest average MSE
    for (lam, h), scores in mse_scores_dict.items():
        avg_mse = np.mean(scores)
        if avg_mse < best_avg_score:
            best_avg_score = avg_mse
            best_lam = lam
            best_h = h
    # Return the best hyperparameters and their average MSE score
    return best_h, best_lam, best_avg_score

def iterative_feature_selection(X_train, Y_train, T_train, column_names, threshold=1e-4,lambdas = np.logspace(-2, 1, 5), hs = np.linspace(0.2, 1.5, 5),n_splits = 3):
    """
    Iteratively perform feature selection by cross-validating hyperparameters and dropping features with near-zero coefficients.

    Args:
        X_train (np.ndarray): Training feature matrix.
        Y_train (np.ndarray): Training outcome vector.
        T_train (np.ndarray): Training treatment vector.
        column_names (list or array-like): Names of features corresponding to columns in X_train.
        threshold (float, optional): Coefficient threshold below which a feature is dropped (default: 1e-4).
        lambdas (iterable, optional): Candidate Lasso regularization parameters (default: np.logspace(-2, 1, 5)).
        hs (iterable, optional): Candidate bandwidth values for kernel smoothing (default: np.linspace(0.2, 1.5, 5)).
        n_splits (int, optional): Number of cross-validation splits (default: 3).

    Returns:
        tuple: (XI_opt, BETA_opt, best_h, best_lam, best_loss, df, selected_features_mask) where:
            - XI_opt (np.ndarray): Final optimized XI parameter vector.
            - BETA_opt (np.ndarray): Final optimized beta parameter vector.
            - best_h (float): Selected optimal bandwidth.
            - best_lam (float): Selected optimal Lasso regularization parameter.
            - best_loss (float): Final loss value from the hyperopt training.
            - df (pd.DataFrame): DataFrame summarizing feature selection iterations.
            - selected_features_mask (np.ndarray): Boolean mask indicating the selected features from the original dataset.
    """

    iteration = 0
    feature_dropped = True
    all_results = []
    selected_features_mask = np.ones(X_train.shape[1], dtype=bool)

    while feature_dropped:
        iteration += 1
        print(f"---- Iteration {iteration} ----")
        # Cross-validation to select best_h and best_lam
        best_h, best_lam,mean_score = perform_cv(X_train, Y_train, T_train,lambdas,hs)
        # best_h, best_lam,mean_score = 0.4,0.01,0.2
        print(f"best_h:{best_h}, best_lam:{best_lam},mean_outer_score: {mean_score}")

        # Train using best_h and best_lam
        XI_opt, BETA_opt, _, best_loss = hyperopt_train(X_train, Y_train, T_train, best_h, objective_func=objective_lasso, lam=best_lam)
        # Check which features to drop
        zero_coefficients = np.abs(XI_opt) < threshold
        feature_dropped = np.any(zero_coefficients)
        selected_features_mask[selected_features_mask] = ~zero_coefficients

        if feature_dropped:
            # Storing results before filtering
            print("drop features:", np.array(column_names)[zero_coefficients])
            selected_features = np.array(column_names)[~zero_coefficients]
            all_results.append({
                'Feature Name': selected_features,
                'XI_opt': XI_opt[~zero_coefficients],
                'BETA_opt': BETA_opt[~zero_coefficients]
            })
            # Update data by removing dropped features
            print(zero_coefficients.dtype)
            print(np.all(np.isin(zero_coefficients, [True, False])))
            X_train = X_train[:, ~zero_coefficients]
            column_names = selected_features
            
        else:
            # If no features dropped in this iteration, store final results
            all_results.append({
                'Feature Name': column_names,
                'XI_opt': XI_opt,
                'BETA_opt': BETA_opt
            })
    # Combine all results into a single DataFrame
    iterations = [f"Iteration {i+1}" for i, _ in enumerate(all_results)]
    df = pd.concat([pd.DataFrame(res) for res in all_results], keys=iterations)
    return XI_opt,BETA_opt,best_h,best_lam,best_loss,df,selected_features_mask

### Optimization Functions ###

def calculate_heatmaps_hyperopt(X, T, Y, beta, xi, n, p, best_h, f0, max_evals = 200):
    """
    Fit model parameters via hyperopt and generate corresponding ground truth and estimand heatmaps.
    """
    # Split data into training and test sets
    Z = np.concatenate([X, T.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    Z_train, Z_test = train_test_split(Z, test_size=0.1, random_state=42)
    X_train = Z_train[:, :p]
    T_train = Z_train[:, p]
    Y_train = Z_train[:, p + 1]

    # Fit model and find optimal XI and BETA
    XI_opt, BETA_opt, _, _ = hyperopt_train(X_train, Y_train, T_train, best_h, objective_func=objective_lasso, lam=0.0, max_evals = max_evals)

    # Define the ranges for the heatmap
    arg1_range = np.linspace(-1.5, 1.5, 20)
    arg2_range = np.linspace(-1.5, 1.5, 20)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    # Function using the model
    def f(arg1, arg2):
        return arg1 + nw(arg2, np.dot(X_train, XI_opt) - T_train, Y_train - np.dot(X_train, BETA_opt), best_h)

    # Generate heatmap data
    Z_flat_f0 = np.array([f0(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f0 = Z_flat_f0.reshape(ARG1.shape)
    
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)

    return XI_opt, BETA_opt, Z_f0, Z_f

def optuna_objective(trial, X, Y, T, h, objective_func, lam, K, weights=None):
    """
	Compute the Optuna objective for hyperparameter tuning using spherical coordinates.
	"""

    p = X.shape[1]
    num_angles = p - 1  # number of angles needed
    # For the first angle, restrict to [0, pi/2] to enforce a positive first coordinate.
    angle0 = trial.suggest_uniform("angle_0", 0, np.pi/2)
    angles = [angle0]
    # The remaining angles in [0, pi]
    for i in range(1, num_angles):
        angles.append(trial.suggest_uniform(f"angle_{i}", 0, np.pi))
    XI = np.array(spherical_to_cartesian(*angles))
    loss = objective_func(XI, X, Y, T, h, lam, K, weights)
    return loss

def optuna_train(X_train, Y_train, T_train, h, objective_func, lam, K=norm.pdf, n_trials=100, weights=None):
    """
	Perform hyperparameter optimization using Optuna and spherical coordinates.
	"""

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: optuna_objective(trial, X_train, Y_train, T_train, h, objective_func, lam, K, weights),
                   n_trials=n_trials)
    
    best_trial = study.best_trial
    p = X_train.shape[1]
    num_angles = p - 1
    angles_best = [best_trial.params[f"angle_{i}"] for i in range(num_angles)]
    XI_best = spherical_to_cartesian(*angles_best)
    
    exi = X_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, X_train, h, K)
    eyi = Y_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, Y_train, h, K)
    beta = exi.T @ eyi / np.sum(exi ** 2, axis=0)
    
    return np.array(XI_best), np.array(beta), study, study.best_value

def calculate_heatmaps_optuna(X, T, Y, beta, xi, n, p, best_h, f0, n_trials=200):
    """
	Split data, optimize model parameters with Optuna, and compute corresponding heatmap grids.
    """
    # Split data into training and test sets.
    Z = np.concatenate([X, T.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    Z_train, _ = train_test_split(Z, test_size=0.1, random_state=42)
    X_train = Z_train[:, :p]
    T_train = Z_train[:, p]
    Y_train = Z_train[:, p + 1]

    # Fit model using optuna_train.
    XI_opt, BETA_opt, study, best_value = optuna_train(X_train, Y_train, T_train, best_h,
                                                       objective_func=objective_lasso, lam=0.0, 
                                                       n_trials=n_trials)
    # Define the ranges for the heatmap.
    arg1_range = np.linspace(-1.5, 1.5, 20)
    arg2_range = np.linspace(-1.5, 1.5, 20)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    
    # Define the function using the model.
    def f(arg1, arg2):
        return arg1 + nw(arg2, np.dot(X_train, XI_opt) - T_train, Y_train - np.dot(X_train, BETA_opt), best_h)

    # Generate heatmap data.
    Z_flat_f0 = np.array([f0(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f0 = Z_flat_f0.reshape(ARG1.shape)
    
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)

    return XI_opt, BETA_opt, Z_f0, Z_f

def de_objective(angles, X, Y, T, h, objective_func, lam, K, weights=None):
    """		
    Compute the Differential Evolution objective using spherical coordinates.
    """
    XI = np.array(spherical_to_cartesian(*angles))
    return objective_func(XI, X, Y, T, h, lam, K, weights)

def de_train(X_train, Y_train, T_train, h, objective_func, lam, K=norm.pdf):
    """
    Perform model fitting using Differential Evolution and return optimized parameters.
    """
    p = X_train.shape[1]
    num_angles = p - 1  # For a p-dimensional unit vector
    # Set bounds: first angle in [0, pi/2] to enforce XI[0] >= 0; others in [0, pi]
    bounds = [(0, np.pi/2)] + [(0, np.pi)] * (num_angles - 1)
    
    result = differential_evolution(
        de_objective,
        bounds,
        args=(X_train, Y_train, T_train, h, objective_func, lam, K),
        maxiter=100,
        polish=True
    )
    best_angles = result.x
    XI_best = np.array(spherical_to_cartesian(*best_angles))
    
    # Compute residuals and obtain beta via least-squares as before.
    exi = X_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, X_train, h, K)
    eyi = Y_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, Y_train, h, K)
    beta = exi.T @ eyi / np.sum(exi ** 2, axis=0)
    return XI_best, beta, result.fun

def calculate_heatmaps_de(X, T, Y, beta, xi, n, p, best_h, f0, maxiter=100):
    """
    Uses Differential Evolution to fit the model and computes heatmap grids.
    """
    # Split data into training and (unused) test parts.
    Z = np.concatenate([X, T.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    Z_train, _ = train_test_split(Z, test_size=0.1, random_state=42)
    X_train = Z_train[:, :p]
    T_train = Z_train[:, p]
    Y_train = Z_train[:, p+1]
    
    # Fit the model using DE.
    XI_opt, BETA_opt, best_loss = de_train(X_train, Y_train, T_train, best_h, objective_func=objective_lasso, lam=0.0, K=norm.pdf)
    
    # Define grid ranges for the heatmap.
    arg1_range = np.linspace(-1.5, 1.5, 20)
    arg2_range = np.linspace(-1.5, 1.5, 20)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    
    # Define the model function f using the fitted parameters.
    def f(arg1, arg2):
        return arg1 + nw(arg2, np.dot(X_train, XI_opt) - T_train, Y_train - np.dot(X_train, BETA_opt), best_h)
    
    # Evaluate f0 (the ground truth function) on the grid.
    Z_flat_f0 = np.array([f0(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f0 = Z_flat_f0.reshape(ARG1.shape)
    
    # Evaluate the model-based function f on the grid.
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)
    
    return XI_opt, BETA_opt, Z_f0, Z_f

def cma_objective(angles, X, Y, T, h, objective_func, lam, K, weights=None):
    """
    Compute the CMA-ES objective using spherical coordinates.
    """
    XI = np.array(spherical_to_cartesian(*angles))
    return objective_func(XI, X, Y, T, h, lam, K, weights)

def cma_train(X_train, Y_train, T_train, h, objective_func, lam, K=norm.pdf, sigma=0.1, maxiter=100):
    """
    Perform model fitting using CMA-ES and return optimized parameters.
    """
    p = X_train.shape[1]
    num_angles = p - 1
    # Starting guess: use pi/4 for all angles.
    x0 = np.full(num_angles, np.pi/4)
    # Set bounds: first angle [0, pi/2] and others [0, pi].
    lower_bounds = [0] + [0]*(num_angles - 1)
    upper_bounds = [np.pi/2] + [np.pi]*(num_angles - 1)
    opts = {
        'bounds': [lower_bounds, upper_bounds],
        'maxiter': maxiter,
        'verb_disp': 1,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    
    def wrapped_objective(angles):
        return cma_objective(angles, X_train, Y_train, T_train, h, objective_func, lam, K)
    
    es.optimize(wrapped_objective)
    best_angles = es.result.xbest
    XI_best = np.array(spherical_to_cartesian(*best_angles))
    
    exi = X_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, X_train, h, K)
    eyi = Y_train - high_dim_nw(np.dot(X_train, XI_best) - T_train, Y_train, h, K)
    beta = exi.T @ eyi / np.sum(exi ** 2, axis=0)
    return XI_best, beta, es.result.fbest

def calculate_heatmaps_cma(X, T, Y, beta, xi, n, p, best_h, f0, sigma=0.1, maxiter=100):
    """
    Uses CMA-ES to fit the model and computes heatmap grids.
    """
    Z = np.concatenate([X, T.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    Z_train, _ = train_test_split(Z, test_size=0.1, random_state=42)
    X_train = Z_train[:, :p]
    T_train = Z_train[:, p]
    Y_train = Z_train[:, p+1]
    
    XI_opt, BETA_opt, best_obj = cma_train(X_train, Y_train, T_train, best_h, objective_func=objective_lasso, lam=0.0, K=norm.pdf, sigma=sigma, maxiter=maxiter)
    
    arg1_range = np.linspace(-1.5, 1.5, 20)
    arg2_range = np.linspace(-1.5, 1.5, 20)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    
    def f(arg1, arg2):
        return arg1 + nw(arg2, np.dot(X_train, XI_opt) - T_train, Y_train - np.dot(X_train, BETA_opt), best_h)
    
    Z_flat_f0 = np.array([f0(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f0 = Z_flat_f0.reshape(ARG1.shape)
    
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)
    
    return XI_opt, BETA_opt, Z_f0, Z_f


### Data Extraction Functions ###
def extract_RotGBSG_df(df):
    """
    Extract RotGBSG data with logit-transformed survival probability.
    """
    X = np.array(df[["0", "1", "2", "6", "4", "5"]])
    T = np.array(df["3"]).reshape(-1, 1)
    Y_prob = np.array(df["survival_prob"]).reshape(-1, 1)
    Y = np.log((Y_prob + .000005) / (1.00005 - Y_prob))
    event_time = np.array(df["time"]).reshape(-1, 1)
    delta_reverse = 1 - np.array(df["delta"])
    return X, T, Y.reshape(-1,), Y_prob.reshape(-1,), delta_reverse.reshape(-1,), event_time

def extract_RotGBSG_df_wo_prob(df):
    """
    Extract RotGBSG data from a DataFrame without survival probability transformation.
    """
    X = np.array(df[["0", "1", "2", "6", "4", "5"]])
    T = np.array(df["3"]).reshape(-1, 1)
    event_time = np.array(df["time"]).reshape(-1, 1)
    delta_reverse = 1 - np.array(df["delta"])
    return X, T, delta_reverse.reshape(-1,), event_time

def extract_mimic(df, t_name):
    """
    Extract MIMIC survival data from a DataFrame.
    """
    T = np.array(df[t_name]).reshape(-1, 1)
    Y_prob = np.array(df["survival_prob"]).reshape(-1, 1)
    delta = np.array(df["day_28_flag"]).reshape(-1, 1)
    Y = np.log(Y_prob / (1.0 - Y_prob))
    # Drop the specified columns and then extract the rest
    X = df.drop(columns=[t_name, "survival_prob", "day_28_flag", "mort_day"]).values
    return X, T, Y, Y_prob, delta

### Plotting Functions ###
def heatmap_with_dots(X_test, XI_opt, T_test, BETA_opt, Y_test_class, X_train, T_train, Y_train, h, f):
    """Plot a heatmap with overlaid data points for classification."""

    coord1 = np.dot(X_test, BETA_opt)
    coord2 = (np.dot(X_test, XI_opt) - T_test)
    arg1_range = np.linspace(coord1.min(), coord1.max(), 50)
    arg2_range = np.linspace(coord2.min(), coord2.max(), 50)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    Z_flat_f = np.array([f(arg1, arg2,X_train, XI_opt, T_train, Y_train, BETA_opt, h) for arg1, arg2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    
    # Plotting the heatmap
    c2 = axs.imshow(Z_f, extent=[coord1.min(), coord1.max(), coord2.min(), coord2.max()], origin='lower', aspect='auto', cmap='inferno')
    fig.colorbar(c2, ax=axs)
    
    # Adding the contour where function value is 0
    cs = axs.contour(ARG1, ARG2, Z_f, levels=[0], colors='white', linestyles='dashed')
    axs.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
    
    # Compute the coordinates
    coord1 = np.dot(X_test, BETA_opt)
    coord2 = (np.dot(X_test, XI_opt) - T_test)
    
    # Filter and plot dots based on Y_test_class
    mask_class0 = Y_test_class == 0
    mask_class1 = Y_test_class == 1
    
    # Plot dots for class 0
    axs.scatter(coord1[mask_class0], coord2[mask_class0], c='red', label='Class 0', s=50, edgecolors='black')
    
    # Plot dots for class 1
    axs.scatter(coord1[mask_class1], coord2[mask_class1], c='blue', label='Class 1', s=50, edgecolors='black')
    
    # axs.set_title('Heatmap for f with Dots')
    axs.set_xlabel(r'$X^T \beta$',fontsize = 20)
    axs.set_ylabel(r'$X^T \xi - \tau$', fontsize = 20)
    
    # Show legend
    axs.legend(loc='upper right', fontsize = 12)
    
    plt.show()

def slider_plot(X_test_selected_final, XI_opt, T_test, BETA_opt, Y_test_class, X_train_selected_final, T_train, Y_train, best_h,f):
    """
    Display an interactive slider heatmap plot for visualizing model predictions.
    """
    score_1_test = np.dot(X_test_selected_final,BETA_opt)
    score_2_test = np.dot(X_test_selected_final, XI_opt) 
    Y_score_test = np.squeeze([f(a1,a2, X_train_selected_final, XI_opt, T_train, Y_train, BETA_opt, best_h) for a1,a2 in zip(score_1_test,(score_2_test - T_test))]) # this is equiv to Y_pred
    # Assuming you have your data loaded
    coord1 = score_2_test
    coord2 = T_test
    arg1_range = np.linspace(coord1.min(), coord1.max(), 50)
    arg2_range = np.linspace(coord2.min(), coord2.max(), 50)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    def heatmap_data(score_1_fixed_value):
        Z = np.array([f(score_1_fixed_value, arg1 - arg2,X_train_selected_final, XI_opt, T_train, Y_train, BETA_opt, best_h) for arg1, arg2 in zip(ARG1.ravel(), ARG2.ravel())])
        return Z.reshape(ARG1.shape)

    all_heatmaps = [heatmap_data(score_1_value) for score_1_value in np.arange(score_1_test.min(), score_1_test.max(), 1.0)]
    global_min = np.min(all_heatmaps)
    global_max = np.max(all_heatmaps)
    heatmap = heatmap_data(np.mean(score_1_test))
    fig = go.Figure(data=go.Heatmap(z=heatmap, x=arg1_range, y=arg2_range, colorscale="Inferno", zmin=global_min, zmax=global_max))
    # Add slider (remains unchanged from previous code)
    steps = []
    for score_1_value in np.arange(score_1_test.min(), score_1_test.max(), 1.0):
        step = dict(
            args=[{"z": [heatmap_data(score_1_value)], "colorscale": "Inferno", "zmin": global_min, "zmax": global_max}],
            label=str(round(score_1_value, 2)),
            method="restyle"
        )
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Score 1 Value:"},
        pad={"t": 50},
        steps=steps
    )]
    fig.update_layout(sliders=sliders)
    fig.show()

def plot_dist_comparison(A, B, col_names):
    """
    Compare distributions between two datasets with overlaid markers.
    """
    # Convert to pandas DataFrame if they are numpy arrays
    if isinstance(A, np.ndarray):
        A = pd.DataFrame(A, columns=col_names)
    if isinstance(B, np.ndarray):
        B = pd.DataFrame(B, columns=col_names)
    
    # Check if the input dataframes have the specified columns
    for col in col_names:
        if col not in A.columns or col not in B.columns:
            raise ValueError(f"Column {col} not found in the input dataframes.")
    
    # Determine the number of rows based on the length of col_names
    num_rows = int(np.ceil(len(col_names) / 4.0))

    # Plotting
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(24, num_rows * 4))
    
    # Make sure axes is always a 2D array
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]
    
    for i, column in enumerate(col_names):
        row_idx, col_idx = divmod(i, 4)
        ax = axes[row_idx, col_idx]
        
        A[column].hist(ax=ax, bins=30, alpha=0.75, label='A', color='blue')  # Increased bins to 30
        
        for val in B[column]:
            ax.axvline(x=val, color='red', alpha=0.5)
        
        ax.set_title(f'Distribution of {column} with overlay from B')
        ax.legend()

    # Hide any unused subplot axes
    for j in range(i+1, num_rows*4):
        row_idx, col_idx = divmod(j, 4)
        axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    plt.show()

def plot_scores_dist(score_1,score_2,T,labels):
    """
    Plot scatter distributions of scores and treatments, color-coded by class labels.
    """
    mask_0 = labels == 0
    mask_1 = labels == 1
    fig, ax = plt.subplots()
    plt.scatter(score_2[mask_0], T[mask_0], c=score_1[mask_0], marker='x', cmap='viridis', label='death')
    plt.scatter(score_2[mask_1], T[mask_1], c=score_1[mask_1], marker='o', cmap='viridis', label='survive')
    ax.set_xlabel("score_2")
    ax.set_ylabel("initiation")
    cbar = plt.colorbar()
    cbar.set_label('score_1')
    plt.legend()
    plt.show()

def surface_3d_plot(ARG1, ARG2, Z_f0, Z_f):
    """
    Render a 3D surface plot comparing ground truth and estimand surfaces.
    """
    # Plot both surfaces
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', frame_on=False)

    # Adjusting the gridlines, panes, and background
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')
    ax.xaxis.pane.set_facecolor((0.8, 0.8, 0.8, 1.0))
    ax.yaxis.pane.set_facecolor((0.8, 0.8, 0.8, 1.0))
    ax.zaxis.pane.set_facecolor((0.8, 0.8, 0.8, 1.0))
    
    ax.grid(True, color='k', linestyle=':', linewidth=0.5)

    # Plotting the surfaces with new colormaps
    f0_surface = ax.plot_surface(ARG1, ARG2, Z_f0, cmap='Blues', edgecolor='none', alpha=0.7)
    f_surface = ax.plot_surface(ARG1, ARG2, Z_f, cmap='Reds', edgecolor='none', alpha=0.6)

    # Creating proxies for the legend
    f0_proxy = mlines.Line2D([], [], color='dodgerblue', linewidth=2, label='Ground Truth')
    f_proxy = mlines.Line2D([], [], color='salmon', linewidth=2, label='Estimand')


    # Add the legend to the plot
    ax.legend(handles=[f0_proxy, f_proxy], loc='upper right')

    # Labeling axes
    ax.set_xlabel(r'$X^T \beta$', labelpad=10)
    ax.set_ylabel(r'$X^T \xi - \tau$', labelpad=10)
    ax.set_zlabel(r'$f(X^T \beta, X^T \xi - \tau)$', labelpad=10)
    # ax.set_title("Fancy 3D Surface Plot", pad=20)

    plt.show()

def surface_heatmap_plot(ARG1, ARG2, Z_f0, Z_f):
    """
    Display side-by-side heatmaps for ground truth and estimand surfaces.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate global min and max
    vmin_val = min(Z_f0.min(), Z_f.min())
    vmax_val = max(Z_f0.max(), Z_f.max())

    # Heatmap for Z_f0
    c1 = axs[0].imshow(Z_f0, extent=[ARG1.min(), ARG1.max(), ARG2.min(), ARG2.max()], origin='lower', aspect='auto', cmap='viridis', interpolation='bicubic', vmin=vmin_val, vmax=vmax_val)
    fig.colorbar(c1, ax=axs[0])
    axs[0].set_title('Heatmap for Ground Truth')
    axs[0].set_xlabel(r'$X^T \beta$')
    axs[0].set_ylabel(r'$X^T \xi - \tau$')

    # Heatmap for Z_f
    c2 = axs[1].imshow(Z_f, extent=[ARG1.min(), ARG1.max(), ARG2.min(), ARG2.max()], origin='lower', aspect='auto', cmap='viridis', interpolation='bicubic', vmin=vmin_val, vmax=vmax_val)
    fig.colorbar(c2, ax=axs[1])
    axs[1].set_title(r'Heatmap for Estimand')
    axs[1].set_xlabel(r'$X^T \beta$')
    axs[1].set_ylabel(r'$X^T \xi - \tau$')
    
    plt.tight_layout()
    plt.show()

def surface_3d_plot_single(f, X_test, XI_opt, T_test, BETA_opt):
    """
    Render a 3D surface plot for a single function.
    """
    coord1 = np.dot(BETA_opt, X_test.T)
    coord2 = (np.dot(X_test, XI_opt) - T_test)
    arg1_range = np.linspace(coord1.min(), coord1.max(), 50)
    arg2_range = np.linspace(coord2.min(), coord2.max(), 50)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    # Evaluate f on the grid
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(ARG1, ARG2, Z_f, cmap='viridis', edgecolor='none')
    fig.colorbar(surf)
    
    ax.set_title('3D Surface Plot for f')
    ax.set_xlabel(r'$X^T \beta$')
    ax.set_ylabel(r'$X^T \xi - \tau$')
    ax.set_zlabel('f Value')
    
    plt.show()

def surface_heatmap_plot_single(f, X_test, XI_opt, T_test, BETA_opt):
    """
    Display a heatmap for a single function based on kernel smoothing.
    """
    coord1 = np.dot(BETA_opt, X_test.T)
    coord2 = (np.dot(X_test, XI_opt) - T_test)
    arg1_range = np.linspace(coord1.min(), coord1.max(), 50)
    arg2_range = np.linspace(coord2.min(), coord2.max(), 50)
    ARG1, ARG2 = np.meshgrid(arg1_range, arg2_range)
    ARG1_flat = ARG1.ravel()
    ARG2_flat = ARG2.ravel()
    Z_flat_f = np.array([f(a1, a2) for a1, a2 in zip(ARG1_flat, ARG2_flat)])
    Z_f = Z_flat_f.reshape(ARG1.shape)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    
    c2 = axs.imshow(Z_f, extent=[coord1.min(), coord1.max(), coord2.min(), coord2.max()], origin='lower', aspect='auto', cmap='viridis')
    fig.colorbar(c2, ax=axs)
    
    axs.set_title('Heatmap for f')
    axs.set_xlabel(r'$X^T \beta$')
    axs.set_ylabel(r'$X^T \xi - \tau$')
    
    plt.show()

def plot_roc(Y_test_class, Y_pred_class):
    """
    Plot the ROC curve with AUC annotation.
    """
    fpr, tpr, threshold = roc_curve(Y_test_class, Y_pred_class)
    roc_auc = auc(fpr, tpr)
    print(f"AUC for our classifier is: {roc_auc}")
    # Plotting the ROC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def show_dist(data):
    """
    Display a histogram of the data distribution.
    """
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    plt.title('Distribution of the numpy array')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def show_dist_matrix(X, column_names=None):
    """
    Plot histograms for each column of a matrix with optional titles.
    """
    num_cols = X.shape[1]
    
    # Define the number of rows and columns for your subplots
    nrows = int(np.ceil(np.sqrt(num_cols)))
    ncols = int(np.ceil(num_cols / nrows))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    axes = axes.ravel()  # Flatten axes for easier indexing
    
    for i in range(num_cols):
        axes[i].hist(X[:, i], bins=30, density=True, alpha=0.6, color='g')
        
        # If column names are provided, use them as titles for the subplots
        if column_names is not None:
            axes[i].set_title(column_names[i])
        else:
            axes[i].set_title(f"Column {i+1}")
        
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    # Remove any remaining empty subplots
    for i in range(num_cols, nrows*ncols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def show_dist_comparison(X_original, X_imputed, column_names=None):
    """
    Compare distributions of original and imputed data using overlaid histograms.
    """
    num_cols = X_original.shape[1]
    
    # Define the number of rows and columns for your subplots
    nrows = int(np.ceil(np.sqrt(num_cols)))
    ncols = int(np.ceil(num_cols / nrows))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    axes = axes.ravel()  # Flatten axes for easier indexing
    
    for i in range(num_cols):
        # Plot the original data in green
        axes[i].hist(X_original[:, i], bins=30, density=True, alpha=0.5, color='g', label='Original')
        # Overlay the imputed data in red
        axes[i].hist(X_imputed[:, i], bins=30, density=True, alpha=0.5, color='r', label='Imputed')
        
        # If column names are provided, use them as titles for the subplots
        if column_names is not None:
            axes[i].set_title(column_names[i])
        else:
            axes[i].set_title(f"Column {i+1}")
        
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    # Remove any remaining empty subplots
    for i in range(num_cols, nrows*ncols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

### Evaluation Metrics Functions ###
def roc_result(Y_test_class, Y_pred_scores):
    # Calculate false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(Y_test_class, Y_pred_scores)
    
    # Compute the AUC (Area Under the Curve) score
    roc_auc = roc_auc_score(Y_test_class, Y_pred_scores)
    print(f"AUC for our classifier is: {roc_auc}")
    
    # Plotting the ROC
    plt.figure(figsize=(10, 6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    
    # Annotating some thresholds
    for i, thresh in enumerate(thresholds):
        if i % 50 == 0:  # plot every 50th threshold; this reduces clutter
            plt.annotate(f"{thresh:.2f}", (fpr[i], tpr[i]), fontsize=9, ha="right")
    
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def reg_result(A,B):
    plt.figure(figsize=(10, 6))
    plt.plot(A, label="Actual values")
    plt.plot(B, label="Predicted values")
    # plt.plot(delta_test, label="delta values")
    # plt.plot(Y_test_class, label="softlabel class values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Comparison of Actual and Predicted Values")
    plt.legend()
    plt.show()

def compute_c_index(y_true_class, y_pred_prob):
    n = len(y_true_class)
    concordant = 0
    permissible = 0
    tied = 0

    for i in range(n):
        for j in range(i+1, n):
            if y_true_class[i] != y_true_class[j]:
                permissible += 1
                if y_pred_prob[i] == y_pred_prob[j]:
                    tied += 1
                elif y_true_class[i] == 1 and y_pred_prob[i] > y_pred_prob[j]:
                    concordant += 1
                elif y_true_class[j] == 1 and y_pred_prob[j] > y_pred_prob[i]:
                    concordant += 1

    c_index = (concordant + 0.5 * tied) / permissible
    print("# permissible: ", permissible)
    return c_index

def cm_result(Y_test_class,Y_pred_class):
    print(confusion_matrix(Y_test_class, Y_pred_class))
    print(f"Precision Score: {precision_score(Y_test_class, Y_pred_class)}")
    print(f"Recall Score: {recall_score(Y_test_class, Y_pred_class)}")
    print(f"F1 Score: {f1_score(Y_test_class, Y_pred_class)}")
    # print(f"ROCAUC Score: {roc_auc_score(Y_test_class, Y_pred_scores)}")

def get_cindex(Y_test_class,Y_pred):
    Y_pred_prob = log_odds_to_prob(Y_pred)
    c_index_value = compute_c_index(Y_test_class, Y_pred_prob)
    return c_index_value

def cindex_result(Y_test_class,Y_pred):
    Y_pred_prob = log_odds_to_prob(Y_pred)
    c_index_value = compute_c_index(Y_test_class, Y_pred_prob)
    print("c_index_value: ", c_index_value)


### Helper Functions ###
def spherical_to_cartesian(*angles):
    """
    Convert angles in n-sphere to Cartesian coordinates.
    E.g. for 3D: r, theta, phi -> x, y, z
    """
    dim = len(angles) + 1
    cart_coords = [np.sin(angles[0])]
    
    for i in range(1, dim - 1):
        product = np.sin(angles[i])
        for j in range(i):
            product *= np.cos(angles[j])
        cart_coords.append(product)
    
    last_coord = 1
    for angle in angles:
        last_coord *= np.cos(angle)
    cart_coords.append(last_coord)
    
    return cart_coords

def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    std_err = np.std(data)
    ci = std_err * 1.96 / np.sqrt(n)  # 1.96 corresponds to 95% CI
    return m, m - ci, m + ci

def log_odds_to_prob(log_odds):
    odds = np.exp(log_odds)
    prob = odds / (1 + odds)
    return prob

def epanechnikov_kernel(u):
    abs_u = np.abs(u)
    mask = abs_u <= 1
    return 0.75 * (1 - u**2) * mask

def even_resample(X, T, Y, Y_prob, delta, num_bins=10, final_sample_size=None):
    df_resample = pd.DataFrame({
        'index': range(X.shape[0])  # Include an index
    })

    for i in range(X.shape[1]):
        df_resample[f'X_col_{i}'] = X[:, i]

    df_resample['T'] = T
    df_resample['Y'] = Y
    df_resample['Y_prob'] = Y_prob
    df_resample['delta'] = delta
    
    labels = range(num_bins)
    df_resample['T_bins'] = pd.cut(df_resample['T'], bins=num_bins, labels=labels)
    
    if final_sample_size is None:
        samples_per_bin = len(df_resample) // num_bins
    else:
        samples_per_bin = final_sample_size // num_bins
    
    subsamples = []
    for _, group in df_resample.groupby('T_bins'):
        if len(group) >= samples_per_bin:
            subsamples.append(group.sample(samples_per_bin, replace=False))
        else:
            subsamples.append(group.sample(samples_per_bin, replace=True))

    subsample_df = pd.concat(subsamples).sample(frac=1).reset_index(drop=True)
    subsample_df = subsample_df.drop(columns=['T_bins'])
    
    X_resampled = subsample_df.filter(like='X_col').values
    T_resampled = subsample_df['T'].values
    Y_resampled = subsample_df['Y'].values
    Y_prob_resampled = subsample_df['Y_prob'].values
    delta_resampled = subsample_df['delta'].values
    sampled_indices = subsample_df['index'].values  # Extract the indices of the sampled rows
    
    return X_resampled, T_resampled, Y_resampled, Y_prob_resampled, delta_resampled, sampled_indices

def smote_resample(X, T, Y, Y_prob, delta, num_bins=10):
    df_resample = pd.DataFrame({
        'index': range(X.shape[0])
    })

    for i in range(X.shape[1]):
        df_resample[f'X_col_{i}'] = X[:, i]

    df_resample['T'] = T
    df_resample['Y'] = Y
    df_resample['Y_prob'] = Y_prob
    df_resample['delta'] = delta
    
    labels = range(num_bins)
    df_resample['T_bins'] = pd.cut(df_resample['T'], bins=num_bins, labels=labels)
    df_resample['T_bins'] = df_resample['T_bins'].astype(int)  # Convert bins to integer type for SMOTE

    min_samples = df_resample['T_bins'].value_counts().min()
    k_neighbors = max(1, min_samples - 1)  # Ensure k is at least 1 and less than min_samples
    # sm = SMOTE(random_state=0, k_neighbors=k_neighbors)
    sm = SMOTEENN(smote=SMOTE(k_neighbors=k_neighbors), random_state=0)

    columns = [col for col in df_resample.columns if col != 'T_bins']
    
    X_resampled_full, _ = sm.fit_resample(df_resample[columns], df_resample['T_bins'])
    subsample_df = pd.DataFrame(X_resampled_full, columns=columns)

    X_resampled = subsample_df.filter(like='X_col').values
    T_resampled = subsample_df['T'].values
    Y_resampled = subsample_df['Y'].values
    Y_prob_resampled = subsample_df['Y_prob'].values
    delta_resampled = subsample_df['delta'].values
    sampled_indices = subsample_df['index'].values

    return X_resampled, T_resampled, Y_resampled, Y_prob_resampled, delta_resampled, sampled_indices

def psd_xi(xi, X, Y, T,best_h,XI_opt):
    Hessian_matrix = nd.Hessian(lambda xi: objective(xi, X, Y, T, best_h))(XI_opt)
    eigenvalues = np.linalg.eigvals(Hessian_matrix)
    is_positive_semi_definite = np.all(eigenvalues >= 0)
    return is_positive_semi_definite

