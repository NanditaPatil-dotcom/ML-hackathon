"""LEGACY: earlier Random Forest search (results in results/archive/bayesian/). The current
Bayesian optimisation is scripts/tune_catboost.py.

PathSnap Bayesian Hyperparameter Optimization:
Uses Optuna to optimize hyperparameters of the chosen top model (Random Forest)
using 5-Fold GroupKFold Cross-Validation on training trips.
"""

import os
import time
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

PROCESSED_TRAIN = "/home/wingsfry/Projects/ML hackathon/outputs/processed_train_features.csv"
PROCESSED_TEST = "/home/wingsfry/Projects/ML hackathon/outputs/processed_test_features.csv"
OUTPUT_DIR = "/home/wingsfry/Projects/ML hackathon/outputs/bayesian"
ARTIFACTS_DIR = "/home/wingsfry/Projects/ML hackathon/artifacts"

FEATURE_COLS = [
    "dist_to_segment", "projection_ratio", "heading_diff", "cos_heading",
    "segment_length", "gps_speed", "rank_by_dist", "dist_margin", "relative_dist_ratio"
]

def run_bayesian_optimization(n_trials=15):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    print("Loading datasets for Bayesian Hyperparameter Tuning...")
    train_df = pd.read_csv(PROCESSED_TRAIN)
    test_df = pd.read_csv(PROCESSED_TEST)
    
    # Balanced subsample for fast Bayesian search
    pos_df = train_df[train_df["label"] == 1]
    neg_df = train_df[train_df["label"] == 0].sample(n=len(pos_df)*2, random_state=42)
    sample_df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    X_train = sample_df[FEATURE_COLS].values
    y_train = sample_df["label"].values
    groups = sample_df["trip_id"].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values
    
    gkf = GroupKFold(n_splits=3)
    
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 30, 80, step=10)
        max_depth = trial.suggest_int("max_depth", 8, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 12)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 6)
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            clf.fit(X_train[train_idx], y_train[train_idx])
            preds = clf.predict(X_train[val_idx])
            scores.append(f1_score(y_train[val_idx], preds, average="macro"))
            
        return float(np.mean(scores))
        
    print(f"Starting Optuna Bayesian Search ({n_trials} trials)...")
    t0 = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    search_time = round(time.time() - t0, 2)
    
    best_params = study.best_params
    best_cv_f1 = study.best_value
    print(f"\nBayesian Optimization Complete in {search_time}s!")
    print(f"Best CV Macro F1: {best_cv_f1:.4f}")
    print(f"Best Hyperparameters: {best_params}")
    
    # Save search trials to CSV
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(OUTPUT_DIR, "bayesian_trials.csv")
    trials_df.to_csv(trials_csv, index=False)
    
    # Refit tuned model on entire train set
    print("Refitting tuned model on full dataset...")
    tuned_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    tuned_clf.fit(X_train, y_train)
    
    # Evaluate tuned model on student test set
    test_preds = tuned_clf.predict(X_test)
    test_proba = tuned_clf.predict_proba(X_test)[:, 1]
    
    test_macro_f1 = f1_score(y_test, test_preds, average="macro")
    test_ll = log_loss(y_test, np.clip(test_proba, 1e-6, 1 - 1e-6))
    
    test_df_eval = test_df.copy()
    test_df_eval["score"] = test_proba
    best_cands = test_df_eval.loc[test_df_eval.groupby(["trip_id", "obs_idx"])["score"].idxmax()]
    top1_acc = best_cands["label"].mean()
    
    print("\nTuned Model Student-Test Performance:")
    print(f"  Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"  Macro F1: {test_macro_f1:.4f}")
    print(f"  Log Loss: {test_ll:.4f}")
    
    # Save tuned model artifact
    tuned_model_path = os.path.join(ARTIFACTS_DIR, "tuned_rf_model.joblib")
    joblib.dump(tuned_clf, tuned_model_path)
    print(f"Saved Tuned Model to: {tuned_model_path}")
    
    # Plot Optimization History
    plt.figure(figsize=(8, 4))
    plt.plot(trials_df["number"], trials_df["value"], marker="o", color="#4f46e5", linewidth=2)
    plt.title("Bayesian Optimization Search History (Optuna TPE)")
    plt.xlabel("Trial Number")
    plt.ylabel("Validation Macro F1")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "bayesian_optimization_history.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved optimization history chart to: {chart_path}")
    
    return best_params

if __name__ == "__main__":
    run_bayesian_optimization(n_trials=12)
