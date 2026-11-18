"""
PathSnap Model Training and Benchmarking:
Trains and compares 5 models from distinct algorithm families:
1. Logistic Regression (Linear/Probabilistic)
2. Decision Tree (Tree-based)
3. Random Forest (Ensemble)
4. HistGradientBoosting (Boosting)
5. Multi-Layer Perceptron (Neural Network)

Evaluates on:
- 5-Fold GroupKFold CV (by trip_id) Macro F1
- Observation-level Top-1 Accuracy on Student-Test
- Student-Test Macro F1
- Route Continuity
- Log Loss
- Training & Inference Time
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

PROCESSED_TRAIN = "/home/wingsfry/Projects/ML hackathon/outputs/processed_train_features.csv"
PROCESSED_TEST = "/home/wingsfry/Projects/ML hackathon/outputs/processed_test_features.csv"
OUTPUT_DIR = "/home/wingsfry/Projects/ML hackathon/outputs/models"
ARTIFACTS_DIR = "/home/wingsfry/Projects/ML hackathon/artifacts"

FEATURE_COLS = [
    "dist_to_segment",
    "projection_ratio",
    "heading_diff",
    "cos_heading",
    "segment_length",
    "gps_speed",
    "rank_by_dist",
    "dist_margin",
    "relative_dist_ratio"
]

def compute_top1_accuracy_and_continuity(df_test, preds_proba):
    """
    Computes observation-level Top-1 accuracy and route continuity on test data.
    """
    df_eval = df_test.copy()
    df_eval["score"] = preds_proba
    
    # For each observation, select candidate with highest score
    best_cands = df_eval.loc[df_eval.groupby(["trip_id", "obs_idx"])["score"].idxmax()]
    
    # Top-1 accuracy = fraction where label == 1
    top1_acc = best_cands["label"].mean()
    
    # Route continuity: fraction of consecutive observations where predicted arc is identical or continuous
    continuity_scores = []
    for tid, group in best_cands.groupby("trip_id"):
        sorted_group = group.sort_values("obs_idx")
        arcs_seq = sorted_group["arc_id"].values
        if len(arcs_seq) > 1:
            # Fraction where consecutive arcs remain on same road or advance
            same_or_step = np.mean(arcs_seq[1:] == arcs_seq[:-1])
            continuity_scores.append(same_or_step)
            
    avg_continuity = np.mean(continuity_scores) if continuity_scores else 0.0
    return float(top1_acc), float(avg_continuity), best_cands

def train_and_evaluate_models():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    print("Loading preprocessed training and testing data...")
    train_df = pd.read_csv(PROCESSED_TRAIN)
    test_df = pd.read_csv(PROCESSED_TEST)
    
    # Subsample training data to ensure clean balance and fast training (100k samples)
    if len(train_df) > 120000:
        pos_df = train_df[train_df["label"] == 1]
        neg_df = train_df[train_df["label"] == 0].sample(n=len(pos_df)*2, random_state=42)
        train_df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    groups_train = train_df["trip_id"].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    
    models = {
        "Logistic Regression": {
            "family": "Linear & Probabilistic",
            "config": "C=1.0, solver='lbfgs', max_iter=500",
            "model": LogisticRegression(C=1.0, max_iter=500, random_state=42),
            "use_scaled": True
        },
        "Decision Tree": {
            "family": "Tree-based",
            "config": "max_depth=10, min_samples_split=10",
            "model": DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42),
            "use_scaled": False
        },
        "Random Forest": {
            "family": "Ensemble",
            "config": "n_estimators=40, max_depth=12, n_jobs=-1",
            "model": RandomForestClassifier(n_estimators=40, max_depth=12, random_state=42, n_jobs=-1),
            "use_scaled": False
        },
        "HistGradientBoosting": {
            "family": "Boosting",
            "config": "max_iter=100, learning_rate=0.1, max_leaf_nodes=31",
            "model": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42),
            "use_scaled": False
        },
        "MLP Classifier": {
            "family": "Neural Networks",
            "config": "hidden=(64, 32), max_iter=25, early_stop=True",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=25, early_stopping=True, random_state=42),
            "use_scaled": True
        }
    }
    
    results = []
    best_top1 = 0.0
    best_model_name = ""
    best_model_obj = None
    
    # 5-Fold GroupKFold CV
    gkf = GroupKFold(n_splits=5)
    
    for name, info in models.items():
        print(f"\n--- Training {name} ({info['family']}) ---")
        clf = info["model"]
        X_tr = X_train_scaled if info["use_scaled"] else X_train
        X_te = X_test_scaled if info["use_scaled"] else X_test
        
        # 1. Cross-validation
        cv_f1s = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_tr, y_train, groups=groups_train)):
            clf_cv = clf.__class__(**clf.get_params())
            clf_cv.fit(X_tr[train_idx], y_train[train_idx])
            val_preds = clf_cv.predict(X_tr[val_idx])
            cv_f1s.append(f1_score(y_train[val_idx], val_preds, average="macro"))
            
        cv_macro_f1 = float(np.mean(cv_f1s))
        
        # 2. Fit full training data
        t0 = time.time()
        clf.fit(X_tr, y_train)
        train_time = round(time.time() - t0, 3)
        
        # 3. Test Evaluation
        t1 = time.time()
        test_preds = clf.predict(X_te)
        test_proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else test_preds.astype(float)
        inf_time = round((time.time() - t1) * 1000.0 / len(test_df) * 1000.0, 2) # ms per 1000 obs
        
        test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
        ll = float(log_loss(y_test, np.clip(test_proba, 1e-6, 1 - 1e-6)))
        top1_acc, continuity, _ = compute_top1_accuracy_and_continuity(test_df, test_proba)
        
        print(f"Results for {name}:")
        print(f"  CV Macro F1: {cv_macro_f1:.4f}")
        print(f"  Student-Test Macro F1: {test_macro_f1:.4f}")
        print(f"  Top-1 Accuracy: {top1_acc*100:.2f}%")
        print(f"  Route Continuity: {continuity*100:.2f}%")
        print(f"  Log Loss: {ll:.4f}")
        print(f"  Train Time: {train_time}s, Inf Time: {inf_time}ms/1k")
        
        results.append({
            "Model Name": name,
            "Algorithm Family": info["family"],
            "Key Configuration": info["config"],
            "CV Macro F1-score": round(cv_macro_f1, 4),
            "Top-1 Accuracy": f"{round(top1_acc * 100, 2)}%",
            "Route Continuity": f"{round(continuity * 100, 2)}%",
            "Student-Test Macro F1-score": round(test_macro_f1, 4),
            "Log Loss": round(ll, 4),
            "Training Time (s)": train_time,
            "Inference Time (ms/1k)": inf_time
        })
        
        if top1_acc > best_top1:
            best_top1 = top1_acc
            best_model_name = name
            best_model_obj = clf
            
    # Save comparison table
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved consolidated model comparison table to: {csv_path}")
    
    # Save best model
    best_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    joblib.dump(best_model_obj, best_path)
    print(f"Saved Best Model ({best_model_name}) to: {best_path}")
    
    # Generate Comparison Visualizations
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    models_names = [r["Model Name"] for r in results]
    cv_scores = [r["CV Macro F1-score"] for r in results]
    test_scores = [r["Student-Test Macro F1-score"] for r in results]
    
    x = np.arange(len(models_names))
    width = 0.35
    plt.bar(x - width/2, cv_scores, width, label="CV Macro F1", color="#3b82f6")
    plt.bar(x + width/2, test_scores, width, label="Student-Test Macro F1", color="#10b981")
    plt.xticks(x, [m.replace(" ", "\n") for m in models_names], fontsize=9)
    plt.ylabel("Macro F1-score")
    plt.title("Cross-Validation vs Student-Test Macro F1")
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    top1_vals = [float(r["Top-1 Accuracy"].replace("%", "")) for r in results]
    colors = ["#6366f1", "#ec4899", "#8b5cf6", "#06b6d4", "#f59e0b"]
    bars = plt.bar(models_names, top1_vals, color=colors)
    plt.xticks(x, [m.replace(" ", "\n") for m in models_names], fontsize=9)
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Observation-level Top-1 Accuracy on Student-Test")
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "model_comparison_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved comparison charts to: {chart_path}")

    return res_df

if __name__ == "__main__":
    train_and_evaluate_models()
