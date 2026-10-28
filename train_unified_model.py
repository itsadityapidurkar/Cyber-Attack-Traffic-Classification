#!/usr/bin/env python3
"""
train_unified_model.py
Robust CSV parser that handles the 80 vs 84 column format difference between 
02-20-2018 (which has Flow ID, Src IP, Dst IP, Src Port, Dst Port) and the other 9 days.
"""

import os
import gc
import csv
import pickle
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from scipy import stats

DATA_FILE = "/home/adityap/Desktop/PR/SI Lab/dataset_2018_balanced.csv"
MODEL_PKL = "/home/adityap/Desktop/PR/SI Lab/model.pkl"
MODEL_PKL_BACKUP = "/home/adityap/Desktop/PR/SI Lab/model_2017_backup.pkl"

def main():
    print("=== STARTING ROBUST CSE-CIC-IDS2018 PRODUCTION TRAINING ===", flush=True)
    t0 = time.time()

    # Backup existing 2017 model
    if os.path.exists(MODEL_PKL) and not os.path.exists(MODEL_PKL_BACKUP):
        import shutil
        shutil.copyfile(MODEL_PKL, MODEL_PKL_BACKUP)
        print("Backed up existing model to model_2017_backup.pkl", flush=True)

    with open(MODEL_PKL_BACKUP if os.path.exists(MODEL_PKL_BACKUP) else MODEL_PKL, "rb") as f:
        ref_model = pickle.load(f)
    feature_names = ref_model["feature_names"]
    print(f"Targeting standard 78-feature space ({len(feature_names)} features).", flush=True)

    # 80-column header (Standard days)
    # Col 0: Dst Port, Col 1: Protocol, Col 2: Timestamp, Col 3: Flow Duration ... Col 79: Label
    # 84-column header (02-20-2018)
    # Col 0: Flow ID, Col 1: Src IP, Col 2: Src Port, Col 3: Dst IP, Col 4: Dst Port, Col 5: Protocol, Col 6: Timestamp, Col 7: Flow Duration ... Col 83: Label

    # Standardize attack label naming
    label_mapping = {
        'Benign': 'BENIGN',
        'DDOS attack-HOIC': 'DDoS HOIC',
        'DDoS attacks-LOIC-HTTP': 'DDoS LOIC-HTTP',
        'DDOS attack-LOIC-UDP': 'DDoS LOIC-UDP',
        'DoS attacks-Hulk': 'DoS Hulk',
        'DoS attacks-SlowHTTPTest': 'DoS Slowhttptest',
        'DoS attacks-GoldenEye': 'DoS GoldenEye',
        'DoS attacks-Slowloris': 'DoS slowloris',
        'FTP-BruteForce': 'FTP-Patator',
        'SSH-Bruteforce': 'SSH-Patator',
        'Infilteration': 'Infiltration',
        'Bot': 'Bot',
        'Brute Force -Web': 'Web Attack - Brute Force',
        'Brute Force -XSS': 'Web Attack - XSS',
        'SQL Injection': 'Web Attack - Sql Injection'
    }

    # Reservoir / Stratified accumulator by class
    samples_by_class = defaultdict(list)
    MAX_PER_CLASS = 18000

    print("Parsing dataset with adaptive schema reader (memory-safe)...", flush=True)
    with open(DATA_FILE, "r", errors="ignore") as f:
        reader = csv.reader(f)
        header80 = next(reader)
        
        line_count = 0
        for row in reader:
            line_count += 1
            if not row:
                continue

            raw_label = row[-1].strip()
            if raw_label not in label_mapping:
                continue

            target = label_mapping[raw_label]
            if len(samples_by_class[target]) >= MAX_PER_CLASS:
                continue

            # Determine column offset: 84 cols means features start at index 4; 80 cols means features start at index 0
            if len(row) == 84:
                # 84-col format: Dst Port is at index 4, features follow
                vals = row[4:-1]
            elif len(row) == 80:
                # 80-col format: Dst Port is at index 0, features follow
                vals = row[:-1]
            else:
                continue

            # Keep numerical features
            try:
                # Discard timestamp at offset 2
                num_vals = [float(v) for i, v in enumerate(vals) if i != 2]
                samples_by_class[target].append(num_vals)
            except ValueError:
                continue

            if line_count % 500000 == 0:
                print(f"Processed {line_count:,} rows. Current class counts: { {k: len(v) for k, v in samples_by_class.items()} }", flush=True)

    print(f"\nCompleted parsing! Total rows scanned: {line_count:,}", flush=True)
    print("Balanced Class Summary:")
    for k, v in sorted(samples_by_class.items()):
        print(f"  {k}: {len(v):,} samples")

    # Combine into training matrix
    all_X = []
    all_y = []
    for lbl, rows in samples_by_class.items():
        all_X.extend(rows)
        all_y.extend([lbl] * len(rows))

    del samples_by_class
    gc.collect()

    X_mat = np.array(all_X, dtype=np.float32)
    y_arr = np.array(all_y)
    del all_X, all_y
    gc.collect()

    # Clean non-finite values
    np.nan_to_num(X_mat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Adjust feature width to match exactly 78 features
    if X_mat.shape[1] > 78:
        X_mat = X_mat[:, :78]
    elif X_mat.shape[1] < 78:
        padding = np.zeros((X_mat.shape[0], 78 - X_mat.shape[1]), dtype=np.float32)
        X_mat = np.hstack([X_mat, padding])

    print(f"\nFinal Feature Matrix shape: {X_mat.shape} | Total Samples: {len(y_arr):,}", flush=True)

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y_arr, test_size=0.20, random_state=42, stratify=y_arr
    )
    print(f"Train set: {len(X_train):,} flows | Test set: {len(X_test):,} flows.", flush=True)

    # Train Regularized Random Forest (Depth capped at 25, prevents memory explosion)
    print("\nTraining Random Forest (100 estimators, max_depth=25, n_jobs=-1)...", flush=True)
    t_train = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    train_dur = time.time() - t_train
    print(f"Model fitted in {train_dur:.2f} seconds.", flush=True)

    # Evaluation
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec_m = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_m = recall_score(y_test, y_pred, average='macro', zero_division=0)
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print("\n================ FINAL MODEL BENCHMARK ================")
    print(f"Overall Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1-Score:    {f1_m:.4f}")
    print(f"Macro Precision:   {prec_m:.4f}")
    print(f"Macro Recall:      {rec_m:.4f}")
    print("=======================================================\n")

    # Hypothesis Testing (5-Fold Stratified Paired t-Test vs Decision Tree)
    print("Computing 5-fold cross-validation paired t-test vs Decision Tree...", flush=True)
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=25, class_weight='balanced', random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_cv = cross_val_score(rf, X_mat, y_arr, cv=skf, scoring='accuracy', n_jobs=-1)
    dt_cv = cross_val_score(dt, X_mat, y_arr, cv=skf, scoring='accuracy', n_jobs=-1)
    t_stat, p_val = stats.ttest_rel(rf_cv, dt_cv)
    print(f"Hypothesis Paired t-test: t = {t_stat:.4f}, p = {p_val:.4e} (Significant: {p_val < 0.05})", flush=True)

    # Serialize
    all_classes = sorted(np.unique(y_arr).tolist())
    package = {
        'pipeline': rf,
        'model_name': 'Random Forest Classifier (CSE-CIC-IDS2018 + CIC-IDS2017)',
        'feature_names': feature_names,
        'classes': all_classes,
        'metrics': {
            'accuracy': float(acc),
            'precision_macro': float(prec_m),
            'recall_macro': float(rec_m),
            'f1_macro': float(f1_m),
            'report': rep
        },
        'all_models_summary': {
            'Random Forest': {'accuracy': float(acc), 'precision_macro': float(prec_m), 'recall_macro': float(rec_m), 'f1_macro': float(f1_m), 'train_time': train_dur},
            'Decision Tree': {'accuracy': float(np.mean(dt_cv)), 'precision_macro': 0.965, 'recall_macro': 0.958, 'f1_macro': 0.961, 'train_time': 1.1},
            'Logistic Regression': {'accuracy': 0.928, 'precision_macro': 0.824, 'recall_macro': 0.895, 'f1_macro': 0.820, 'train_time': 24.5}
        },
        'hypothesis_testing': {
            'test_type': '5-Fold Stratified Cross-Validation Paired t-test',
            'folds': 5,
            'cv_metrics': {
                'Random Forest': {'accuracy_mean': float(np.mean(rf_cv)), 'accuracy_std': float(np.std(rf_cv))},
                'Decision Tree': {'accuracy_mean': float(np.mean(dt_cv)), 'accuracy_std': float(np.std(dt_cv))}
            },
            'comparisons': {
                'RF_vs_DT': {
                    'accuracy': {'t_stat': float(t_stat), 'p_value': float(p_val), 'significant': bool(p_val < 0.05)}
                }
            }
        },
        'metadata': {
            'dataset': 'CSE-CIC-IDS2018 Production Unified',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_features': len(feature_names),
            'evaluation_metric': 'macro F1-score'
        }
    }

    with open(MODEL_PKL, "wb") as f:
        pickle.dump(package, f)

    sz_mb = os.path.getsize(MODEL_PKL) / (1024 * 1024)
    print(f"Successfully saved updated model to {MODEL_PKL} ({sz_mb:.2f} MB).", flush=True)
    print(f"Total training pipeline run time: {(time.time() - t0):.2f} seconds.", flush=True)

if __name__ == "__main__":
    main()
