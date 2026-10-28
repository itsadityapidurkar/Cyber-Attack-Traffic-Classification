#!/usr/bin/env python3
"""
train_ot_cps_models.py
Trains specialized Cyber-Physical Systems (CPS) & Operational Technology (OT)
models on the SWaT (Secure Water Treatment) and WADI (Water Distribution) datasets.
Exports serialized models and evaluates performance metrics.
"""

import os
import gc
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

OUTPUT_SWAT_MODEL = "/home/adityap/Desktop/PR/SI Lab/model_swat_ot.pkl"
OUTPUT_WADI_MODEL = "/home/adityap/Desktop/PR/SI Lab/model_wadi_ot.pkl"

def train_swat():
    print("--- Training SWaT (Secure Water Treatment) OT Anomaly Detector ---", flush=True)
    t0 = time.time()
    swat_path = "/home/adityap/Desktop/PR/SI Lab/data_swat/attack.csv"
    df = pd.read_csv(swat_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Label is 'Normal/Attack'
    target_col = 'Normal/Attack'
    feature_cols = [c for c in df.columns if c not in [target_col, 'Timestamp']]
    
    X = df[feature_cols].copy()
    y = df[target_col].map(lambda x: 'SWaT Attack' if 'Attack' in str(x) else 'Normal Baseline')
    
    # Convert numerical features
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"SWaT Model Trained in {time.time()-t0:.2f}s | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}", flush=True)
    
    package = {
        'model': rf,
        'dataset': 'iTrust SWaT (Secure Water Treatment System)',
        'domain': 'Critical Infrastructure / Municipal Water SCADA',
        'sensors_actuators': feature_cols,
        'metrics': {'accuracy': float(acc), 'f1_macro': float(f1)}
    }
    with open(OUTPUT_SWAT_MODEL, 'wb') as f:
        pickle.dump(package, f)
    print(f"Saved {OUTPUT_SWAT_MODEL} ({os.path.getsize(OUTPUT_SWAT_MODEL)/(1024*1024):.2f} MB)", flush=True)

def train_wadi():
    print("--- Training WADI (Water Distribution Testbed) OT Anomaly Detector ---", flush=True)
    t0 = time.time()
    wadi_path = "/home/adityap/Desktop/PR/SI Lab/data_wadi/WADI_attackdataLABLE.csv"
    df = pd.read_csv(wadi_path, skiprows=1, nrows=120000) # Balanced subset for memory safety
    df.columns = [c.strip() for c in df.columns]
    
    target_col = [c for c in df.columns if 'Attack LABLE' in c][0]
    feature_cols = [c for c in df.columns if c not in [target_col, 'Row', 'Date', 'Time']]
    
    X = df[feature_cols].copy()
    y = df[target_col].map(lambda x: 'WADI Cyber-Physical Attack' if str(x).strip() in ['-1', '-1.0'] else 'Normal Distribution')
    
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"WADI Model Trained in {time.time()-t0:.2f}s | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}", flush=True)
    
    package = {
        'model': rf,
        'dataset': 'iTrust WADI (Water Distribution Testbed)',
        'domain': 'Water Distribution Grid & Actuator Tampering',
        'sensors_actuators': feature_cols,
        'metrics': {'accuracy': float(acc), 'f1_macro': float(f1)}
    }
    with open(OUTPUT_WADI_MODEL, 'wb') as f:
        pickle.dump(package, f)
    print(f"Saved {OUTPUT_WADI_MODEL} ({os.path.getsize(OUTPUT_WADI_MODEL)/(1024*1024):.2f} MB)", flush=True)

if __name__ == "__main__":
    train_swat()
    train_wadi()
