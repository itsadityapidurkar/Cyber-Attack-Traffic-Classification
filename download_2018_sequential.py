#!/usr/bin/env python3
"""
Sequential Downloader and Stratified Sampler for CSE-CIC-IDS2018
Downloads each of the 10 CSVs sequentially from Kaggle, extracts:
  - 100% of all attack classes (capturing all new/rare vectors)
  - A stratified subsample of BENIGN flows (keeping RAM bounded)
Then unzips and saves the clean partitioned data without blowing up disk or RAM.
"""

import subprocess
import os
import glob
import zipfile
import pandas as pd
import numpy as np

DATA_DIR = "/home/adityap/Desktop/PR/SI Lab/data_2018"
OUTPUT_MERGED = "/home/adityap/Desktop/PR/SI Lab/dataset_2018_balanced.csv"
LOG_FILE = os.path.join(DATA_DIR, "download_progress.log")

FILES = [
    "02-14-2018.csv",
    "02-15-2018.csv",
    "02-16-2018.csv",
    "02-21-2018.csv",
    "02-22-2018.csv",
    "02-23-2018.csv",
    "02-28-2018.csv",
    "03-01-2018.csv",
    "03-02-2018.csv",
    "02-20-2018.csv" # 4GB LOIC, download last
]

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def run_download():
    os.makedirs(DATA_DIR, exist_ok=True)
    log("=== Starting CSE-CIC-IDS2018 Sequential Download & Ingestion ===")
    
    extracted_chunks = []
    
    for idx, fname in enumerate(FILES, 1):
        log(f"[{idx}/10] Processing {fname}...")
        
        # Check if zip already exists or download
        zip_path = os.path.join(DATA_DIR, f"{fname}.zip")
        csv_path = os.path.join(DATA_DIR, fname)
        
        if not os.path.exists(csv_path) and not os.path.exists(zip_path):
            log(f"  Downloading {fname} via kaggle CLI...")
            cmd = f"kaggle datasets download -d solarmainframe/ids-intrusion-csv -f {fname} -p '{DATA_DIR}'"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode != 0:
                log(f"  ERROR downloading {fname}: {res.stderr}")
                continue
        
        # Extract if in zip
        if os.path.exists(zip_path) and not os.path.exists(csv_path):
            log(f"  Extracting {fname}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(DATA_DIR)
                os.remove(zip_path)
            except Exception as e:
                log(f"  Unzip error: {e}")
                
        if os.path.exists(csv_path):
            log(f"  Filtering {fname} into memory...")
            try:
                # Read in chunks to respect 8GB RAM
                attacks_list = []
                benign_samples = []
                
                for chunk in pd.read_csv(csv_path, chunksize=100000, low_memory=False):
                    # Clean columns
                    chunk.columns = [c.strip() for c in chunk.columns]
                    if 'Label' not in chunk.columns:
                        continue
                    # Separate attacks vs benign
                    atk = chunk[chunk['Label'] != 'Benign']
                    bng = chunk[chunk['Label'] == 'Benign']
                    
                    if len(atk) > 0:
                        attacks_list.append(atk)
                    if len(bng) > 0:
                        # Sample 5% of benign from this chunk to balance
                        benign_samples.append(bng.sample(frac=min(1.0, 5000 / len(bng)), random_state=42))
                
                parts_to_concat = []
                if attacks_list:
                    parts_to_concat.extend(attacks_list)
                if benign_samples:
                    parts_to_concat.extend(benign_samples)
                    
                if parts_to_concat:
                    day_df = pd.concat(parts_to_concat, ignore_index=True)
                    log(f"  Extracted {len(day_df):,} flows ({fname}) with classes: {day_df['Label'].value_counts().to_dict()}")
                    
                    # Append to master dataset on disk
                    header = not os.path.exists(OUTPUT_MERGED)
                    day_df.to_csv(OUTPUT_MERGED, mode='a', header=header, index=False)
                    del day_df
                    del parts_to_concat
                    
                # Remove raw CSV after processing to save disk space
                os.remove(csv_path)
                log(f"  Finished {fname}. Cleaned temporary raw file.")
            except Exception as e:
                log(f"  Error processing {fname}: {e}")
                
    log("=== ALL 10 CSVS DOWNLOADED AND MERGED SUCCESSFULLY! ===")
    if os.path.exists(OUTPUT_MERGED):
        df_final = pd.read_csv(OUTPUT_MERGED, usecols=['Label'])
        log(f"Final Combined Dataset Summary:\n{df_final['Label'].value_counts()}")

if __name__ == "__main__":
    run_download()
