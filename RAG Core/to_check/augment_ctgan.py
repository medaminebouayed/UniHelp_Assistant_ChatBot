import json
import pandas as pd
import numpy as np
import torch
# FIXED: Changed CTGANSynthesizer to CTGAN
from ctgan import CTGAN 
import os
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def check_cuda():
    """Check if GPU is accessible"""
    if torch.cuda.is_available():
        print(f"🚀 CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("⚠️ CUDA not available. Running on CPU.")
        return False

def load_chunks(files):
    """Load and flatten JSON chunks into a DataFrame"""
    all_records = []
    for file in files:
        if not os.path.exists(file):
            print(f"❌ File {file} not found. Skipping.")
            continue
        
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                metadata = item.get('metadata', {})
                record = {
                    'page_content': item.get('page_content', '')[:500],
                    'site': metadata.get('site', 'Unknown'),
                    'category': metadata.get('category', 'Unknown'),
                    'header': metadata.get('header', 'Unknown'),
                    'token_count': metadata.get('token_count', 0),
                    'chunk_length': metadata.get('chunk_length', 0)
                }
                all_records.append(record)
    
    return pd.DataFrame(all_records)

def train_and_sample(df, num_samples=200):
    """Train CTGAN and generate synthetic samples"""
    discrete_columns = ['site', 'category', 'header', 'page_content']
    
    print(f"📊 Training on {len(df)} records...")
    
    # FIXED: Using the CTGAN class
    # CTGAN (2026) handles CUDA internally if torch detects it
    synthesizer = CTGAN(
        epochs=300,
        batch_size=min(500, len(df)),
        verbose=True,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        cuda=torch.cuda.is_available() # Explicitly pass cuda status
    )
    
    synthesizer.fit(df, discrete_columns)
    
    print(f"🎲 Generating {num_samples} samples...")
    synthetic_data = synthesizer.sample(num_samples)
    return synthetic_data

def main():
    print(f"🐍 Python Version: {sys.version}")
    use_cuda = check_cuda()
    
    # Ensuring these files exist in your path
    chunk_files = ['iit_chunks.json', 'isb_chunks.json']
    df = load_chunks(chunk_files)
    
    if df.empty:
        print("❌ No data loaded. Check if .json files are in the directory.")
        return
    
    synthetic_df = train_and_sample(df, num_samples=200)
    
    output_file = 'synthetic_data_ctgan.json'
    synthetic_df.to_json(output_file, orient='records', force_ascii=False, indent=2)
    print(f"✅ CTGAN Augmentation complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()