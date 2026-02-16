import os
import glob
import pandas as pd
import numpy as np
import sys

# Add project root to sys.path to allow imports from backend module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.feature_extraction import extract_features

def build_dataset():
    print("=" * 60)
    print("  BUILDING CUSTOM VOICE DATASET")
    print("=" * 60)

    # Define paths
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(backend_dir, 'datasets', 'voice', 'raw')
    output_csv = os.path.join(backend_dir, 'datasets', 'voice', 'custom_data.csv')

    healthy_dir = os.path.join(raw_data_dir, 'healthy')
    parkinson_dir = os.path.join(raw_data_dir, 'parkinson')

    # Create directories if they don't exist
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(parkinson_dir, exist_ok=True)

    # Collect files
    healthy_files = glob.glob(os.path.join(healthy_dir, '*.wav'))
    parkinson_files = glob.glob(os.path.join(parkinson_dir, '*.wav'))

    print(f"Found {len(healthy_files)} healthy samples.")
    print(f"Found {len(parkinson_files)} parkinson samples.")

    if len(healthy_files) == 0 and len(parkinson_files) == 0:
        print("\n[WARNING] No .wav files found!")
        print(f"Please put healthy audio in:   {healthy_dir}")
        print(f"Please put parkinson audio in: {parkinson_dir}")
        return

    # Process files
    data = []
    
    # Process Healthy (Label 0)
    for file_path in healthy_files:
        print(f"Processing: {os.path.basename(file_path)} (Healthy)...", end=" ")
        features = extract_features(file_path)
        if np.all(features == 0):
             print("[FAILED]")
             continue
        
        row = list(features) + [0] # Append label 0
        data.append(row)
        print("[OK]")

    # Process Parkinson (Label 1)
    for file_path in parkinson_files:
        print(f"Processing: {os.path.basename(file_path)} (Parkinson)...", end=" ")
        features = extract_features(file_path)
        if np.all(features == 0):
             print("[FAILED]")
             continue
        
        row = list(features) + [1] # Append label 1
        data.append(row)
        print("[OK]")

    # Create DataFrame
    # Column names must match feature_extraction.py output + 'status'
    columns = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
        'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE', 
        'status'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"\nSaved dataset to {output_csv}")
    print(f"Total samples: {len(df)}")
    print(f"Healthy: {len(df[df['status']==0])}")
    print(f"Parkinson: {len(df[df['status']==1])}")

if __name__ == "__main__":
    build_dataset()
