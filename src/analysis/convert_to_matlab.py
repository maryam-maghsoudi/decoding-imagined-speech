"""
convert_to_matlab.py
====================
Converts scaling analysis summary.json files to MATLAB .mat format.

For each model, creates a .mat file containing mean_r and std_r per subject and k.

Usage:
    python convert_to_matlab.py
    python convert_to_matlab.py --models RNN_full CNN1D_windowed
    python convert_to_matlab.py --output_dir matlab_data
"""

import argparse
import json
import os
from typing import Dict, List
import numpy as np
from scipy.io import savemat


def find_model_dirs(base_dir: str = "scaling_out_no_flash") -> List[str]:
    """Find all model directories with summary.json files."""
    if not os.path.exists(base_dir):
        return []
    
    model_dirs = []
    for item in os.listdir(base_dir):
        model_path = os.path.join(base_dir, item)
        if os.path.isdir(model_path):
            summary_path = os.path.join(model_path, "summary.json")
            if os.path.exists(summary_path):
                model_dirs.append(item)
    
    return sorted(model_dirs)


def load_summary(model_key: str, base_dir: str = "scaling_out_no_flash") -> Dict:
    """Load summary.json for a specific model."""
    summary_path = os.path.join(base_dir, model_key, "summary.json")
    with open(summary_path, 'r') as f:
        return json.load(f)


def convert_model_to_matlab(model_key: str, summary_data: Dict, output_dir: str) -> None:
    """Convert one model's summary data to MATLAB format."""
    
    # Get all subjects and k values
    subjects = sorted(summary_data.keys())
    all_k_values = set()
    for subject_data in summary_data.values():
        all_k_values.update(int(k) for k in subject_data.keys())
    k_values = sorted(all_k_values)
    
    # Initialize arrays
    n_subjects = len(subjects)
    n_k = len(k_values)
    
    mean_r_matrix = np.full((n_subjects, n_k), np.nan)
    std_r_matrix = np.full((n_subjects, n_k), np.nan)
    n_combos_matrix = np.full((n_subjects, n_k), np.nan)
    
    # Fill arrays
    for i, subject in enumerate(subjects):
        subject_data = summary_data[subject]
        for j, k in enumerate(k_values):
            k_str = str(k)
            if k_str in subject_data:
                data = subject_data[k_str]
                mean_r_matrix[i, j] = data['mean_r_mean']
                std_r_matrix[i, j] = data['mean_r_std']
                n_combos_matrix[i, j] = data['n_combos']
    
    # Prepare data for MATLAB
    matlab_data = {
        'model_key': model_key,
        'subjects': subjects,
        'k_values': k_values,
        'mean_r': mean_r_matrix,
        'std_r': std_r_matrix,
        'n_combos': n_combos_matrix,
        'description': {
            'mean_r': 'Mean Pearson r across combinations (subjects x k)',
            'std_r': 'Standard deviation of Pearson r across combinations (subjects x k)', 
            'n_combos': 'Number of combinations tested (subjects x k)',
            'subjects': 'Subject IDs (rows of matrices)',
            'k_values': 'Training set sizes (columns of matrices)',
            'model_key': 'Model architecture and mode'
        }
    }
    
    # Save to .mat file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_key}_scaling_data.mat")
    savemat(output_path, matlab_data, oned_as='column')
    
    print(f"Saved {model_key}: {output_path}")
    print(f"  Shape: {n_subjects} subjects × {n_k} k-values")
    print(f"  Mean r range: [{np.nanmin(mean_r_matrix):.3f}, {np.nanmax(mean_r_matrix):.3f}]")
    print()


def main():
    parser = argparse.ArgumentParser(description="Convert scaling analysis to MATLAB format")
    parser.add_argument("--models", nargs="*", 
                        help="Specific models to convert (default: all found)")
    parser.add_argument("--base_dir", default="scaling_out_no_flash",
                        help="Base directory containing model results")
    parser.add_argument("--output_dir", default="matlab_scaling_data",
                        help="Output directory for .mat files")
    args = parser.parse_args()
    
    # Find available models
    available_models = find_model_dirs(args.base_dir)
    if not available_models:
        print(f"No model directories with summary.json found in {args.base_dir}")
        return
    
    # Determine which models to process
    if args.models:
        models_to_process = []
        for model in args.models:
            if model in available_models:
                models_to_process.append(model)
            else:
                print(f"Warning: Model '{model}' not found. Available: {available_models}")
        if not models_to_process:
            print("No valid models to process.")
            return
    else:
        models_to_process = available_models
    
    print(f"Found {len(available_models)} models: {available_models}")
    print(f"Processing {len(models_to_process)} models: {models_to_process}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Process each model
    for model_key in models_to_process:
        try:
            summary_data = load_summary(model_key, args.base_dir)
            convert_model_to_matlab(model_key, summary_data, args.output_dir)
        except Exception as e:
            print(f"Error processing {model_key}: {e}\n")
    
    print(f"Done. MATLAB files saved in {args.output_dir}/")


if __name__ == "__main__":
    main()