"""
Extract transformer results from LOSO benchmark results.
Creates four arrays:
1. mean_r_true: mean correlation for each subject (actual models)
2. mean_r_null: mean correlation for each subject (null models)  
3. clf_acc_true: classification accuracy for each subject (actual models)
4. clf_acc_null: classification accuracy for each subject (null models)
"""

import json
import numpy as np
from pathlib import Path

# Base directory
BASE_DIR = "/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/benchmark/no_flash_removal/loso_out_transformer"

# All subjects
SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17"
]

def extract_transformer_metrics():
    """Extract all metrics and create the four arrays."""
    
    mean_r_true = []
    mean_r_null = []
    clf_acc_true = []
    clf_acc_null = []
    
    missing_files = []
    
    for subject in SUBJECTS:
        # File paths
        true_file = Path(BASE_DIR) / f"heldout_{subject}_results.json"
        null_file = Path(BASE_DIR) / f"heldout_{subject}_null_transformer_results.json"
        
        # Load true results
        try:
            with open(true_file, 'r') as f:
                true_data = json.load(f)
            true_metrics = true_data["Transformer_full"]
            mean_r_true.append(true_metrics["mean_r"])
            clf_acc_true.append(true_metrics["clf_acc"])
        except FileNotFoundError:
            print(f"Missing true file: {true_file}")
            missing_files.append(true_file)
            mean_r_true.append(np.nan)
            clf_acc_true.append(np.nan)
        except KeyError as e:
            print(f"Missing key in {true_file}: {e}")
            mean_r_true.append(np.nan)
            clf_acc_true.append(np.nan)
            
        # Load null results
        try:
            with open(null_file, 'r') as f:
                null_data = json.load(f)
            null_metrics = null_data["Transformer_full"]
            mean_r_null.append(null_metrics["mean_r"])
            clf_acc_null.append(null_metrics["clf_acc"])
        except FileNotFoundError:
            print(f"Missing null file: {null_file}")
            missing_files.append(null_file)
            mean_r_null.append(np.nan)
            clf_acc_null.append(np.nan)
        except KeyError as e:
            print(f"Missing key in {null_file}: {e}")
            mean_r_null.append(np.nan)
            clf_acc_null.append(np.nan)
    
    # Convert to numpy arrays
    mean_r_true = np.array(mean_r_true)
    mean_r_null = np.array(mean_r_null)
    clf_acc_true = np.array(clf_acc_true)
    clf_acc_null = np.array(clf_acc_null)
    
    return mean_r_true, mean_r_null, clf_acc_true, clf_acc_null, missing_files

def print_summary(mean_r_true, mean_r_null, clf_acc_true, clf_acc_null):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("TRANSFORMER LOSO RESULTS SUMMARY")
    print("="*60)
    print(f"Number of subjects: {len(SUBJECTS)}")
    
    # Remove NaN values for statistics
    mean_r_true_clean = mean_r_true[~np.isnan(mean_r_true)]
    mean_r_null_clean = mean_r_null[~np.isnan(mean_r_null)]
    clf_acc_true_clean = clf_acc_true[~np.isnan(clf_acc_true)]
    clf_acc_null_clean = clf_acc_null[~np.isnan(clf_acc_null)]
    
    print(f"\nMean Correlation (r):")
    print(f"  True models : {mean_r_true_clean.mean():.4f} ± {mean_r_true_clean.std():.4f} (n={len(mean_r_true_clean)})")
    print(f"  Null models : {mean_r_null_clean.mean():.4f} ± {mean_r_null_clean.std():.4f} (n={len(mean_r_null_clean)})")
    
    print(f"\nClassification Accuracy:")
    print(f"  True models : {clf_acc_true_clean.mean():.4f} ± {clf_acc_true_clean.std():.4f} (n={len(clf_acc_true_clean)})")
    print(f"  Null models : {clf_acc_null_clean.mean():.4f} ± {clf_acc_null_clean.std():.4f} (n={len(clf_acc_null_clean)})")
    
    print(f"\nPer-subject results:")
    print(f"{'Subject':<8} {'mean_r_true':<12} {'mean_r_null':<12} {'clf_acc_true':<13} {'clf_acc_null':<13}")
    print("-" * 70)
    for i, subj in enumerate(SUBJECTS):
        print(f"{subj:<8} {mean_r_true[i]:<12.4f} {mean_r_null[i]:<12.4f} "
              f"{clf_acc_true[i]:<13.4f} {clf_acc_null[i]:<13.4f}")

def main():
    """Main function."""
    print("Extracting transformer LOSO results...")
    
    mean_r_true, mean_r_null, clf_acc_true, clf_acc_null, missing_files = extract_transformer_metrics()
    
    if missing_files:
        print(f"\nWARNING: Missing {len(missing_files)} files:")
        for f in missing_files:
            print(f"  {f}")
    
    print_summary(mean_r_true, mean_r_null, clf_acc_true, clf_acc_null)
    
    # Save arrays
    np.save("transformer_mean_r_true.npy", mean_r_true)
    np.save("transformer_mean_r_null.npy", mean_r_null)  
    np.save("transformer_clf_acc_true.npy", clf_acc_true)
    np.save("transformer_clf_acc_null.npy", clf_acc_null)
    
    print(f"\nArrays saved:")
    print(f"  transformer_mean_r_true.npy - shape {mean_r_true.shape}")
    print(f"  transformer_mean_r_null.npy - shape {mean_r_null.shape}")
    print(f"  transformer_clf_acc_true.npy - shape {clf_acc_true.shape}")
    print(f"  transformer_clf_acc_null.npy - shape {clf_acc_null.shape}")
    
    return mean_r_true, mean_r_null, clf_acc_true, clf_acc_null

if __name__ == "__main__":
    mean_r_true, mean_r_null, clf_acc_true, clf_acc_null = main()