"""
run_full_evaluation.py
======================
Comprehensive evaluation script that runs all combinations of:
- Heldout subjects
- Img→lis mapping models  
- Text encoders

For each combination, evaluates imagined MEG and saves results.

Usage:
    python run_full_evaluation.py

Output structure:
    ./full_eval_results/
    ├── sub-01/
    │   ├── CNN1D/
    │   │   ├── bert/
    │   │   │   ├── eval_img_metrics.json
    │   │   │   ├── ranks_eval_img.npy
    │   │   │   └── per_word_stats.json
    │   │   ├── whisper/
    │   │   ├── wav2vec/
    │   │   └── bert_wav2vec/
    │   ├── ShallowMLP/
    │   └── ...
    ├── sub-03/
    └── ...
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import time

# =============================================================================
#  CONFIGURATION
# =============================================================================

# All available subjects
ALL_SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]

# All available text encoders
ALL_ENCODERS = ["bert", "whisper", "wav2vec", "bert_wav2vec"]

# All available img→lis architectures
ALL_ARCHITECTURES = ["CNN1D", "ShallowMLP", "UNet1D", "RNN", "TCN", "LinearLag"]

# Path templates (adjust these to match your setup)
BENCHMARK_DIR = "/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/benchmark/no_flash_removal/loso_out"
CHECKPOINT_TEMPLATE = "{benchmark_dir}/models/heldout_{subject}/{arch}_full.pt"
LINEARLAG_TEMPLATE = "{benchmark_dir}/models/heldout_{subject}/LinearLag_W.npy"

# Output directory for results
RESULTS_DIR = "./full_eval_results"
COMPARE_SCRIPT = "contrastive_word_meg_compare.py"

# Logging
LOG_FILE = os.path.join(RESULTS_DIR, "evaluation_log.txt")


def log_message(message: str, print_console: bool = True) -> None:
    """Log message to file and optionally console."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    if print_console:
        print(log_entry)
    
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")


def find_available_checkpoints(
    subjects: List[str],
    architectures: List[str],
    benchmark_dir: str = BENCHMARK_DIR,
) -> Dict[str, Dict[str, str]]:
    """
    Find all available img→lis model checkpoints.
    
    Returns
    -------
    dict : {subject: {arch: checkpoint_path}}
    """
    available = {}
    
    for subject in subjects:
        available[subject] = {}
        for arch in architectures:
            if arch == "LinearLag":
                # LinearLag uses .npy files instead of .pt files
                ckpt_path = LINEARLAG_TEMPLATE.format(
                    benchmark_dir=benchmark_dir,
                    subject=subject
                )
            else:
                ckpt_path = CHECKPOINT_TEMPLATE.format(
                    benchmark_dir=benchmark_dir,
                    subject=subject,
                    arch=arch
                )
            
            if os.path.exists(ckpt_path):
                available[subject][arch] = ckpt_path
            else:
                log_message(f"  Missing: {ckpt_path}", print_console=False)
    
    return available


def check_encoder_models(encoders: List[str], models_dir: str = "./compare_out/models") -> List[str]:
    """Check which encoder models are trained and available."""
    available = []
    
    for encoder in encoders:
        meg_path = os.path.join(models_dir, encoder, "meg_encoder.pt")
        txt_path = os.path.join(models_dir, encoder, "text_encoder.pt")
        
        if os.path.exists(meg_path) and os.path.exists(txt_path):
            available.append(encoder)
        else:
            log_message(f"  Missing encoder models: {encoder}")
    
    return available


def run_evaluation(
    subject: str,
    arch: str,
    ckpt_path: str,
    encoders: List[str],
    timeout_minutes: int = 30,
) -> Dict[str, bool]:
    """
    Run evaluation for one (subject, arch) combination with all encoders.
    
    Returns
    -------
    dict : {encoder: success_bool}
    """
    results = {}
    
    # Create output directory for this combination
    output_dir = os.path.join(RESULTS_DIR, subject, arch)
    os.makedirs(output_dir, exist_ok=True)
    
    log_message(f"  Running: {subject}/{arch} with encoders {encoders}")
    
    # Build command
    cmd = [
        sys.executable, COMPARE_SCRIPT,
        "--phase", "eval_img",
        "--encoders"] + encoders + [
        "--img_lis_ckpt", ckpt_path,
        "--img_lis_arch", arch,
        "--heldout_subject", subject,
    ]
    
    try:
        # Run the evaluation
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
        )
        
        if result.returncode == 0:
            log_message(f"    SUCCESS: {subject}/{arch}")
            
            # Move results to organized directory structure
            import shutil
            compare_results_dir = "./compare_out/results"
            compare_comparison_dir = "./compare_out/comparison"
            
            for encoder in encoders:
                # Copy individual encoder results
                src_dir = os.path.join(compare_results_dir, encoder)
                dst_dir = os.path.join(output_dir, encoder)
                
                if os.path.exists(src_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                    
                    # Copy relevant files
                    eval_files = [
                        f"eval_img_{subject}_metrics.json",
                        f"ranks_eval_img_{subject}.npy",
                        f"per_word_stats_eval_img_{subject}.json",
                        f"ranks_eval_img_{subject}_top20.npy",
                        f"word_labels_eval_img_{subject}_top20.npy",
                        f"top20_words_eval_img_{subject}.json"
                    ]
                    
                    for filename in eval_files:
                        src_file = os.path.join(src_dir, filename)
                        dst_file = os.path.join(dst_dir, filename.replace(f"_{subject}", ""))
                        
                        if os.path.exists(src_file):
                            shutil.copy2(src_file, dst_file)
                    
                    results[encoder] = True
                else:
                    results[encoder] = False
            
            # Copy architecture-specific comparison plots and summaries
            # These are now saved in subdirectories by architecture
            if arch == "LinearLag":
                arch_mode = "LinearLag"
            else:
                arch_mode = f"{arch}_full"  # Assume full mode, update if windowed detection needed
                if "windowed" in ckpt_path:
                    arch_mode = f"{arch}_windowed"
            
            comparison_src_dir = os.path.join(compare_comparison_dir, arch_mode)
            comparison_dst_dir = os.path.join(output_dir, "comparison")
            
            if os.path.exists(comparison_src_dir):
                os.makedirs(comparison_dst_dir, exist_ok=True)
                
                # Copy all comparison files for this architecture
                for item in os.listdir(comparison_src_dir):
                    src_file = os.path.join(comparison_src_dir, item)
                    dst_file = os.path.join(comparison_dst_dir, item)
                    
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)
                        
                log_message(f"    Copied comparison plots for {arch_mode}")
        else:
            log_message(f"    FAILED: {subject}/{arch}")
            log_message(f"    stdout: {result.stdout[-500:]}")  # Last 500 chars
            log_message(f"    stderr: {result.stderr[-500:]}")
            for encoder in encoders:
                results[encoder] = False
    
    except subprocess.TimeoutExpired:
        log_message(f"    TIMEOUT: {subject}/{arch} (>{timeout_minutes}min)")
        for encoder in encoders:
            results[encoder] = False
    except Exception as e:
        log_message(f"    ERROR: {subject}/{arch} - {e}")
        for encoder in encoders:
            results[encoder] = False
    
    return results


def aggregate_results_for_plotting(results_dir: str = RESULTS_DIR) -> None:
    """Aggregate all rank data for easy plotting across subjects and architectures."""
    import numpy as np
    
    log_message("Aggregating results for plotting across all architectures...")
    
    # Structure: {arch: {encoder: {subject: ranks_array}}}
    aggregated = {}
    
    # Scan all result files
    for subject_dir in os.listdir(results_dir):
        subject_path = os.path.join(results_dir, subject_dir)
        if not os.path.isdir(subject_path) or subject_dir.startswith('.'):
            continue
        
        for arch_dir in os.listdir(subject_path):
            arch_path = os.path.join(subject_path, arch_dir)
            if not os.path.isdir(arch_path):
                continue
            
            if arch_dir not in aggregated:
                aggregated[arch_dir] = {}
            
            for encoder_dir in os.listdir(arch_path):
                encoder_path = os.path.join(arch_path, encoder_dir)
                if not os.path.isdir(encoder_path):
                    continue
                
                if encoder_dir not in aggregated[arch_dir]:
                    aggregated[arch_dir][encoder_dir] = {}
                
                # Load ranks file if it exists
                ranks_file = os.path.join(encoder_path, "ranks_eval_img.npy")
                if os.path.exists(ranks_file):
                    try:
                        ranks = np.load(ranks_file)
                        aggregated[arch_dir][encoder_dir][subject_dir] = ranks
                    except Exception as e:
                        log_message(f"    Warning: Could not load {ranks_file}: {e}")
    
    # Save aggregated data for each architecture
    for arch, arch_data in aggregated.items():
        arch_output_dir = os.path.join(results_dir, "aggregated", arch)
        os.makedirs(arch_output_dir, exist_ok=True)
        
        for encoder, encoder_data in arch_data.items():
            if not encoder_data:  # Skip if no data
                continue
                
            # Combine all subjects' ranks for this (arch, encoder) pair
            all_ranks = []
            subject_info = {}
            
            for subject, ranks in encoder_data.items():
                all_ranks.append(ranks)
                subject_info[subject] = {
                    "n_samples": len(ranks),
                    "median_rank": int(np.median(ranks)),
                    "R@1": float((ranks <= 1).mean()),
                    "R@5": float((ranks <= 5).mean()),
                    "R@10": float((ranks <= 10).mean()),
                    "MRR": float((1.0 / ranks).mean()),
                }
            
            # Concatenate all ranks across subjects
            combined_ranks = np.concatenate(all_ranks) if all_ranks else np.array([])
            
            # Save combined ranks for plotting
            np.save(os.path.join(arch_output_dir, f"{encoder}_ranks_all_subjects.npy"), 
                   combined_ranks)
            
            # Save metadata
            metadata = {
                "architecture": arch,
                "encoder": encoder,
                "total_samples": len(combined_ranks),
                "n_subjects": len(subject_info),
                "subjects": list(subject_info.keys()),
                "per_subject": subject_info,
                "combined_metrics": {
                    "R@1": float((combined_ranks <= 1).mean()) if len(combined_ranks) > 0 else 0,
                    "R@5": float((combined_ranks <= 5).mean()) if len(combined_ranks) > 0 else 0,
                    "R@10": float((combined_ranks <= 10).mean()) if len(combined_ranks) > 0 else 0,
                    "MRR": float((1.0 / combined_ranks).mean()) if len(combined_ranks) > 0 else 0,
                    "median_rank": int(np.median(combined_ranks)) if len(combined_ranks) > 0 else 0,
                }
            }
            
            with open(os.path.join(arch_output_dir, f"{encoder}_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
    
    # Create plotting script
    plotting_script = os.path.join(results_dir, "plot_rank_curves.py")
    with open(plotting_script, "w") as f:
        f.write('''"""
Auto-generated plotting script for rank curves.
Usage: python plot_rank_curves.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "."
ARCHITECTURES = ["CNN1D", "ShallowMLP", "UNet1D", "RNN", "TCN"]
ENCODERS = ["bert", "whisper", "wav2vec", "bert_wav2vec"]
COLORS = {"bert": "#4C72B0", "whisper": "#DD8452", "wav2vec": "#55A868", "bert_wav2vec": "#C44E52"}

def plot_arch_comparison():
    """Plot rank CDF curves comparing encoders within each architecture."""
    for arch in ARCHITECTURES:
        arch_dir = os.path.join("aggregated", arch)
        if not os.path.exists(arch_dir):
            continue
            
        plt.figure(figsize=(10, 6))
        
        for encoder in ENCODERS:
            ranks_file = os.path.join(arch_dir, f"{encoder}_ranks_all_subjects.npy")
            meta_file = os.path.join(arch_dir, f"{encoder}_metadata.json")
            
            if os.path.exists(ranks_file) and os.path.exists(meta_file):
                ranks = np.load(ranks_file)
                with open(meta_file) as f:
                    meta = json.load(f)
                
                if len(ranks) > 0:
                    # CDF calculation
                    max_rank = min(50, ranks.max())  # Show up to rank 50
                    x = np.arange(1, max_rank + 1)
                    cdf = [(ranks <= r).mean() for r in x]
                    
                    plt.plot(x, cdf, 
                            label=f'{encoder} (n={meta["total_samples"]}, R@1={meta["combined_metrics"]["R@1"]:.3f})',
                            color=COLORS.get(encoder, "#888"), linewidth=2.5)
        
        plt.xlabel("Rank k")
        plt.ylabel("P(rank ≤ k)")
        plt.title(f"Rank CDF Comparison - {arch} (All Subjects Combined)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(1, 50)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"rank_cdf_{arch.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: rank_cdf_{arch.lower()}.png")

def plot_encoder_comparison():
    """Plot rank CDF curves comparing architectures within each encoder."""
    for encoder in ENCODERS:
        plt.figure(figsize=(10, 6))
        
        for arch in ARCHITECTURES:
            arch_dir = os.path.join("aggregated", arch)
            ranks_file = os.path.join(arch_dir, f"{encoder}_ranks_all_subjects.npy")
            meta_file = os.path.join(arch_dir, f"{encoder}_metadata.json")
            
            if os.path.exists(ranks_file) and os.path.exists(meta_file):
                ranks = np.load(ranks_file)
                with open(meta_file) as f:
                    meta = json.load(f)
                
                if len(ranks) > 0:
                    max_rank = min(50, ranks.max())
                    x = np.arange(1, max_rank + 1)
                    cdf = [(ranks <= r).mean() for r in x]
                    
                    plt.plot(x, cdf, 
                            label=f'{arch} (n={meta["total_samples"]}, R@1={meta["combined_metrics"]["R@1"]:.3f})',
                            linewidth=2.5)
        
        plt.xlabel("Rank k")
        plt.ylabel("P(rank ≤ k)")
        plt.title(f"Rank CDF Comparison - {encoder} Encoder (All Subjects Combined)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(1, 50)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"rank_cdf_{encoder}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: rank_cdf_{encoder}.png")

if __name__ == "__main__":
    print("Generating rank curve plots...")
    plot_arch_comparison()
    plot_encoder_comparison()
    print("Done!")
''')
    
    log_message(f"Aggregated data saved to: {os.path.join(results_dir, 'aggregated/')}")
    log_message(f"Plotting script created: {plotting_script}")


def generate_summary(
    available_checkpoints: Dict[str, Dict[str, str]],
    available_encoders: List[str],
    results: Dict[str, Dict[str, Dict[str, bool]]],
) -> None:
    """Generate summary of all evaluations."""
    summary = {
        "total_combinations": 0,
        "successful_combinations": 0,
        "available_checkpoints": available_checkpoints,
        "available_encoders": available_encoders,
        "results": results,
        "success_rate_by_subject": {},
        "success_rate_by_arch": {},
        "success_rate_by_encoder": {},
    }
    
    # Count totals
    total_success = 0
    total_attempts = 0
    
    for subject, archs in results.items():
        subj_success = 0
        subj_total = 0
        
        for arch, encoders in archs.items():
            for encoder, success in encoders.items():
                total_attempts += 1
                subj_total += 1
                if success:
                    total_success += 1
                    subj_success += 1
        
        summary["success_rate_by_subject"][subject] = (
            subj_success / subj_total if subj_total > 0 else 0
        )
    
    # Success rates by architecture and encoder
    arch_stats = {}
    encoder_stats = {}
    
    for subject, archs in results.items():
        for arch, encoders in archs.items():
            if arch not in arch_stats:
                arch_stats[arch] = {"success": 0, "total": 0}
            
            for encoder, success in encoders.items():
                if encoder not in encoder_stats:
                    encoder_stats[encoder] = {"success": 0, "total": 0}
                
                arch_stats[arch]["total"] += 1
                encoder_stats[encoder]["total"] += 1
                
                if success:
                    arch_stats[arch]["success"] += 1
                    encoder_stats[encoder]["success"] += 1
    
    for arch, stats in arch_stats.items():
        summary["success_rate_by_arch"][arch] = (
            stats["success"] / stats["total"] if stats["total"] > 0 else 0
        )
    
    for encoder, stats in encoder_stats.items():
        summary["success_rate_by_encoder"][encoder] = (
            stats["success"] / stats["total"] if stats["total"] > 0 else 0
        )
    
    summary["total_combinations"] = total_attempts
    summary["successful_combinations"] = total_success
    summary["overall_success_rate"] = total_success / total_attempts if total_attempts > 0 else 0
    
    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    log_message(f"Summary saved to: {summary_path}")
    log_message(f"Overall success rate: {summary['overall_success_rate']:.1%} "
                f"({total_success}/{total_attempts})")


def main():
    """Main evaluation loop."""
    parser = argparse.ArgumentParser(description="Run comprehensive img→lis evaluation")
    parser.add_argument("--subjects", nargs="*", default=ALL_SUBJECTS,
                        help="Subjects to evaluate (default: all)")
    parser.add_argument("--architectures", nargs="*", default=ALL_ARCHITECTURES,
                        help="Architectures to test (default: all)")
    parser.add_argument("--encoders", nargs="*", default=ALL_ENCODERS,
                        help="Encoders to test (default: all)")
    parser.add_argument("--benchmark_dir", default=BENCHMARK_DIR,
                        help="Path to benchmark results directory")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout per evaluation in minutes (default: 30)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be run without executing")
    
    args = parser.parse_args()
    
    log_message("="*60)
    log_message("COMPREHENSIVE IMG→LIS EVALUATION")
    log_message("="*60)
    
    # Check availability
    log_message("Checking available img→lis model checkpoints...")
    available_checkpoints = find_available_checkpoints(
        args.subjects, args.architectures, args.benchmark_dir
    )
    
    log_message("Checking available encoder models...")
    available_encoders = check_encoder_models(args.encoders)
    
    if not available_encoders:
        log_message("ERROR: No trained encoder models found. Run training first.")
        return
    
    # Count total combinations
    total_combinations = sum(
        len(archs) for archs in available_checkpoints.values()
    )
    
    log_message(f"Found:")
    log_message(f"  Subjects: {len(available_checkpoints)} / {len(args.subjects)}")
    log_message(f"  Checkpoints: {total_combinations} total combinations")
    log_message(f"  Encoders: {available_encoders}")
    log_message(f"  Results dir: {RESULTS_DIR}")
    
    if args.dry_run:
        log_message("\nDRY RUN - would evaluate:")
        for subject, archs in available_checkpoints.items():
            for arch in archs.keys():
                log_message(f"  {subject}/{arch} with encoders {available_encoders}")
        return
    
    # Run evaluations
    log_message(f"\nStarting evaluation (timeout: {args.timeout}min per combination)...")
    results = {}
    
    combination_num = 0
    for subject, archs in available_checkpoints.items():
        results[subject] = {}
        
        for arch, ckpt_path in archs.items():
            combination_num += 1
            log_message(f"\n[{combination_num}/{total_combinations}] {subject}/{arch}")
            
            eval_results = run_evaluation(
                subject, arch, ckpt_path, available_encoders, args.timeout
            )
            results[subject][arch] = eval_results
    
    # Generate summary
    log_message("\n" + "="*60)
    log_message("EVALUATION COMPLETE")
    log_message("="*60)
    
    generate_summary(available_checkpoints, available_encoders, results)
    
    # Aggregate results for plotting
    aggregate_results_for_plotting(RESULTS_DIR)
    
    log_message(f"All results saved in: {RESULTS_DIR}")
    log_message(f"Log file: {LOG_FILE}")
    log_message(f"To generate plots, run: cd {RESULTS_DIR} && python plot_rank_curves.py")


if __name__ == "__main__":
    main()