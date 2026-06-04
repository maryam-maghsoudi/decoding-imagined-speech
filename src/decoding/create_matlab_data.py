#!/usr/bin/env python3
"""
create_matlab_data.py
=====================
Combines all evaluation results into a single .mat file for Matlab analysis.

Creates a structured .mat file with:
- All-words rank data for each subject/architecture/encoder combination
- Top-20 words rank data for each combination  
- Metadata for plotting and analysis

Output: full_eval_results.mat

Usage in Matlab:
    load('full_eval_results.mat');
    % Plot rank curves for subject sub-03, LinearLag architecture, bert_wav2vec encoder
    ranks_all = data.all_words.sub_03.LinearLag.bert_wav2vec.ranks;
    ranks_top20 = data.top20.sub_03.LinearLag.bert_wav2vec.ranks;
    top20_words = data.metadata.top20_words.sub_03.LinearLag;
"""

import os
import json
import numpy as np
from pathlib import Path
from scipy.io import savemat
from typing import Dict, List, Any
import sys

# Configuration
RESULTS_DIR = "./full_eval_results"
OUTPUT_FILE = "full_eval_results.mat"

def load_json_safe(file_path: str) -> Dict:
    """Load JSON file safely, return empty dict if file doesn't exist."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def sanitize_matlab_fieldname(name: str) -> str:
    """Convert names to valid Matlab field names."""
    # Replace hyphens with underscores and remove invalid characters
    name = name.replace('-', '_').replace('.', '_')
    # Ensure it starts with a letter
    if name[0].isdigit():
        name = 'sub_' + name
    return name

def create_matlab_structure() -> Dict[str, Any]:
    """Create the main data structure for Matlab."""
    print("Creating Matlab data structure...")
    
    # Main structure
    data = {
        'all_words': {},      # All-words rank data
        'top20': {},          # Top-20 words rank data  
        'metadata': {         # Metadata for plotting
            'subjects': [],
            'architectures': [],
            'encoders': [],
            'top20_words': {},    # Top-20 word lists per subject/arch
            'vocab_size': 76,     # Total vocabulary size
            'description': 'MEG word retrieval evaluation results'
        }
    }
    
    return data

def collect_available_combinations() -> List[tuple]:
    """Scan results directory to find all available combinations."""
    print("Scanning for available combinations...")
    
    combinations = []
    results_path = Path(RESULTS_DIR)
    
    if not results_path.exists():
        print(f"ERROR: Results directory {RESULTS_DIR} not found!")
        return combinations
    
    for subject_dir in results_path.iterdir():
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
            continue
            
        subject = subject_dir.name
        
        for arch_dir in subject_dir.iterdir():
            if not arch_dir.is_dir():
                continue
                
            arch = arch_dir.name
            
            for encoder_dir in arch_dir.iterdir():
                if not encoder_dir.is_dir():
                    continue
                    
                encoder = encoder_dir.name
                
                # Check if required files exist
                ranks_all_file = encoder_dir / "ranks_eval_img.npy"
                ranks_top20_file = encoder_dir / "ranks_eval_img_top20.npy"
                
                if ranks_all_file.exists() and ranks_top20_file.exists():
                    combinations.append((subject, arch, encoder))
                else:
                    print(f"  Missing files for {subject}/{arch}/{encoder}")
    
    print(f"Found {len(combinations)} complete combinations")
    return combinations

def load_combination_data(subject: str, arch: str, encoder: str) -> Dict[str, Any]:
    """Load all data for one subject/architecture/encoder combination."""
    base_path = Path(RESULTS_DIR) / subject / arch / encoder
    
    result = {
        'ranks_all': None,
        'ranks_top20': None, 
        'word_labels_top20': None,
        'top20_words': None,
        'metrics': None,
        'per_word_stats': None
    }
    
    try:
        # Load all-words ranks
        ranks_all_file = base_path / "ranks_eval_img.npy"
        if ranks_all_file.exists():
            result['ranks_all'] = np.load(ranks_all_file)
        
        # Load top-20 ranks
        ranks_top20_file = base_path / "ranks_eval_img_top20.npy"
        if ranks_top20_file.exists():
            result['ranks_top20'] = np.load(ranks_top20_file)
        
        # Load top-20 word labels
        labels_top20_file = base_path / "word_labels_eval_img_top20.npy" 
        if labels_top20_file.exists():
            result['word_labels_top20'] = np.load(labels_top20_file)
        
        # Load top-20 word list
        top20_file = base_path / "top20_words_eval_img.json"
        if top20_file.exists():
            result['top20_words'] = load_json_safe(top20_file)
        
        # Load metrics
        metrics_file = base_path / "eval_img_metrics.json"
        if metrics_file.exists():
            result['metrics'] = load_json_safe(metrics_file)
        
        # Load per-word stats
        stats_file = base_path / "per_word_stats_eval_img.json"
        if stats_file.exists():
            result['per_word_stats'] = load_json_safe(stats_file)
            
    except Exception as e:
        print(f"  Error loading {subject}/{arch}/{encoder}: {e}")
    
    return result

def add_combination_to_structure(data: Dict, subject: str, arch: str, encoder: str, 
                                combo_data: Dict) -> None:
    """Add one combination's data to the main structure."""
    
    # Sanitize names for Matlab
    subj_field = sanitize_matlab_fieldname(subject)
    arch_field = sanitize_matlab_fieldname(arch)
    enc_field = sanitize_matlab_fieldname(encoder)
    
    # Initialize nested structure if needed
    if subj_field not in data['all_words']:
        data['all_words'][subj_field] = {}
    if arch_field not in data['all_words'][subj_field]:
        data['all_words'][subj_field][arch_field] = {}
        
    if subj_field not in data['top20']:
        data['top20'][subj_field] = {}
    if arch_field not in data['top20'][subj_field]:
        data['top20'][subj_field][arch_field] = {}
    
    # Add all-words data
    if combo_data['ranks_all'] is not None:
        data['all_words'][subj_field][arch_field][enc_field] = {
            'ranks': combo_data['ranks_all'],
            'metrics': combo_data['metrics'] or {},
            'per_word_stats': combo_data['per_word_stats'] or {}
        }
    
    # Add top-20 data
    if combo_data['ranks_top20'] is not None:
        word_labels = combo_data['word_labels_top20']
        if word_labels is None:
            word_labels = np.array([])
        
        data['top20'][subj_field][arch_field][enc_field] = {
            'ranks': combo_data['ranks_top20'],
            'word_labels': word_labels,
            'metrics': combo_data['metrics'] or {}
        }
    
    # Add top-20 words list to metadata (one per subject/arch combination)
    if subj_field not in data['metadata']['top20_words']:
        data['metadata']['top20_words'][subj_field] = {}
    if arch_field not in data['metadata']['top20_words'][subj_field]:
        if combo_data['top20_words']:
            data['metadata']['top20_words'][subj_field][arch_field] = combo_data['top20_words']

def finalize_metadata(data: Dict, combinations: List[tuple]) -> None:
    """Add final metadata to the structure."""
    
    # Extract unique subjects, architectures, encoders
    subjects = sorted(list(set(subj for subj, _, _ in combinations)))
    architectures = sorted(list(set(arch for _, arch, _ in combinations)))
    encoders = sorted(list(set(enc for _, _, enc in combinations)))
    
    data['metadata']['subjects'] = [sanitize_matlab_fieldname(s) for s in subjects]
    data['metadata']['architectures'] = [sanitize_matlab_fieldname(a) for a in architectures]
    data['metadata']['encoders'] = [sanitize_matlab_fieldname(e) for e in encoders]
    data['metadata']['n_combinations'] = len(combinations)
    
    # Add original names mapping
    data['metadata']['original_names'] = {
        'subjects': {sanitize_matlab_fieldname(s): s for s in subjects},
        'architectures': {sanitize_matlab_fieldname(a): a for a in architectures},
        'encoders': {sanitize_matlab_fieldname(e): e for e in encoders}
    }
    
    print(f"Metadata summary:")
    print(f"  Subjects: {len(subjects)} - {subjects}")
    print(f"  Architectures: {len(architectures)} - {architectures}")
    print(f"  Encoders: {len(encoders)} - {encoders}")

def create_matlab_usage_example() -> str:
    """Create example Matlab code for using the data."""
    
    example_code = '''
% Example Matlab usage:
% load('full_eval_results.mat');

%% 1. Plot all-words rank curve for sub-03, LinearLag, bert_wav2vec
subject = 'sub_03';
architecture = 'LinearLag'; 
encoder = 'bert_wav2vec';

ranks_all = data.all_words.(subject).(architecture).(encoder).ranks;
vocab_size = data.metadata.vocab_size;

% Calculate CDF
max_rank = min(50, max(ranks_all));
x = 1:max_rank;
cdf = arrayfun(@(r) mean(ranks_all <= r), x);

% Plot
figure;
plot(x, cdf, 'LineWidth', 2);
xlabel('Rank k');
ylabel('P(rank ≤ k)');
title(sprintf('All Words - %s/%s/%s', subject, architecture, encoder));
grid on;

%% 2. Plot top-20 words rank curve
ranks_top20 = data.top20.(subject).(architecture).(encoder).ranks;
top20_words = data.metadata.top20_words.(subject).(architecture);

max_rank = min(50, max(ranks_top20));
x = 1:max_rank;
cdf_top20 = arrayfun(@(r) mean(ranks_top20 <= r), x);

hold on;
plot(x, cdf_top20, 'LineWidth', 2);
legend({'All words', 'Top-20 words'});

%% 3. Compare encoders for same subject/architecture
figure;
encoders = data.metadata.encoders;
colors = lines(length(encoders));

for i = 1:length(encoders)
    enc = encoders{i};
    if isfield(data.all_words.(subject).(architecture), enc)
        ranks = data.all_words.(subject).(architecture).(enc).ranks;
        max_rank = min(50, max(ranks));
        x = 1:max_rank;
        cdf = arrayfun(@(r) mean(ranks <= r), x);
        
        plot(x, cdf, 'Color', colors(i,:), 'LineWidth', 2);
        hold on;
    end
end

xlabel('Rank k');
ylabel('P(rank ≤ k)');
title(sprintf('Encoder Comparison - %s/%s', subject, architecture));
legend(encoders);
grid on;
'''
    
    return example_code

def main():
    """Main function to create the Matlab data file."""
    print("="*60)
    print("CREATING MATLAB DATA FILE")
    print("="*60)
    
    # Create main structure
    data = create_matlab_structure()
    
    # Find all combinations
    combinations = collect_available_combinations()
    
    if not combinations:
        print("ERROR: No complete combinations found!")
        return
    
    # Process each combination
    print(f"\nProcessing {len(combinations)} combinations...")
    
    for i, (subject, arch, encoder) in enumerate(combinations):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(combinations)}] Processing {subject}/{arch}/{encoder}")
        
        # Load data for this combination
        combo_data = load_combination_data(subject, arch, encoder)
        
        # Add to main structure
        add_combination_to_structure(data, subject, arch, encoder, combo_data)
    
    # Finalize metadata
    finalize_metadata(data, combinations)
    
    # Save to .mat file
    print(f"\nSaving to {OUTPUT_FILE}...")
    try:
        savemat(OUTPUT_FILE, {'data': data}, do_compression=True)
        print(f"✅ Successfully saved {OUTPUT_FILE}")
        
        # Print file size
        file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return
    
    # Create usage example
    example_file = "matlab_usage_example.m"
    with open(example_file, 'w') as f:
        f.write(create_matlab_usage_example())
    print(f"✅ Created usage example: {example_file}")
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"Matlab file ready: {OUTPUT_FILE}")
    print(f"Usage example: {example_file}")
    print("\nIn Matlab:")
    print(f"  load('{OUTPUT_FILE}');")
    print("  % Access data as: data.all_words.sub_03.LinearLag.bert_wav2vec.ranks")
    print("  % Top-20 data: data.top20.sub_03.LinearLag.bert_wav2vec.ranks")

if __name__ == "__main__":
    main()