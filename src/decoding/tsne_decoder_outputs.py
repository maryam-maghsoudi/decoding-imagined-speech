"""
tsne_decoder_outputs.py
=======================
Generate separate t-SNE visualizations of decoder outputs for:
1. Real listened MEG → decoder → embeddings → t-SNE
2. Imagined MEG → LinearLag mapping → decoder → embeddings → t-SNE

Uses BERT+Wav2Vec2 encoder and aggregates data across all subjects.
Colors points by word (top-20 most frequent or top-20 best decodable).

Usage
-----
  python tsne_decoder_outputs.py --heldout_subject sub-01
  python tsne_decoder_outputs.py --heldout_subject sub-01 --color_by best_decodable
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

import mne
mne.set_log_level("ERROR")

# Import from contrastive_word_meg_compare.py
sys.path.insert(0, str(Path(__file__).parent))
from contrastive_word_meg_compare import (
    SUBJECTS, POEM_KEYS, ONSET_DIR, OUT_DIR, BASE_PATH,
    MEGWordDataset, TextEncoder, make_meg_encoder,
    build_embeddings_for_encoder, load_encoder_checkpoint,
    LinearLagModule, DEVICE, SEED, N_SESSIONS, DS_FACTOR,
    EPOCH_TMIN_S, WIN_PRE, WIN_POST, WIN_SIZE, EMB_DIM,
)

# Output directory
TSNE_OUT = os.path.join(OUT_DIR, "tsne_decoder_outputs")
os.makedirs(TSNE_OUT, exist_ok=True)

# Configuration
ENCODER_NAME = "bert"  # BERT + Wav2Vec2
print("Encoder: "+ ENCODER_NAME)
MAPPING_ARCH = "RNN"
TOP_K_WORDS = 20


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def load_meg_trial(subject: str, cond: str, session: int) -> np.ndarray:
    """Load and preprocess a single MEG trial."""
    fname = f"{subject}_sess-{session}_task-{cond}_meg-epo.fif"
    fpath = os.path.join(BASE_PATH, subject, f"ses-{session}", "meg", fname)
    epochs = mne.read_epochs(fpath, preload=True)
    raw = epochs.get_data().mean(axis=0)
    new_T = raw.shape[1] // DS_FACTOR
    data_ds = resample(raw, new_T, axis=1).astype(np.float32)
    # Normalize
    mu = data_ds.mean(axis=1, keepdims=True)
    sd = np.maximum(data_ds.std(axis=1, keepdims=True), 1e-12)
    data_ds = (data_ds - mu) / sd
    return data_ds


def onset_to_window_raw(onset_s: float, n_t: int) -> Tuple[int, int]:
    """Convert onset time to window indices."""
    orig_onset = int(round((onset_s - EPOCH_TMIN_S) * 100.0))  # 100 Hz sampling
    orig_start = orig_onset - WIN_PRE
    orig_end = orig_onset + WIN_POST
    if orig_start < 0 or orig_end > n_t:
        return None
    return orig_start, orig_end


def load_linearlag_mapping(heldout_subject: str) -> LinearLagModule:
    """Load LinearLag mapping model for the heldout subject."""
    # Look for LinearLag checkpoint
    bench_dir = Path(__file__).parent.parent / "benchmark" / "no_flash_removal"
    ckpt_path = bench_dir / "loso_out" / "models" / f"heldout_{heldout_subject}" / f"{MAPPING_ARCH}_full.npy"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"LinearLag checkpoint not found: {ckpt_path}")
    
    W = np.load(str(ckpt_path))
    return LinearLagModule(W).eval().to(DEVICE)


# =============================================================================
#  EMBEDDING EXTRACTION
# =============================================================================

def extract_real_listened_embeddings(
    meg_encoder: torch.nn.Module,
    heldout_subject: str,
    all_subjects: List[str],
    vocab: Dict[str, int],
) -> Tuple[np.ndarray, List[str]]:
    """Extract embeddings from real listened MEG across all subjects."""
    print(f"Extracting real listened embeddings across {len(all_subjects)} subjects...")
    
    # Build dataset from all subjects
    full_ds = MEGWordDataset(
        subjects=all_subjects,
        poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR,
        cond_suffix="lis",
        remove_flashes=False,
    )
    
    # Filter to vocab words only
    valid_pairs = [(meg_win, word) for meg_win, word in full_ds.pairs if word in vocab]
    
    if not valid_pairs:
        return np.empty((0, EMB_DIM), dtype=np.float32), []
    
    meg_encoder.eval()
    all_embs = []
    all_words = []
    
    # Process in batches
    batch_size = 256
    for i in range(0, len(valid_pairs), batch_size):
        batch = valid_pairs[i:i + batch_size]
        meg_windows = [pair[0] for pair in batch]
        word_labels = [pair[1] for pair in batch]
        
        if not meg_windows:
            continue
            
        x = torch.from_numpy(np.stack(meg_windows)).to(DEVICE)
        with torch.no_grad():
            embeddings = meg_encoder(x).cpu().numpy()
        
        all_embs.append(embeddings)
        all_words.extend(word_labels)
    
    if not all_embs:
        return np.empty((0, EMB_DIM), dtype=np.float32), []
    
    final_embs = np.concatenate(all_embs, axis=0)
    print(f"  Real listened: {len(final_embs)} embeddings from {len(set(all_words))} unique words")
    return final_embs, all_words


def extract_predicted_listened_embeddings(
    meg_encoder: torch.nn.Module,
    heldout_subject: str,
    mapping_model: LinearLagModule,
    vocab: Dict[str, int],
) -> Tuple[np.ndarray, List[str]]:
    """Extract embeddings from imagined MEG → LinearLag mapping → decoder."""
    print(f"Extracting predicted listened embeddings for {heldout_subject}...")
    
    all_embs = []
    all_words = []
    
    for poem_key in POEM_KEYS:
        onset_file = os.path.join(ONSET_DIR, f"{poem_key}_word_onsets.json")
        if not os.path.exists(onset_file):
            continue
            
        with open(onset_file) as f:
            word_onsets = json.load(f)
        
        cond = f"{poem_key}img"
        
        for session in range(N_SESSIONS):
            try:
                # Load imagined MEG data
                data = load_meg_trial(heldout_subject, cond, session)
                n_t = data.shape[-1]
                
                # Apply LinearLag mapping
                x_img = torch.from_numpy(data).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    x_pred = mapping_model(x_img).squeeze(0).cpu().numpy()
                
                # Extract word windows
                for w in word_onsets:
                    word = w["word"].strip().lower()
                    if word not in vocab:
                        continue
                    
                    window_idx = onset_to_window_raw(w["start"], x_pred.shape[-1])
                    if window_idx is None:
                        continue
                    
                    start, end = window_idx
                    window = x_pred[:, start:end]
                    if window.shape[-1] != WIN_SIZE:
                        continue
                    
                    # Encode through MEG encoder
                    x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        embedding = meg_encoder(x).cpu().numpy()
                    
                    all_embs.append(embedding[0])  # Remove batch dimension
                    all_words.append(word)
                    
            except Exception as e:
                print(f"    Warning: {heldout_subject}/{cond}/ses-{session}: {e}")
                continue
    
    if not all_embs:
        return np.empty((0, EMB_DIM), dtype=np.float32), []
    
    final_embs = np.stack(all_embs, axis=0)
    print(f"  Predicted listened: {len(final_embs)} embeddings from {len(set(all_words))} unique words")
    return final_embs, all_words


# =============================================================================
#  WORD SELECTION
# =============================================================================

def get_top_words_by_frequency(word_labels: List[str], top_k: int = TOP_K_WORDS) -> List[str]:
    """Get top-K most frequent words."""
    counts = Counter(word_labels)
    return [word for word, _ in counts.most_common(top_k)]


def get_top_words_by_decodability(
    embeddings: np.ndarray,
    word_labels: List[str],
    text_embeddings: np.ndarray,
    vocab: Dict[str, int],
    top_k: int = TOP_K_WORDS,
) -> List[str]:
    """Get top-K best decodable words based on ranking performance."""
    print("  Computing per-word decodability...")
    
    word_ranks = {}
    
    # Group embeddings by word
    word_to_embs = {}
    for emb, word in zip(embeddings, word_labels):
        if word not in word_to_embs:
            word_to_embs[word] = []
        word_to_embs[word].append(emb)
    
    # Compute ranking for each word
    text_embs_tensor = torch.from_numpy(text_embeddings).to(DEVICE)
    
    for word, word_embs in word_to_embs.items():
        if word not in vocab:
            continue
        
        word_idx = vocab[word]
        true_text_emb = text_embs_tensor[word_idx]
        
        ranks = []
        for emb in word_embs:
            emb_tensor = torch.from_numpy(emb).to(DEVICE)
            similarities = emb_tensor @ text_embs_tensor.T
            rank = int((similarities > similarities[word_idx]).sum().item()) + 1
            ranks.append(rank)
        
        word_ranks[word] = np.median(ranks)
    
    # Sort by median rank (lower is better)
    sorted_words = sorted(word_ranks.items(), key=lambda x: x[1])
    return [word for word, _ in sorted_words[:top_k]]


# =============================================================================
#  VISUALIZATION
# =============================================================================

def plot_tsne_by_words(
    embeddings: np.ndarray,
    word_labels: List[str],
    top_words: List[str],
    title: str,
    save_path: str,
    subsample_max: int = 5000,
) -> None:
    """Create t-SNE plot colored by words."""
    print(f"  Generating t-SNE plot: {title}")
    
    # Subsample if too many points
    if len(embeddings) > subsample_max:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(embeddings), subsample_max, replace=False)
        embeddings = embeddings[idx]
        word_labels = [word_labels[i] for i in idx]
    
    print(f"    Running t-SNE on {len(embeddings)} points...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=SEED, n_jobs=1)
    coords = tsne.fit_transform(embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Count word frequencies for legend
    word_counts = Counter(word_labels)
    
    # Plot other words in grey background
    word_labels_array = np.array(word_labels)
    other_mask = ~np.isin(word_labels_array, top_words)
    if other_mask.any():
        ax.scatter(
            coords[other_mask, 0], coords[other_mask, 1],
            c="lightgrey", s=4, alpha=0.3, label="other", rasterized=True,
        )
    
    # Plot top words in color
    cmap = cm.get_cmap("tab20", len(top_words))
    for w_idx, word in enumerate(top_words):
        mask = word_labels_array == word
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(w_idx)], s=15, alpha=0.8,
            label=f"{word} ({word_counts[word]})", rasterized=True,
        )
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=9, ncol=2, loc="upper right", 
              framealpha=0.9, edgecolor="grey")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"    [saved] {save_path}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate t-SNE plots of decoder outputs",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--heldout_subject", required=True,
        help="Subject to use for imagined MEG (e.g., sub-01)"
    )
    parser.add_argument(
        "--color_by", choices=["frequency", "best_decodable"], default="frequency",
        help="How to select top words: by frequency or decodability"
    )
    args = parser.parse_args()
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print(f"Device: {DEVICE}")
    print(f"Heldout subject: {args.heldout_subject}")
    print(f"Encoder: {ENCODER_NAME}")
    print(f"Mapping: {MAPPING_ARCH}")
    print(f"Color by: {args.color_by}")
    print(f"Output dir: {TSNE_OUT}\n")
    
    # Get training subjects (all except heldout)
    train_subjects = [s for s in SUBJECTS if s != args.heldout_subject]
    
    # Build vocabulary from training subjects
    print("Building vocabulary from training subjects...")
    ref_ds = MEGWordDataset(
        subjects=train_subjects,
        poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR,
        cond_suffix="lis",
        remove_flashes=False,
    )
    vocab = ref_ds.vocab
    n_channels = ref_ds.pairs[0][0].shape[0]
    
    # Load text embeddings
    print(f"Building {ENCODER_NAME} text embeddings...")
    emb_cache = {}
    raw_text_emb = build_embeddings_for_encoder(
        ENCODER_NAME, ref_ds.words, ref_ds.word_audio, emb_cache
    )
    
    # Load trained encoder
    print(f"Loading trained {ENCODER_NAME} encoder...")
    meg_encoder, text_encoder = load_encoder_checkpoint(
        ENCODER_NAME, n_channels, raw_text_emb, OUT_DIR
    )
    
    # Get text embeddings for decodability computation
    with torch.no_grad():
        text_embeddings = text_encoder.get_all().cpu().numpy()
    
    # Extract real listened embeddings
    real_embs, real_words = extract_real_listened_embeddings(
        meg_encoder, args.heldout_subject, train_subjects, vocab
    )
    
    # # Load LinearLag mapping model
    # mapping_model = load_linearlag_mapping(args.heldout_subject)
    
    # # Extract predicted listened embeddings
    # pred_embs, pred_words = extract_predicted_listened_embeddings(
    #     meg_encoder, args.heldout_subject, mapping_model, vocab
    # )
    
    # Select top words
    if args.color_by == "frequency":
        # Use combined frequency from both real and predicted
        combined_words = real_words
        top_words = get_top_words_by_frequency(combined_words, TOP_K_WORDS)
        word_selection = "most frequent"
    else:
        # Use decodability from real listened data
        top_words = get_top_words_by_decodability(
            real_embs, real_words, text_embeddings, vocab, TOP_K_WORDS
        )
        word_selection = "best decodable"
    
    print(f"\nTop {TOP_K_WORDS} {word_selection} words: {top_words}")
    
    # Generate t-SNE plots
    subj_tag = args.heldout_subject.replace("sub-", "")
    
    # Real listened t-SNE
    real_title = (f"Real Listened MEG → {ENCODER_NAME.upper()} Decoder\n"
                  f"Training subjects, colored by top-{TOP_K_WORDS} {word_selection} words")
    real_path = os.path.join(TSNE_OUT, f"tsne_real_listened_{subj_tag}_{args.color_by}.png")
    plot_tsne_by_words(real_embs, real_words, top_words, real_title, real_path)

    # Predicted listened t-SNE  
    # pred_title = (f"Imagined MEG → {MAPPING_ARCH} → {ENCODER_NAME.upper()} Decoder\n"
    #               f"Subject {args.heldout_subject}, colored by top-{TOP_K_WORDS} {word_selection} words")
    # pred_path = os.path.join(TSNE_OUT, f"tsne_predicted_listened_{subj_tag}_{args.color_by}.png")
    # plot_tsne_by_words(pred_embs, pred_words, top_words, pred_title, pred_path)
    
    # Save summary
    summary = {
        "heldout_subject": args.heldout_subject,
        "encoder_name": ENCODER_NAME,
        "mapping_arch": MAPPING_ARCH,
        "color_by": args.color_by,
        "top_words": top_words,
        "n_real_embeddings": len(real_embs),
        # "n_predicted_embeddings": len(pred_embs),
        "n_unique_real_words": len(set(real_words)),
        # "n_unique_predicted_words": len(set(pred_words)),
        "vocab_size": len(vocab),
    }
    
    summary_path = os.path.join(TSNE_OUT, f"summary_{subj_tag}_{args.color_by}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {summary_path}")
    print(f"\nDone! Check {TSNE_OUT}/")


if __name__ == "__main__":
    main()