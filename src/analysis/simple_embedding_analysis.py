"""
simple_embedding_analysis.py
============================
Simple analysis of learned word embeddings that works directly with checkpoints.

This script:
1. Loads embeddings from the trained text encoder checkpoint
2. Loads actual word vocabulary from onset files
3. Analyzes word similarities and clusters

Usage
-----
python simple_embedding_analysis.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Config
CONTRASTIVE_DIR = Path(__file__).parent.parent / "contrastive_learning"
ONSET_DIR = CONTRASTIVE_DIR / "onset_out"
CONTRASTIVE_OUT = CONTRASTIVE_DIR / "contrastive_out"
OUT_DIR = "./simple_analysis_out"
os.makedirs(OUT_DIR, exist_ok=True)

def load_vocabulary_from_onsets() -> List[str]:
    """Load actual vocabulary from onset files."""
    print("Loading vocabulary from onset files...")
    
    all_words = set()
    
    for poem_file in ["poem1_word_onsets.json", "poem2_word_onsets.json"]:
        onset_path = ONSET_DIR / poem_file
        if onset_path.exists():
            print(f"  Loading words from {poem_file}")
            with open(onset_path) as f:
                word_data = json.load(f)
            
            for word_info in word_data:
                word = word_info["word"].strip().lower()
                if len(word) >= 1:  # Minimum word length
                    all_words.add(word)
        else:
            print(f"  WARNING: {onset_path} not found")
    
    words = sorted(list(all_words))
    print(f"  Found {len(words)} unique words")
    print(f"  Sample words: {words[:10]}")
    return words

def load_embeddings_from_checkpoint() -> Tuple[List[str], np.ndarray]:
    """Load embeddings from checkpoint and match with vocabulary."""
    print("Loading embeddings from checkpoint...")
    
    ckpt_path = CONTRASTIVE_OUT / "text_encoder.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    if 'embeddings' not in checkpoint:
        raise RuntimeError("No embeddings in checkpoint")
    
    raw_embeddings = checkpoint['embeddings']  # (V, 768)
    vocab_size = raw_embeddings.shape[0]
    
    print(f"  Checkpoint has {vocab_size} embeddings of dimension {raw_embeddings.shape[1]}")
    
    # Apply learned projection
    proj_layers = {k: v for k, v in checkpoint.items() if k.startswith('proj.')}
    
    if len(proj_layers) >= 4:  # Has full projection
        print("  Applying learned projection...")
        
        w1 = proj_layers['proj.0.weight']  # (256, 768)
        b1 = proj_layers['proj.0.bias']    # (256,)
        w2 = proj_layers['proj.3.weight']  # (128, 256)
        b2 = proj_layers['proj.3.bias']    # (128,)
        
        with torch.no_grad():
            h1 = torch.relu(raw_embeddings @ w1.T + b1)
            final_emb = h1 @ w2.T + b2
            final_emb = F.normalize(final_emb, dim=-1)
        
        embeddings = final_emb.numpy()
        print(f"  Final embeddings: {embeddings.shape}")
    else:
        print("  Using raw embeddings (no projection found)")
        embeddings = raw_embeddings.numpy()
    
    # Try to load actual vocabulary
    try:
        actual_words = load_vocabulary_from_onsets()
        
        # If vocab sizes match, use actual words
        if len(actual_words) == vocab_size:
            print(f"  Vocabulary size matches! Using actual word labels.")
            return actual_words, embeddings
        else:
            print(f"  Vocabulary mismatch: checkpoint={vocab_size}, onsets={len(actual_words)}")
            print(f"  Using generic labels")
    except Exception as e:
        print(f"  Could not load onset vocabulary: {e}")
    
    # Fallback to generic labels
    words = [f"word_{i:03d}" for i in range(vocab_size)]
    return words, embeddings

def compute_similarities(embeddings: np.ndarray, words: List[str], top_k: int = 10) -> Dict:
    """Compute similarity matrix and find nearest neighbors."""
    print("Computing word similarities...")
    
    similarity = cosine_similarity(embeddings)
    
    # Find nearest neighbors
    neighbors = {}
    for i, word in enumerate(words):
        scores = similarity[i]
        sorted_indices = np.argsort(scores)[::-1]
        sorted_indices = sorted_indices[sorted_indices != i]  # exclude self
        
        neighbors[word] = [
            (words[j], float(scores[j]))
            for j in sorted_indices[:top_k]
        ]
    
    return {"similarity_matrix": similarity, "neighbors": neighbors}

def cluster_words(embeddings: np.ndarray, words: List[str], n_clusters: int = 6) -> Dict:
    """Cluster words using K-means."""
    print(f"Clustering words into {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    clusters = {}
    for i in range(n_clusters):
        cluster_words = [words[j] for j, label in enumerate(labels) if label == i]
        clusters[f"cluster_{i}"] = cluster_words
        print(f"  Cluster {i}: {len(cluster_words)} words")
    
    return {"labels": labels, "clusters": clusters}

def plot_similarity_matrix(similarity: np.ndarray, words: List[str], max_show: int = 40):
    """Plot similarity matrix heatmap."""
    print(f"Plotting similarity matrix (showing first {max_show} words)...")
    
    n_show = min(max_show, len(words))
    sim_sub = similarity[:n_show, :n_show]
    words_sub = words[:n_show]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        sim_sub,
        xticklabels=words_sub,
        yticklabels=words_sub,
        cmap='RdYlBu_r',
        center=0,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(f'Word Similarity Matrix (First {n_show} Words)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    out_path = os.path.join(OUT_DIR, "similarity_matrix.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

def plot_tsne(embeddings: np.ndarray, words: List[str], cluster_labels=None):
    """Create t-SNE visualization."""
    print("Creating t-SNE visualization...")
    
    # Set perplexity safely
    perplexity = min(30, max(5, len(words) // 4))
    if perplexity < 5:
        print(f"  Too few words ({len(words)}) for t-SNE, skipping...")
        return
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    
    if cluster_labels is not None:
        n_clusters = len(set(cluster_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            plt.scatter(coords[mask, 0], coords[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)
    else:
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=50)
    
    # Add word labels (show every nth word to avoid overcrowding)
    step = max(1, len(words) // 30)
    for i in range(0, len(words), step):
        plt.annotate(words[i], (coords[i, 0], coords[i, 1]),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    plt.title('t-SNE Visualization of Word Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    if cluster_labels is not None:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    suffix = "_clustered" if cluster_labels is not None else ""
    out_path = os.path.join(OUT_DIR, f"tsne{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

def print_analysis_results(neighbors: Dict, clusters: Dict):
    """Print interesting analysis results."""
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    # Show some nearest neighbors
    print("\nNearest neighbors for some words:")
    sample_words = list(neighbors.keys())[:8]  # Show first 8 words
    
    for word in sample_words:
        print(f"\n'{word}' is most similar to:")
        for neighbor, sim in neighbors[word][:5]:
            print(f"  {neighbor:<15} (similarity: {sim:.3f})")
    
    # Show clusters
    print(f"\nWord clusters:")
    for cluster_name, cluster_words in clusters.items():
        print(f"\n{cluster_name} ({len(cluster_words)} words):")
        # Show up to 12 words per cluster
        display_words = cluster_words[:12]
        print(f"  {', '.join(display_words)}")
        if len(cluster_words) > 12:
            print(f"  ... and {len(cluster_words) - 12} more")

def save_results(words: List[str], embeddings: np.ndarray, 
                neighbors: Dict, clusters: Dict):
    """Save analysis results."""
    print("Saving results...")
    
    # Save neighbors
    with open(os.path.join(OUT_DIR, "nearest_neighbors.json"), 'w') as f:
        json.dump(neighbors, f, indent=2)
    
    # Save clusters
    with open(os.path.join(OUT_DIR, "clusters.json"), 'w') as f:
        json.dump(clusters, f, indent=2)
    
    # Save embeddings
    np.savez(os.path.join(OUT_DIR, "embeddings.npz"),
             embeddings=embeddings, words=words)
    
    # Save vocabulary
    with open(os.path.join(OUT_DIR, "vocabulary.txt"), 'w') as f:
        for word in words:
            f.write(f"{word}\n")
    
    print(f"  Results saved to: {OUT_DIR}")

def main():
    print("Simple Embedding Analysis")
    print("=" * 60)
    
    # Load data
    words, embeddings = load_embeddings_from_checkpoint()
    print(f"Loaded {len(words)} words with {embeddings.shape[1]}D embeddings")
    
    # Compute similarities
    sim_results = compute_similarities(embeddings, words, top_k=10)
    similarity = sim_results["similarity_matrix"]
    neighbors = sim_results["neighbors"]
    
    # Cluster words
    cluster_results = cluster_words(embeddings, words, n_clusters=6)
    cluster_labels = cluster_results["labels"]
    clusters = cluster_results["clusters"]
    
    # Create visualizations
    plot_similarity_matrix(similarity, words)
    plot_tsne(embeddings, words)
    plot_tsne(embeddings, words, cluster_labels)
    
    # Print and save results
    print_analysis_results(neighbors, clusters)
    save_results(words, embeddings, neighbors, clusters)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"All results saved to: {OUT_DIR}")
    print("Check the .png files for visualizations!")

if __name__ == "__main__":
    main()