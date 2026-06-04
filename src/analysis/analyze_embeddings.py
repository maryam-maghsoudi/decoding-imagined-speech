"""
analyze_embeddings.py
=====================
Analyze learned word embeddings from the trained contrastive MEG-text model.

This script loads the trained text_encoder and extracts embeddings for all words
in the vocabulary, then analyzes:
1. Word similarity matrices
2. Nearest neighbors for each word
3. Clustering of semantically related words
4. t-SNE/UMAP visualization
5. Distance distributions

Usage
-----
python analyze_embeddings.py [--top_k 10] [--save_embeddings]
"""

import argparse
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
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add contrastive learning directory to path
_HERE = Path(__file__).parent.resolve()
_CONTRASTIVE = _HERE.parent / "contrastive_learning"
sys.path.insert(0, str(_CONTRASTIVE))

from contrastive_word_meg import (
    SUBJECTS, POEM_KEYS, ONSET_DIR, DEVICE, SEED,
    MEGWordDataset, TextEncoder,
    build_text_embeddings, TEXT_ENCODER
)

# =============================================================================
#  CONFIG
# =============================================================================
CONTRASTIVE_OUT = str(_CONTRASTIVE / "contrastive_out")
OUT_DIR = "./embedding_analysis_out"
os.makedirs(OUT_DIR, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
#  LOAD TRAINED MODELS
# =============================================================================

def load_vocabulary() -> Tuple[Dict[str, int], List[str]]:
    """
    Load the vocabulary used during training by recreating the dataset.
    Returns (vocab_dict, word_list).
    """
    print("Loading vocabulary from training data...")
    dataset = MEGWordDataset(
        subjects=SUBJECTS,
        poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR,
        cond_suffix="lis",
        remove_flashes=False,
    )
    print(f"  Vocabulary size: {len(dataset.vocab)} words")
    return dataset.vocab, dataset.words

def build_text_embeddings_safe(words: List[str], method: str = TEXT_ENCODER) -> torch.Tensor:
    """
    Safely build text embeddings with better error handling.
    """
    print(f"Building text embeddings using method: {method}")
    print(f"Number of words: {len(words)}")
    print(f"Sample words: {words[:5]}")
    
    if method == "bert":
        return build_text_embeddings_bert_safe(words)
    elif method == "glove":
        from contrastive_word_meg import build_text_embeddings_glove
        return build_text_embeddings_glove(words, "")  # Will use gensim download
    elif method == "random":
        from contrastive_word_meg import build_text_embeddings_random
        return build_text_embeddings_random(words)
    else:
        raise ValueError(f"Unknown text encoder method: {method}")

def build_text_embeddings_bert_safe(words: List[str]) -> torch.Tensor:
    """
    Safe BERT embedding extraction with better error handling.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        
        model_name = "bert-base-uncased"
        print(f"Loading BERT tokenizer/model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
        
        embeddings = []
        CHUNK = 32  # Smaller chunks for safety
        
        with torch.no_grad():
            for i in range(0, len(words), CHUNK):
                batch = words[i : i + CHUNK]
                print(f"  Processing batch {i//CHUNK + 1}/{(len(words)-1)//CHUNK + 1}")
                
                try:
                    enc = tokenizer(
                        batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=8,
                    ).to(DEVICE)
                    
                    out = model(**enc).last_hidden_state   # (B, seq_len, 768)
                    
                    # Mean over non-padding tokens
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    summed = (out * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1)
                    emb = (summed / counts).cpu()          # (B, 768)
                    
                    embeddings.append(emb)
                    
                except Exception as e:
                    print(f"  Error in batch {i//CHUNK + 1}: {e}")
                    # Create zero embeddings for failed batch
                    batch_size = len(batch)
                    zero_emb = torch.zeros(batch_size, 768)
                    embeddings.append(zero_emb)
        
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        
        if not embeddings:
            raise RuntimeError("No embeddings were successfully created")
        
        result = torch.cat(embeddings, dim=0).float()   # (V, 768)
        print(f"  BERT embeddings shape: {result.shape}")
        return result
        
    except Exception as e:
        print(f"  BERT embedding failed: {e}")
        print("  Falling back to random embeddings...")
        return build_text_embeddings_random_fallback(words)

def build_text_embeddings_random_fallback(words: List[str], dim: int = 768) -> torch.Tensor:
    """Fallback random embeddings."""
    print(f"  Creating random embeddings: {len(words)} words, {dim} dimensions")
    rng = np.random.default_rng(SEED)
    raw = rng.standard_normal((len(words), dim)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12
    return torch.from_numpy(raw)

def load_trained_text_encoder(vocab: Dict[str, int], words: List[str]) -> TextEncoder:
    """Load the trained TextEncoder from the contrastive training."""
    print("Loading trained text encoder...")
    
    # Build the same text embeddings used during training
    raw_emb = build_text_embeddings_safe(words, method=TEXT_ENCODER)
    
    # Create TextEncoder and load trained weights
    text_encoder = TextEncoder(raw_emb).to(DEVICE)
    
    ckpt_path = os.path.join(CONTRASTIVE_OUT, "text_encoder.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Text encoder checkpoint not found: {ckpt_path}")
    
    text_encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    text_encoder.eval()
    
    print(f"  Loaded from: {ckpt_path}")
    print(f"  Text encoder input dim: {text_encoder.embeddings.shape}")
    return text_encoder

# =============================================================================
#  EMBEDDING EXTRACTION
# =============================================================================

def extract_all_embeddings(text_encoder: TextEncoder, words: List[str]) -> np.ndarray:
    """
    Extract embeddings for all words in vocabulary.
    Returns: (V, embedding_dim) array
    """
    print("Extracting embeddings for all words...")
    with torch.no_grad():
        embeddings = text_encoder.get_all().cpu().numpy()  # (V, D)
    
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    return embeddings

# =============================================================================
#  SIMILARITY ANALYSIS
# =============================================================================

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between all words."""
    print("Computing similarity matrix...")
    similarity = cosine_similarity(embeddings)
    return similarity

def find_nearest_neighbors(
    embeddings: np.ndarray, 
    words: List[str], 
    top_k: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Find top-k nearest neighbors for each word.
    Returns dict: word -> [(neighbor, similarity_score), ...]
    """
    print(f"Finding top-{top_k} nearest neighbors for each word...")
    
    similarity = compute_similarity_matrix(embeddings)
    neighbors = {}
    
    for i, word in enumerate(words):
        # Get similarity scores for this word (exclude self)
        scores = similarity[i]
        
        # Sort by similarity (descending) and exclude self
        sorted_indices = np.argsort(scores)[::-1]
        sorted_indices = sorted_indices[sorted_indices != i]  # exclude self
        
        # Get top-k neighbors
        top_indices = sorted_indices[:top_k]
        neighbors[word] = [
            (words[j], float(scores[j])) 
            for j in top_indices
        ]
    
    return neighbors

def analyze_distance_distribution(embeddings: np.ndarray) -> Dict[str, float]:
    """Analyze the distribution of distances between word embeddings."""
    print("Analyzing distance distribution...")
    
    # Compute all pairwise cosine similarities
    similarity = cosine_similarity(embeddings)
    
    # Get upper triangle (excluding diagonal) to avoid duplicates
    n = similarity.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    similarities = similarity[upper_tri_indices]
    
    # Convert to distances
    distances = 1 - similarities
    
    stats = {
        "mean_similarity": float(np.mean(similarities)),
        "std_similarity": float(np.std(similarities)),
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
    }
    
    return stats

# =============================================================================
#  CLUSTERING ANALYSIS
# =============================================================================

def perform_kmeans_clustering(
    embeddings: np.ndarray, 
    words: List[str], 
    n_clusters: int = 8
) -> Tuple[np.ndarray, Dict[int, List[str]]]:
    """
    Perform K-means clustering on word embeddings.
    Returns: (cluster_labels, cluster_to_words_dict)
    """
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    labels = kmeans.fit_predict(embeddings)
    
    # Group words by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for word, label in zip(words, labels):
        clusters[label].append(word)
    
    # Print cluster summary
    for i in range(n_clusters):
        print(f"  Cluster {i}: {len(clusters[i])} words")
    
    return labels, clusters

# =============================================================================
#  VISUALIZATION
# =============================================================================

def plot_similarity_matrix(
    similarity: np.ndarray, 
    words: List[str], 
    max_words: int = 50
) -> None:
    """Plot similarity matrix heatmap (limited to max_words for readability)."""
    print(f"Plotting similarity matrix (showing top {max_words} words)...")
    
    # Limit to first max_words for visualization
    if len(words) > max_words:
        similarity_sub = similarity[:max_words, :max_words]
        words_sub = words[:max_words]
    else:
        similarity_sub = similarity
        words_sub = words
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_sub, 
        xticklabels=words_sub, 
        yticklabels=words_sub,
        cmap='RdYlBu_r', 
        center=0, 
        square=True,
        fmt='.2f',
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(f'Word Embedding Similarity Matrix (Top {len(words_sub)} Words)')
    plt.xlabel('Words')
    plt.ylabel('Words')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    out_path = os.path.join(OUT_DIR, "similarity_matrix.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

def plot_tsne_visualization(
    embeddings: np.ndarray, 
    words: List[str], 
    cluster_labels: np.ndarray = None
) -> None:
    """Create t-SNE visualization of word embeddings."""
    print("Creating t-SNE visualization...")
    
    # Compute t-SNE
    perplexity = min(30, max(5, (len(words)-1)//3))  # Ensure perplexity is valid
    if perplexity <= 0:
        print(f"  Skipping t-SNE: not enough words ({len(words)})")
        return
        
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(14, 10))
    
    if cluster_labels is not None:
        # Color by clusters
        n_clusters = len(np.unique(cluster_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            plt.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50
            )
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)
    
    # Add word labels
    for i, word in enumerate(words):
        plt.annotate(
            word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, alpha=0.8
        )
    
    plt.title('t-SNE Visualization of Word Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    if cluster_labels is not None:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    suffix = "_clustered" if cluster_labels is not None else ""
    out_path = os.path.join(OUT_DIR, f"tsne_visualization{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

def plot_distance_distribution(stats: Dict[str, float]) -> None:
    """Plot distribution of pairwise distances."""
    print("Plotting distance distribution...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create some sample data for histogram (since we only have summary stats)
    # In a real implementation, you'd save the actual distance arrays
    ax1.text(0.5, 0.5, 
             f"Mean Similarity: {stats['mean_similarity']:.3f}\n"
             f"Std Similarity: {stats['std_similarity']:.3f}\n"
             f"Min Similarity: {stats['min_similarity']:.3f}\n"
             f"Max Similarity: {stats['max_similarity']:.3f}",
             transform=ax1.transAxes, ha='center', va='center',
             fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat'))
    ax1.set_title('Similarity Statistics')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    ax2.text(0.5, 0.5,
             f"Mean Distance: {stats['mean_distance']:.3f}\n"
             f"Std Distance: {stats['std_distance']:.3f}\n"
             f"Min Distance: {stats['min_distance']:.3f}\n"
             f"Max Distance: {stats['max_distance']:.3f}",
             transform=ax2.transAxes, ha='center', va='center',
             fontsize=12, bbox=dict(boxstyle="round", facecolor='lightblue'))
    ax2.set_title('Distance Statistics')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.suptitle('Pairwise Distance/Similarity Statistics')
    plt.tight_layout()
    
    out_path = os.path.join(OUT_DIR, "distance_stats.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

def plot_hierarchical_clustering(
    embeddings: np.ndarray, 
    words: List[str], 
    max_words: int = 30
) -> None:
    """Create hierarchical clustering dendrogram."""
    print(f"Creating hierarchical clustering dendrogram (top {max_words} words)...")
    
    # Limit words for readability
    if len(words) > max_words:
        embeddings_sub = embeddings[:max_words]
        words_sub = words[:max_words]
    else:
        embeddings_sub = embeddings
        words_sub = words
    
    # Compute distance matrix
    distances = pdist(embeddings_sub, metric='cosine')
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distances, method='ward')
    
    plt.figure(figsize=(12, 8))
    dendrogram(
        linkage_matrix,
        labels=words_sub,
        leaf_rotation=45,
        leaf_font_size=10
    )
    plt.title(f'Hierarchical Clustering of Word Embeddings (Top {len(words_sub)} Words)')
    plt.xlabel('Words')
    plt.ylabel('Distance')
    plt.tight_layout()
    
    out_path = os.path.join(OUT_DIR, "hierarchical_clustering.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

# =============================================================================
#  SAVE RESULTS
# =============================================================================

def save_results(
    words: List[str],
    embeddings: np.ndarray,
    neighbors: Dict[str, List[Tuple[str, float]]],
    clusters: Dict[int, List[str]],
    stats: Dict[str, float],
    save_embeddings: bool = False
) -> None:
    """Save analysis results to files."""
    print("Saving analysis results...")
    
    # Save nearest neighbors
    neighbors_path = os.path.join(OUT_DIR, "nearest_neighbors.json")
    with open(neighbors_path, 'w') as f:
        json.dump(neighbors, f, indent=2)
    print(f"  Saved neighbors: {neighbors_path}")
    
    # Save clusters
    clusters_path = os.path.join(OUT_DIR, "word_clusters.json")
    with open(clusters_path, 'w') as f:
        json.dump(clusters, f, indent=2)
    print(f"  Saved clusters: {clusters_path}")
    
    # Save statistics
    stats_path = os.path.join(OUT_DIR, "embedding_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats: {stats_path}")
    
    # Save embeddings if requested
    if save_embeddings:
        embeddings_path = os.path.join(OUT_DIR, "word_embeddings.npz")
        np.savez(embeddings_path, embeddings=embeddings, words=words)
        print(f"  Saved embeddings: {embeddings_path}")
    
    # Save vocabulary
    vocab_path = os.path.join(OUT_DIR, "vocabulary.json")
    with open(vocab_path, 'w') as f:
        json.dump(words, f, indent=2)
    print(f"  Saved vocabulary: {vocab_path}")

def print_interesting_findings(
    neighbors: Dict[str, List[Tuple[str, float]]],
    clusters: Dict[int, List[str]]
) -> None:
    """Print some interesting findings."""
    print(f"\n{'='*60}")
    print("INTERESTING FINDINGS")
    print(f"{'='*60}")
    
    # Show nearest neighbors for some interesting words
    interesting_words = ['the', 'and', 'love', 'heart', 'time', 'world', 'life', 'death']
    available_interesting = [w for w in interesting_words if w in neighbors]
    
    if available_interesting:
        print("\nNearest neighbors for some key words:")
        for word in available_interesting[:5]:  # Show first 5
            print(f"\n'{word}' is most similar to:")
            for neighbor, sim in neighbors[word][:5]:
                print(f"  {neighbor:<15} (similarity: {sim:.3f})")
    
    # Show clusters
    print(f"\nWord clusters found:")
    for cluster_id, words in clusters.items():
        print(f"\nCluster {cluster_id} ({len(words)} words):")
        print(f"  {', '.join(words[:10])}" + 
              (f" ... and {len(words)-10} more" if len(words) > 10 else ""))

# =============================================================================
#  MAIN
# =============================================================================

def load_vocabulary_from_checkpoint() -> Tuple[List[str], np.ndarray]:
    """
    Load vocabulary and embeddings directly from the trained text encoder checkpoint.
    This is the most reliable way since it uses exactly what was saved during training.
    """
    print("Loading vocabulary and embeddings from checkpoint...")
    
    ckpt_path = os.path.join(CONTRASTIVE_OUT, "text_encoder.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Text encoder checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    if 'embeddings' not in checkpoint:
        raise RuntimeError("No embeddings found in checkpoint")
    
    raw_embeddings = checkpoint['embeddings']  # (V, base_dim)
    vocab_size = raw_embeddings.shape[0]
    
    print(f"  Found {vocab_size} words in checkpoint")
    print(f"  Raw embeddings shape: {raw_embeddings.shape}")
    
    # Apply the learned projection manually
    proj_layers = {}
    for key, value in checkpoint.items():
        if key.startswith('proj.'):
            proj_layers[key] = value
    
    if not proj_layers:
        print("  No projection layers found, using raw embeddings")
        return None, raw_embeddings.numpy()
    
    device = raw_embeddings.device
    
    # Simple 2-layer projection (matching TextEncoder architecture)
    if 'proj.0.weight' in proj_layers and 'proj.3.weight' in proj_layers:
        w1 = proj_layers['proj.0.weight'].to(device)  # (256, base_dim)
        b1 = proj_layers['proj.0.bias'].to(device)    # (256,)
        w2 = proj_layers['proj.3.weight'].to(device)  # (emb_dim, 256)
        b2 = proj_layers['proj.3.bias'].to(device)    # (emb_dim,)
        
        # Forward pass: raw -> proj -> final embeddings
        with torch.no_grad():
            h1 = torch.relu(raw_embeddings @ w1.T + b1)  # (V, 256)
            final_emb = h1 @ w2.T + b2                   # (V, emb_dim)
            final_emb = F.normalize(final_emb, dim=-1)   # L2 normalize
        
        print(f"  Final embeddings shape: {final_emb.shape}")
        
        # For words, we need to reconstruct them from the training dataset
        # But since we have the embeddings, we can create generic word labels
        words = [f"word_{i:03d}" for i in range(vocab_size)]
        
        return words, final_emb.cpu().numpy()
    
    raise RuntimeError("Could not reconstruct projection from checkpoint")

def load_embeddings_directly(vocab: Dict[str, int], words: List[str]) -> np.ndarray:
    """
    Alternative approach: load the trained text encoder and extract embeddings
    without rebuilding the base embeddings (useful if BERT fails).
    """
    print("Trying direct embedding extraction...")
    
    ckpt_path = os.path.join(CONTRASTIVE_OUT, "text_encoder.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Text encoder checkpoint not found: {ckpt_path}")
    
    # Load the checkpoint and inspect it
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # The TextEncoder stores embeddings in 'embeddings' buffer and has a projection
    if 'embeddings' in checkpoint:
        raw_embeddings = checkpoint['embeddings']  # (V, base_dim)
        print(f"  Found raw embeddings: {raw_embeddings.shape}")
        
        # Apply the learned projection manually
        proj_layers = {}
        for key, value in checkpoint.items():
            if key.startswith('proj.'):
                proj_layers[key] = value
        
        if proj_layers:
            # Reconstruct minimal projection
            device = raw_embeddings.device
            
            # Simple 2-layer projection (matching TextEncoder)
            if 'proj.0.weight' in proj_layers and 'proj.3.weight' in proj_layers:
                w1 = proj_layers['proj.0.weight'].to(device)  # (256, base_dim)
                b1 = proj_layers['proj.0.bias'].to(device)    # (256,)
                w2 = proj_layers['proj.3.weight'].to(device)  # (emb_dim, 256)
                b2 = proj_layers['proj.3.bias'].to(device)    # (emb_dim,)
                
                # Forward pass: raw -> proj -> final embeddings
                with torch.no_grad():
                    h1 = torch.relu(raw_embeddings @ w1.T + b1)  # (V, 256)
                    final_emb = h1 @ w2.T + b2                   # (V, emb_dim)
                    final_emb = F.normalize(final_emb, dim=-1)   # L2 normalize
                
                print(f"  Projected embeddings: {final_emb.shape}")
                return final_emb.cpu().numpy()
    
    raise RuntimeError("Could not extract embeddings directly from checkpoint")

def main():
    parser = argparse.ArgumentParser(description="Analyze learned word embeddings")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of nearest neighbors to find for each word")
    parser.add_argument("--n_clusters", type=int, default=8,
                        help="Number of clusters for K-means")
    parser.add_argument("--save_embeddings", action="store_true",
                        help="Save embeddings array to file")
    parser.add_argument("--max_viz_words", type=int, default=50,
                        help="Maximum words to show in similarity matrix")
    parser.add_argument("--skip_bert", action="store_true",
                        help="Skip BERT loading and try direct embedding extraction")
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Text encoder method: {TEXT_ENCODER}")
    print(f"Top-k neighbors: {args.top_k}")
    print(f"Number of clusters: {args.n_clusters}\n")
    
    # Extract embeddings using one of two approaches
    if args.skip_bert:
        print("Using direct embedding extraction from checkpoint...")
        words, embeddings = load_vocabulary_from_checkpoint()
    else:
        print("Using full model loading approach...")
        try:
            vocab, words = load_vocabulary()
            text_encoder = load_trained_text_encoder(vocab, words)
            embeddings = extract_all_embeddings(text_encoder, words)
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Falling back to direct embedding extraction...")
            words, embeddings = load_vocabulary_from_checkpoint()
    
    # Perform analysis
    print(f"\n{'='*60}")
    print("PERFORMING ANALYSIS")
    print(f"{'='*60}")
    
    similarity = compute_similarity_matrix(embeddings)
    neighbors = find_nearest_neighbors(embeddings, words, args.top_k)
    cluster_labels, clusters = perform_kmeans_clustering(
        embeddings, words, args.n_clusters
    )
    stats = analyze_distance_distribution(embeddings)
    
    # Create visualizations
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    plot_similarity_matrix(similarity, words, args.max_viz_words)
    plot_tsne_visualization(embeddings, words)
    plot_tsne_visualization(embeddings, words, cluster_labels)
    plot_distance_distribution(stats)
    plot_hierarchical_clustering(embeddings, words)
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    save_results(
        words, embeddings, neighbors, clusters, stats, 
        args.save_embeddings
    )
    
    # Print findings
    print_interesting_findings(neighbors, clusters)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {OUT_DIR}")
    print(f"Total words analyzed: {len(words)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

if __name__ == "__main__":
    main()