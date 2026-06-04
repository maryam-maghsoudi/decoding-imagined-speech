"""
check_whisper_embeddings.py
============================
Step 1 diagnostic: load the 4 audio stimuli, extract Whisper embeddings,
and visualise them so we can sanity-check the embedding space before
building the full contrastive decoder.

What this script does
---------------------
1. Discovers the 4 .wav files under AUDIO_DIR
2. Loads OpenAI Whisper (base or medium — configurable)
3. Extracts encoder hidden states for each file using THREE strategies:
     a. mean-pool over the full encoder output          → "mean"
     b. last-layer CLS-equivalent (first token)         → "first_token"
     c. mean of last 4 encoder layers                   → "last4_mean"
4. Prints embedding shapes and basic stats
5. Computes all 6 pairwise cosine similarities
6. Saves a figure with:
     - 3×3 grid of cosine-similarity matrices (one per strategy)
     - 2D PCA scatter of each strategy
     - A table of all pairwise cosine similarities
7. Saves embeddings to disk as .npy for use in the full pipeline

Run
---
    python check_whisper_embeddings.py

Requirements
------------
    pip install openai-whisper torch torchaudio matplotlib scikit-learn
"""

import os
import itertools
from pathlib import Path
import pdb
# pdb.set_trace()

import numpy as np
import torch
import whisper
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =============================================================================
#  CONFIG  — edit these if needed
# =============================================================================
AUDIO_DIR   = "/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/rnn/audio_wav"
OUT_DIR     = "./whisper_embed_check"
WHISPER_MODEL = "base"   # "base" | "small" | "medium" — start with base, fast

# Label each file as melody or poem so we can colour the plots
# Keys are substrings that should appear in the filename (case-insensitive)
STIMULUS_META = {
    "melody1": {"label": "melody1", "type": "melody"},
    "melody2": {"label": "melody2", "type": "melody"},
    "poem1":   {"label": "poem1",   "type": "poem"},
    "poem2":   {"label": "poem2",   "type": "poem"},
}

os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
#  1.  DISCOVER AUDIO FILES
# =============================================================================

def discover_audio(audio_dir: str) -> dict:
    """
    Returns dict: stimulus_key → filepath
    Tries to match filenames to keys in STIMULUS_META.
    """
    found = {}
    wav_files = sorted(Path(audio_dir).glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")

    print(f"\nFound {len(wav_files)} .wav files:")
    for f in wav_files:
        print(f"  {f.name}")
        for key in STIMULUS_META:
            if key.lower() in f.name.lower():
                found[key] = str(f)
                break
        else:
            # fallback: use filename stem as key
            found[f.stem] = str(f)

    print(f"\nMatched stimuli: {list(found.keys())}")
    if len(found) < 2:
        raise ValueError("Could not match audio files to stimulus keys — "
                         "check STIMULUS_META or filenames.")
    return found


# =============================================================================
#  2.  WHISPER EMBEDDING EXTRACTION
# =============================================================================

def load_audio_for_whisper(filepath: str, model: whisper.Whisper) -> torch.Tensor:
    """
    Load a wav file and return a padded/trimmed mel spectrogram tensor
    ready for Whisper encoder input.
    Shape: (1, n_mels, n_frames)
    """
    audio = whisper.load_audio(filepath)
    audio = whisper.pad_or_trim(audio)
    mel   = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(DEVICE)
    return mel.unsqueeze(0)   # (1, n_mels, n_frames)


def extract_whisper_embeddings(
    audio_files: dict,
    model:       whisper.Whisper,
) -> dict:
    """
    Returns dict:
      strategy → { stimulus_key → np.ndarray (D,) }

    Strategies:
      "mean"       — mean pool over time of final encoder layer
      "first_tok"  — first time-step of final encoder layer
      "last4_mean" — mean over time, averaged across last 4 encoder layers
    """
    model.eval()

    results = {
        "mean":      {},
        "first_tok": {},
        "last4_mean": {},
    }

    print(f"\nExtracting Whisper ({WHISPER_MODEL}) embeddings...")

    with torch.no_grad():
        for key, fpath in audio_files.items():
            print(f"  processing: {key}  ({Path(fpath).name})")
            mel = load_audio_for_whisper(fpath, model)   # (1, n_mels, T)

            # ---- hook to capture intermediate layer outputs ----
            layer_outputs = []
            hooks = []
            for layer in model.encoder.blocks:
                def _hook(module, inp, out, _store=layer_outputs):
                    _store.append(out.detach().cpu())
                hooks.append(layer.register_forward_hook(_hook))

            # forward pass through encoder only
            enc_out = model.encoder(mel)   # (1, T', D)

            for h in hooks:
                h.remove()

            enc_np = enc_out.squeeze(0).cpu().numpy()   # (T', D)

            # strategy a: mean pool final layer
            results["mean"][key] = enc_np.mean(axis=0)

            # strategy b: first token of final layer
            results["first_tok"][key] = enc_np[0]

            # strategy c: mean of last 4 layers, then mean-pool over time
            last4 = layer_outputs[-4:]                         # list of (1, T', D)
            last4_np = np.stack([l.squeeze(0).numpy() for l in last4], axis=0)
            # shape: (4, T', D)
            results["last4_mean"][key] = last4_np.mean(axis=0).mean(axis=0)

            print(f"    embedding dim: {enc_np.shape[1]}  "
                  f"time steps: {enc_np.shape[0]}")

    return results


# =============================================================================
#  3.  COSINE SIMILARITY
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a / (np.linalg.norm(a) + 1e-12), b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def pairwise_cosine_matrix(embeddings: dict) -> tuple:
    """
    embeddings: { key → np.ndarray (D,) }
    Returns (keys, matrix (N, N))
    """
    keys = list(embeddings.keys())
    N    = len(keys)
    mat  = np.zeros((N, N))
    for i, j in itertools.product(range(N), range(N)):
        mat[i, j] = cosine_sim(embeddings[keys[i]], embeddings[keys[j]])
    return keys, mat


# =============================================================================
#  4.  PLOTTING
# =============================================================================

def plot_similarity_matrix(
    ax:     plt.Axes,
    mat:    np.ndarray,
    labels: list,
    title:  str,
) -> None:
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_pca(
    ax:          plt.Axes,
    embeddings:  dict,
    stim_meta:   dict,
    title:       str,
) -> None:
    keys  = list(embeddings.keys())
    vecs  = np.stack([embeddings[k] for k in keys])
    if vecs.shape[0] < 2:
        ax.set_title(f"{title}\n(need ≥2 points for PCA)")
        return

    n_components = min(2, vecs.shape[0], vecs.shape[1])
    pca   = PCA(n_components=n_components)
    proj  = pca.fit_transform(vecs)   # (N, 2)

    colors = {"melody": "#2196F3", "poem": "#E91E63"}
    for i, key in enumerate(keys):
        stype = stim_meta.get(key, {}).get("type", "unknown")
        color = colors.get(stype, "grey")
        ax.scatter(proj[i, 0], proj[i, 1] if n_components > 1 else 0,
                   c=color, s=120, zorder=3)
        ax.annotate(key, (proj[i, 0], proj[i, 1] if n_components > 1 else 0),
                    fontsize=8, ha="left", va="bottom")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    if n_components > 1:
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title, fontsize=9)
    ax.axhline(0, color="grey", lw=0.5, linestyle="--")
    ax.axvline(0, color="grey", lw=0.5, linestyle="--")

    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2196F3", label="melody"),
        Patch(color="#E91E63", label="poem"),
    ], fontsize=7)


def make_plots(
    all_embeddings: dict,
    audio_files:    dict,
    stim_meta:      dict,
) -> None:
    strategies = list(all_embeddings.keys())
    N_strat    = len(strategies)

    fig, axes = plt.subplots(2, N_strat, figsize=(6 * N_strat, 10))
    if N_strat == 1:
        axes = axes[:, np.newaxis]

    for col, strat in enumerate(strategies):
        embs        = all_embeddings[strat]
        keys, mat   = pairwise_cosine_matrix(embs)

        # row 0: similarity matrix
        plot_similarity_matrix(axes[0, col], mat, keys,
                               title=f"Cosine sim — {strat}")

        # row 1: PCA
        plot_pca(axes[1, col], embs, stim_meta,
                 title=f"PCA — {strat}")

    plt.suptitle(
        f"Whisper ({WHISPER_MODEL}) embedding diagnostic\n"
        f"Audio dir: {AUDIO_DIR}",
        fontsize=11,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "whisper_embedding_check.png")
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"\n[saved] {out}")


# =============================================================================
#  5.  PRINT SUMMARY TABLE
# =============================================================================

def print_summary(all_embeddings: dict) -> None:
    print("\n" + "="*60)
    print("PAIRWISE COSINE SIMILARITIES")
    print("="*60)
    for strat, embs in all_embeddings.items():
        keys, mat = pairwise_cosine_matrix(embs)
        print(f"\n  Strategy: {strat}")
        header = f"{'':12s}" + "".join(f"{k:12s}" for k in keys)
        print(f"  {header}")
        for i, ki in enumerate(keys):
            row = f"  {ki:12s}" + "".join(f"{mat[i,j]:12.4f}" for j in range(len(keys)))
            print(row)

    print("\n" + "="*60)
    print("EMBEDDING NORMS (should all be similar)")
    print("="*60)
    for strat, embs in all_embeddings.items():
        print(f"\n  Strategy: {strat}")
        for key, vec in embs.items():
            print(f"    {key:12s}  norm={np.linalg.norm(vec):.4f}  "
                  f"mean={vec.mean():.4f}  std={vec.std():.4f}  "
                  f"dim={vec.shape[0]}")


# =============================================================================
#  6.  SAVE EMBEDDINGS
# =============================================================================

def save_embeddings(all_embeddings: dict) -> None:
    for strat, embs in all_embeddings.items():
        out = os.path.join(OUT_DIR, f"whisper_{WHISPER_MODEL}_{strat}_embeddings.npy")
        # save as dict: { key: array }
        np.save(out, embs)
        print(f"[saved] {out}")

    # also save a combined dict for easy loading later
    combined_path = os.path.join(OUT_DIR, "all_embeddings.npy")
    np.save(combined_path, all_embeddings)
    print(f"[saved] {combined_path}")


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Whisper model: {WHISPER_MODEL}")
    print(f"Audio dir: {AUDIO_DIR}")
    print(f"Output dir: {OUT_DIR}")

    # 1. find audio files
    audio_files = discover_audio(AUDIO_DIR)

    # 2. load whisper
    print(f"\nLoading Whisper ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    print(f"  encoder layers : {len(model.encoder.blocks)}")
    print(f"  embedding dim  : {model.dims.n_audio_state}")

    # 3. extract embeddings
    all_embeddings = extract_whisper_embeddings(audio_files, model)

    # 4. print summary
    print_summary(all_embeddings)

    # 5. plot
    make_plots(all_embeddings, audio_files, STIMULUS_META)

    # 6. save
    print("\nSaving embeddings...")
    save_embeddings(all_embeddings)

    print("\nDone. Check whisper_embed_check/ for:")
    print("  whisper_embedding_check.png  — similarity matrices + PCA")
    print("  whisper_*_embeddings.npy     — per-strategy embeddings")
    print("  all_embeddings.npy           — combined dict")
    print("\nWhat to look for:")
    print("  - melody1 & melody2 should be more similar to each other than to poems")
    print("  - poem1 & poem2 should be more similar to each other than to melodies")
    print("  - PCA should show melody/poem clusters")
    print("  - If structure looks wrong, try a larger Whisper model (small/medium)")


if __name__ == "__main__":
    main()