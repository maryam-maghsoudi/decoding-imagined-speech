"""
check_audio_embeddings.py
==========================
Extended diagnostic: extract embeddings from 4 audio stimuli using:
  1. Whisper (base)         — semantic / speech model
  2. wav2vec 2.0 (large)    — self-supervised speech model (used in BrainMagick)
  3. Acoustic features      — MFCCs, chroma, spectral centroid, RMS, ZCR
                              (hand-crafted, no neural network)

For each model / feature set we compute:
  - Pairwise cosine similarity matrix
  - PCA scatter
  - Within-category vs cross-category similarity gap

Goal: find which representation best separates all 4 stimuli individually,
not just melody-vs-poem.

Run
---
    python check_audio_embeddings.py

Requirements
------------
    pip install openai-whisper transformers torch torchaudio librosa matplotlib scikit-learn
"""

import os
import itertools
from pathlib import Path

import numpy as np
import torch
import whisper
import torchaudio
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# =============================================================================
#  CONFIG
# =============================================================================
AUDIO_DIR     = "/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/rnn/audio_wav"
OUT_DIR       = "./audio_embed_check"
WHISPER_MODEL = "base"
WAV2VEC_MODEL = "facebook/wav2vec2-large-xlsr-53"   # same as BrainMagick
TARGET_SR     = 16000   # wav2vec and whisper both expect 16kHz

STIMULUS_META = {
    "melody1": {"label": "melody1", "type": "melody"},
    "melody2": {"label": "melody2", "type": "melody"},
    "poem1":   {"label": "poem1",   "type": "poem"},
    "poem2":   {"label": "poem2",   "type": "poem"},
}

os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# =============================================================================
#  AUDIO DISCOVERY
# =============================================================================

def discover_audio(audio_dir: str) -> dict:
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
            found[f.stem] = str(f)

    print(f"Matched: {list(found.keys())}")
    return found


def load_waveform(filepath: str, target_sr: int = TARGET_SR):
    """Returns (waveform np.ndarray mono, sample_rate)."""
    wav, sr = torchaudio.load(filepath)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)   # stereo → mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy(), target_sr   # (T,)


# =============================================================================
#  1. WHISPER  (reuse from previous script, mean strategy only)
# =============================================================================

def extract_whisper(audio_files: dict) -> dict:
    print("\n--- Whisper embeddings ---")
    model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    model.eval()
    embs = {}
    with torch.no_grad():
        for key, fpath in audio_files.items():
            audio = whisper.load_audio(fpath)
            audio = whisper.pad_or_trim(audio)
            mel   = whisper.log_mel_spectrogram(
                audio, n_mels=model.dims.n_mels
            ).to(DEVICE)
            enc   = model.encoder(mel.unsqueeze(0))   # (1, T', D)
            embs[key] = enc.squeeze(0).mean(0).cpu().numpy()
            print(f"  {key}: dim={embs[key].shape[0]}")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return embs


# =============================================================================
#  2. wav2vec 2.0
# =============================================================================

def extract_wav2vec(audio_files: dict) -> dict:
    """
    Three strategies:
      mean_last    — mean pool last layer
      mean_last4   — mean pool of last 4 layers averaged
      mean_all     — mean pool all transformer layers averaged
    Returns dict: strategy → { key → np.ndarray }
    """
    print(f"\n--- wav2vec 2.0 embeddings ({WAV2VEC_MODEL}) ---")
    print("  Loading model (this may take a moment)...")
    # Use FeatureExtractor instead of Processor — avoids the CTC tokenizer
    # which requires a network download and isn't needed for embeddings
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_MODEL)
    model             = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL,
                                                      output_hidden_states=True)
    model             = model.to(DEVICE)
    model.eval()

    results = {"w2v_mean_last": {}, "w2v_mean_last4": {}, "w2v_mean_all": {}}

    with torch.no_grad():
        for key, fpath in audio_files.items():
            wav, sr = load_waveform(fpath)
            inputs  = feature_extractor(wav, sampling_rate=sr,
                                        return_tensors="pt", padding=True)
            inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}

            out     = model(**inputs)
            # out.hidden_states: tuple of (1, T, D) — one per layer + embedding
            hidden  = out.hidden_states   # len = n_layers + 1

            last_layer = out.last_hidden_state.squeeze(0).cpu().numpy()  # (T, D)
            last4      = np.stack([h.squeeze(0).cpu().numpy()
                                   for h in hidden[-4:]])                 # (4, T, D)
            all_layers = np.stack([h.squeeze(0).cpu().numpy()
                                   for h in hidden])                      # (L+1, T, D)

            results["w2v_mean_last"][key]  = last_layer.mean(0)
            results["w2v_mean_last4"][key] = last4.mean(axis=0).mean(0)
            results["w2v_mean_all"][key]   = all_layers.mean(axis=0).mean(0)

            print(f"  {key}: layers={len(hidden)}  "
                  f"T={last_layer.shape[0]}  D={last_layer.shape[1]}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results


# =============================================================================
#  3. ACOUSTIC FEATURES  (librosa)
# =============================================================================

def extract_acoustic(audio_files: dict) -> dict:
    """
    Hand-crafted features that capture timbre, pitch, rhythm:
      mfcc       — 40 MFCCs, mean over time  (captures timbre)
      chroma     — 12 chroma features, mean  (captures pitch class / harmony)
      combined   — concat of mfcc + chroma + spectral_centroid + rms + zcr
    """
    print("\n--- Acoustic (librosa) features ---")
    results = {"mfcc": {}, "chroma": {}, "acoustic_combined": {}}

    for key, fpath in audio_files.items():
        wav, sr = load_waveform(fpath)

        # MFCCs
        mfcc   = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40)     # (40, T)
        mfcc_m = mfcc.mean(axis=1)                                   # (40,)

        # Chroma
        chroma   = librosa.feature.chroma_stft(y=wav, sr=sr)        # (12, T)
        chroma_m = chroma.mean(axis=1)                               # (12,)

        # Spectral centroid
        sc   = librosa.feature.spectral_centroid(y=wav, sr=sr)      # (1, T)
        sc_m = sc.mean(axis=1)                                       # (1,)

        # RMS energy
        rms   = librosa.feature.rms(y=wav)                          # (1, T)
        rms_m = rms.mean(axis=1)                                     # (1,)

        # Zero-crossing rate
        zcr   = librosa.feature.zero_crossing_rate(y=wav)           # (1, T)
        zcr_m = zcr.mean(axis=1)                                     # (1,)

        combined = np.concatenate([mfcc_m, chroma_m, sc_m, rms_m, zcr_m])

        results["mfcc"][key]              = mfcc_m
        results["chroma"][key]            = chroma_m
        results["acoustic_combined"][key] = combined

        print(f"  {key}: mfcc={mfcc_m.shape[0]}  chroma={chroma_m.shape[0]}  "
              f"combined={combined.shape[0]}")

    return results


# =============================================================================
#  UTILITIES
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def pairwise_cosine_matrix(embeddings: dict):
    keys = list(embeddings.keys())
    N    = len(keys)
    mat  = np.zeros((N, N))
    for i, j in itertools.product(range(N), range(N)):
        mat[i, j] = cosine_sim(embeddings[keys[i]], embeddings[keys[j]])
    return keys, mat


def within_vs_cross_gap(keys: list, mat: np.ndarray) -> float:
    """
    Gap = mean(within-category sim) - mean(cross-category sim).
    Higher is better — means the model separates modalities.
    Categories: melody={melody1,melody2}, poem={poem1,poem2}
    """
    melody_idx = [i for i, k in enumerate(keys) if "melody" in k]
    poem_idx   = [i for i, k in enumerate(keys) if "poem"   in k]

    within, cross = [], []
    for i, j in itertools.combinations(range(len(keys)), 2):
        same_cat = (i in melody_idx and j in melody_idx) or \
                   (i in poem_idx   and j in poem_idx)
        (within if same_cat else cross).append(mat[i, j])

    return (np.mean(within) - np.mean(cross)) if cross else 0.0


def all4_discriminability(keys: list, mat: np.ndarray) -> float:
    """
    Mean off-diagonal distance from identity = mean(1 - sim(i,j)) for i≠j.
    Higher = stimuli are more spread out in embedding space = better for
    4-way classification.
    """
    N    = len(keys)
    vals = [1 - mat[i, j] for i in range(N) for j in range(N) if i != j]
    return float(np.mean(vals))


# =============================================================================
#  PLOTTING
# =============================================================================

def plot_sim_matrix(ax, mat, labels, title, gap, disc):
    im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"{title}\ngap={gap:.3f}  disc={disc:.3f}", fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_pca_scatter(ax, embeddings, stim_meta, title):
    keys = list(embeddings.keys())
    vecs = np.stack([embeddings[k] for k in keys])
    if vecs.shape[0] < 2:
        return
    n_comp = min(2, vecs.shape[0], vecs.shape[1])
    pca    = PCA(n_components=n_comp)
    proj   = pca.fit_transform(vecs)

    colors = {"melody": "#2196F3", "poem": "#E91E63"}
    for i, key in enumerate(keys):
        stype = stim_meta.get(key, {}).get("type", "unknown")
        color = colors.get(stype, "grey")
        ax.scatter(proj[i, 0], proj[i, 1] if n_comp > 1 else 0,
                   c=color, s=140, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(key,
                    (proj[i, 0], proj[i, 1] if n_comp > 1 else 0),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=7)
    if n_comp > 1:
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.axhline(0, color="grey", lw=0.4, linestyle="--")
    ax.axvline(0, color="grey", lw=0.4, linestyle="--")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2196F3", label="melody"),
        Patch(color="#E91E63", label="poem"),
    ], fontsize=6)


def make_all_plots(all_embeddings: dict) -> None:
    strategy_names = list(all_embeddings.keys())
    N = len(strategy_names)

    fig, axes = plt.subplots(2, N, figsize=(4 * N, 9))
    if N == 1:
        axes = axes[:, np.newaxis]

    for col, strat in enumerate(strategy_names):
        embs      = all_embeddings[strat]
        keys, mat = pairwise_cosine_matrix(embs)
        gap       = within_vs_cross_gap(keys, mat)
        disc      = all4_discriminability(keys, mat)

        plot_sim_matrix(axes[0, col], mat, keys, strat, gap, disc)
        plot_pca_scatter(axes[1, col], embs, STIMULUS_META, f"PCA — {strat}")

    plt.suptitle(
        "Audio embedding comparison: Whisper vs wav2vec 2.0 vs Acoustic\n"
        "gap = within−cross similarity  |  disc = mean pairwise distance (higher=better)",
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "audio_embedding_comparison.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\n[saved] {out}")


# =============================================================================
#  SUMMARY TABLE
# =============================================================================

def print_summary_table(all_embeddings: dict) -> None:
    print("\n" + "="*70)
    print(f"{'Strategy':<25}  {'gap':>8}  {'disc':>8}  "
          f"{'m1-m2':>8}  {'p1-p2':>8}  {'m1-p1':>8}")
    print("="*70)
    for strat, embs in all_embeddings.items():
        keys, mat = pairwise_cosine_matrix(embs)
        kidx      = {k: i for i, k in enumerate(keys)}
        gap       = within_vs_cross_gap(keys, mat)
        disc      = all4_discriminability(keys, mat)

        # specific pairs
        def s(a, b):
            if a in kidx and b in kidx:
                return f"{mat[kidx[a], kidx[b]]:8.4f}"
            return f"{'N/A':>8}"

        print(f"{strat:<25}  {gap:8.4f}  {disc:8.4f}  "
              f"{s('melody1','melody2')}  {s('poem1','poem2')}  "
              f"{s('melody1','poem1')}")

    print("="*70)
    print("gap  : within-category sim − cross-category sim (higher = melody/poem separation)")
    print("disc : mean pairwise distance (higher = all 4 stimuli spread out)")
    print("m1-m2: melody1 vs melody2 similarity (lower = more discriminable)")
    print("p1-p2: poem1 vs poem2 similarity     (lower = more discriminable)")


# =============================================================================
#  SAVE
# =============================================================================

def save_all(all_embeddings: dict) -> None:
    path = os.path.join(OUT_DIR, "all_audio_embeddings.npy")
    np.save(path, all_embeddings)
    print(f"[saved] {path}")

    # also save best strategy separately for easy loading in the full pipeline
    # (we'll pick after seeing results, but save all)
    for strat, embs in all_embeddings.items():
        p = os.path.join(OUT_DIR, f"{strat}_embeddings.npy")
        np.save(p, embs)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    audio_files = discover_audio(AUDIO_DIR)

    all_embeddings = {}

    # 1. Whisper
    all_embeddings["whisper_mean"] = extract_whisper(audio_files)

    # 2. wav2vec 2.0
    w2v = extract_wav2vec(audio_files)
    all_embeddings.update(w2v)

    # 3. Acoustic
    acou = extract_acoustic(audio_files)
    all_embeddings.update(acou)

    # summary table
    print_summary_table(all_embeddings)

    # plots
    make_all_plots(all_embeddings)

    # save
    save_all(all_embeddings)

    print("\n" + "="*70)
    print("WHAT TO LOOK FOR:")
    print("  Best for 4-way decoding  → highest 'disc' score")
    print("  Best melody/poem split   → highest 'gap' score")
    print("  m1-m2 and p1-p2 < 0.99  → within-category discrimination exists")
    print(f"\nOutputs saved to: {OUT_DIR}/")
    print("  audio_embedding_comparison.png")
    print("  all_audio_embeddings.npy")


if __name__ == "__main__":
    main()