"""
extract_word_onsets.py
======================
Use WhisperX forced alignment to extract word-level onset/offset timestamps
for the poem audio files, then visualise the waveform with word onset markers
so you can visually verify the alignment.

What this script does
---------------------
1. Discovers poem .wav files under AUDIO_DIR (melody files are skipped)
2. Runs WhisperX forced alignment using the provided transcripts
3. Saves word-level timestamps to:
     onset_out/{poem_name}_word_onsets.csv
     onset_out/{poem_name}_word_onsets.json
4. Plots waveform + vertical lines at each word onset and saves:
     onset_out/{poem_name}_alignment_check.png

Install
-------
    pip install whisperx
    # whisperx requires torch + torchaudio already in your env

    # if whisperx is not available on the cluster, fallback is used:
    # pip install stable-whisper   (alternative forced aligner)

Run
---
    python extract_word_onsets.py

Config
------
Edit AUDIO_DIR, TRANSCRIPT_DIR, OUT_DIR below.
Transcripts should be plain .txt files named the same as the audio:
    poem1.wav  →  poem1.txt
    poem2.wav  →  poem2.txt
"""

import os
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
#  CONFIG  — edit these
# =============================================================================
AUDIO_DIR      = "/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/rnn/audio_wav"
TRANSCRIPT_DIR = "/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/contrastive_learning/poem_transcript"
OUT_DIR        = "./onset_out"
LANGUAGE       = "en"        # language of poems
WHISPERX_MODEL = "medium"      # "base" | "small" | "medium" — larger = more accurate
DEVICE         = "cpu"
COMPUTE_TYPE   = "int8"    # float16 not supported on all GPUs; int8 works everywhere

# Only process files matching these keys (skip melodies)
POEM_KEYS = ["poem1", "poem2"]

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
#  AUDIO DISCOVERY
# =============================================================================

def discover_poems(audio_dir: str, transcript_dir: str) -> list:
    """
    Returns list of dicts:
      { "key": str, "audio_path": str, "transcript_path": str, "transcript": str }
    """
    found = []
    for key in POEM_KEYS:
        # find wav
        wav_matches = list(Path(audio_dir).glob(f"*{key}*.wav"))
        if not wav_matches:
            print(f"  WARNING: no wav found for {key} in {audio_dir}")
            continue
        wav_path = str(wav_matches[0])

        # find transcript
        txt_matches = list(Path(transcript_dir).glob(f"*{key}*.txt"))
        if not txt_matches:
            print(f"  WARNING: no transcript found for {key} in {transcript_dir}")
            print(f"           expected a .txt file containing '{key}' in the name")
            print(f"           will attempt transcription-only mode (no forced alignment)")
            transcript = None
            txt_path   = None
        else:
            txt_path   = str(txt_matches[0])
            transcript = Path(txt_path).read_text().strip()
            print(f"  {key}: audio={Path(wav_path).name}  "
                  f"transcript={Path(txt_path).name}  "
                  f"({len(transcript.split())} words)")

        found.append({
            "key":             key,
            "audio_path":      wav_path,
            "transcript_path": txt_path,
            "transcript":      transcript,
        })
    return found


# =============================================================================
#  WHISPERX ALIGNMENT
# =============================================================================

def run_whisperx(poem_info: dict) -> list:
    """
    Returns list of word dicts:
      { "word": str, "start": float (s), "end": float (s) }
    Uses forced alignment if transcript is provided, else ASR only.
    """
    import whisperx

    key        = poem_info["key"]
    audio_path = poem_info["audio_path"]
    transcript = poem_info["transcript"]

    print(f"\n  [{key}] Loading WhisperX model ({WHISPERX_MODEL})...")
    model = whisperx.load_model(
        WHISPERX_MODEL, DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,
    )

    print(f"  [{key}] Transcribing / aligning...")
    audio = whisperx.load_audio(audio_path)

    if transcript:
        # ----- forced alignment mode -----
        # first get segment boundaries from whisper
        result = model.transcribe(audio, batch_size=8, language=LANGUAGE)

        # override the transcript with ground truth
        for seg in result["segments"]:
            seg["text"] = transcript   # crude but works for short poems

        # load alignment model
        align_model, metadata = whisperx.load_align_model(
            language_code=LANGUAGE, device=DEVICE
        )
        aligned = whisperx.align(
            result["segments"], align_model, metadata,
            audio, DEVICE, return_char_alignments=False
        )
        words = aligned["word_segments"]

    else:
        # ----- ASR only mode -----
        result = model.transcribe(audio, batch_size=8, language=LANGUAGE)
        align_model, metadata = whisperx.load_align_model(
            language_code=LANGUAGE, device=DEVICE
        )
        aligned = whisperx.align(
            result["segments"], align_model, metadata,
            audio, DEVICE, return_char_alignments=False
        )
        words = aligned["word_segments"]

    # normalise output format
    clean = []
    for w in words:
        if "word" not in w:
            continue
        clean.append({
            "word":  w["word"].strip().lower(),
            "start": float(w.get("start", 0.0)),
            "end":   float(w.get("end",   0.0)),
        })

    print(f"  [{key}] Aligned {len(clean)} words")
    return clean


# =============================================================================
#  FALLBACK: stable-whisper
# =============================================================================

def run_stable_whisper(poem_info: dict) -> list:
    """Fallback if whisperx is not available."""
    import stable_whisper

    key        = poem_info["key"]
    audio_path = poem_info["audio_path"]

    print(f"\n  [{key}] Using stable-whisper fallback...")
    model  = stable_whisper.load_model(WHISPERX_MODEL)
    result = model.align(audio_path, poem_info["transcript"], language=LANGUAGE)

    clean = []
    for w in result.all_words():
        clean.append({
            "word":  w.word.strip().lower(),
            "start": float(w.start),
            "end":   float(w.end),
        })
    print(f"  [{key}] Aligned {len(clean)} words")
    return clean


# =============================================================================
#  SAVE ONSETS
# =============================================================================

def save_onsets(key: str, words: list) -> None:
    # CSV
    csv_path = os.path.join(OUT_DIR, f"{key}_word_onsets.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "start", "end", "duration"])
        writer.writeheader()
        for w in words:
            writer.writerow({
                "word":     w["word"],
                "start":    f"{w['start']:.4f}",
                "end":      f"{w['end']:.4f}",
                "duration": f"{w['end']-w['start']:.4f}",
            })
    print(f"  [saved] {csv_path}")

    # JSON
    json_path = os.path.join(OUT_DIR, f"{key}_word_onsets.json")
    with open(json_path, "w") as f:
        json.dump(words, f, indent=2)
    print(f"  [saved] {json_path}")


# =============================================================================
#  PLOTTING
# =============================================================================

def plot_alignment(key: str, audio_path: str, words: list) -> None:
    """
    Two-panel figure:
      Top:    full waveform + word onset lines + word labels
      Bottom: zoomed first 10 seconds for fine-grained inspection
    """
    wav, sr = torchaudio.load(audio_path)
    wav     = wav.mean(dim=0).numpy()   # mono
    times   = np.arange(len(wav)) / sr

    fig, axes = plt.subplots(2, 1, figsize=(20, 8))

    for ax_idx, (ax, xlim) in enumerate(zip(
        axes,
        [(0, times[-1]), (0, min(10.0, times[-1]))]
    )):
        ax.plot(times, wav, color="#4A90D9", linewidth=0.5, alpha=0.8)
        ax.set_xlim(xlim)
        ax.set_ylim(wav.min() * 1.2, wav.max() * 1.2)

        # word onset lines
        for w in words:
            t = w["start"]
            if xlim[0] <= t <= xlim[1]:
                ax.axvline(t, color="#E74C3C", linewidth=1.0,
                           alpha=0.8, linestyle="--")
                ax.text(
                    t, wav.max() * 1.05,
                    w["word"],
                    rotation=90, fontsize=6 if ax_idx == 0 else 8,
                    va="bottom", ha="right", color="#C0392B",
                    clip_on=True,
                )

        label = ("Full waveform" if ax_idx == 0
                 else f"Zoom: first {xlim[1]:.0f}s")
        ax.set_title(f"{key} — {label}  ({len(words)} words aligned)",
                     fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)

    plt.suptitle(
        f"WhisperX forced alignment — {key}\n"
        f"Red dashed lines = word onsets  |  Check labels match waveform",
        fontsize=11,
    )
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{key}_alignment_check.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


# =============================================================================
#  PRINT WORD TABLE
# =============================================================================

def print_word_table(key: str, words: list) -> None:
    print(f"\n  [{key}] Word timestamps:")
    print(f"  {'#':>3}  {'word':<20}  {'start':>7}  {'end':>7}  {'dur':>6}")
    print(f"  {'-'*50}")
    for i, w in enumerate(words):
        dur = w["end"] - w["start"]
        print(f"  {i+1:>3}  {w['word']:<20}  "
              f"{w['start']:>7.3f}  {w['end']:>7.3f}  {dur:>6.3f}s")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print(f"Device : {DEVICE}")
    print(f"Model  : {WHISPERX_MODEL}")
    print(f"OutDir : {OUT_DIR}\n")

    poems = discover_poems(AUDIO_DIR, TRANSCRIPT_DIR)
    if not poems:
        raise RuntimeError("No poem files found. Check AUDIO_DIR and POEM_KEYS.")

    all_onsets = {}

    for poem_info in poems:
        key = poem_info["key"]
        print(f"\n{'='*60}")
        print(f"  Processing: {key}")
        print(f"{'='*60}")

        # try whisperx first, fall back to stable-whisper
        try:
            words = run_whisperx(poem_info)
        except ImportError:
            print("  whisperx not found, trying stable-whisper...")
            try:
                words = run_stable_whisper(poem_info)
            except ImportError:
                raise ImportError(
                    "Neither whisperx nor stable-whisper is installed.\n"
                    "Run: pip install whisperx\n"
                    "  or: pip install stable-whisper"
                )

        if not words:
            print(f"  WARNING: no words aligned for {key}, skipping")
            continue

        all_onsets[key] = words

        print_word_table(key, words)
        save_onsets(key, words)
        plot_alignment(key, poem_info["audio_path"], words)

    # save combined file for easy loading in contrastive pipeline
    combined_path = os.path.join(OUT_DIR, "all_word_onsets.json")
    with open(combined_path, "w") as f:
        json.dump(all_onsets, f, indent=2)
    print(f"\n[saved] {combined_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, words in all_onsets.items():
        starts = [w["start"] for w in words]
        durs   = [w["end"] - w["start"] for w in words]
        print(f"  {key}: {len(words)} words  "
              f"first={starts[0]:.2f}s  last={starts[-1]:.2f}s  "
              f"mean_dur={np.mean(durs):.3f}s")

    print(f"\nOutputs in {OUT_DIR}/:")
    print("  {poem}_word_onsets.csv      — load into Excel/pandas")
    print("  {poem}_word_onsets.json     — load in Python")
    print("  {poem}_alignment_check.png  — visual check")
    print("  all_word_onsets.json        — combined, use in contrastive pipeline")
    print("\nNext step: check the alignment plots carefully.")
    print("If onsets look off, try WHISPERX_MODEL = 'small' or 'medium'")


if __name__ == "__main__":
    main()