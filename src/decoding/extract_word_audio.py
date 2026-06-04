"""
extract_word_audio.py
======================
Extract individual word audio segments from full poem recordings.

This script reads the word onset times and extracts audio segments
for each word, saving them as individual .wav files for use with
Whisper and Wav2Vec2 encoders.

Usage:
    python extract_word_audio.py

Output:
    ./audio/poem1/word.wav
    ./audio/poem2/word.wav
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import librosa
import soundfile as sf
import numpy as np


# =============================================================================
#  CONFIG
# =============================================================================
ONSET_DIR = "./onset_out"
AUDIO_SOURCE_DIR = "/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/rnn/audio_wav"
AUDIO_OUTPUT_DIR = "./audio"

POEM_KEYS = ["poem1", "poem2"]

# Audio extraction settings
SAMPLE_RATE = 16000  # Target sample rate for Whisper/Wav2Vec2
PADDING_MS = 50      # Add padding around word boundaries (ms)


def load_word_onsets(poem_key: str) -> List[Dict]:
    """Load word onset times from JSON file."""
    onset_file = os.path.join(ONSET_DIR, f"{poem_key}_word_onsets.json")
    if not os.path.exists(onset_file):
        raise FileNotFoundError(f"Onset file not found: {onset_file}")
    
    with open(onset_file, 'r') as f:
        return json.load(f)


def extract_word_segments(
    audio_path: str,
    word_onsets: List[Dict],
    output_dir: str,
    sample_rate: int = SAMPLE_RATE,
    padding_ms: int = PADDING_MS,
) -> None:
    """
    Extract individual word audio segments from full recording.
    
    Parameters
    ----------
    audio_path : str
        Path to full poem audio file
    word_onsets : list
        List of {word: str, start: float, end: float} dicts
    output_dir : str
        Directory to save individual word files
    sample_rate : int
        Target sample rate
    padding_ms : int
        Padding around word boundaries in milliseconds
    """
    print(f"  Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio_duration = len(audio) / sr
    
    padding_samples = int(padding_ms * sample_rate / 1000)
    
    os.makedirs(output_dir, exist_ok=True)
    extracted = 0
    skipped = 0
    
    for word_info in word_onsets:
        word = word_info["word"].strip().lower()
        start_s = word_info["start"]
        end_s = word_info["end"] if "end" in word_info else start_s + 0.5  # fallback duration
        
        # Skip very short words or invalid times
        if len(word) < 1 or start_s < 0 or end_s <= start_s:
            skipped += 1
            continue
            
        # Convert to samples with padding
        start_sample = max(0, int(start_s * sample_rate) - padding_samples)
        end_sample = min(len(audio), int(end_s * sample_rate) + padding_samples)
        
        # Skip if segment is too short or extends beyond audio
        if end_sample - start_sample < sample_rate * 0.1:  # minimum 100ms
            print(f"    WARNING: Skipping '{word}' - segment too short ({end_s-start_s:.3f}s)")
            skipped += 1
            continue
            
        if start_s >= audio_duration:
            print(f"    WARNING: Skipping '{word}' - onset beyond audio duration")
            skipped += 1
            continue
        
        # Extract segment
        segment = audio[start_sample:end_sample]
        
        # Save as individual wav file
        output_path = os.path.join(output_dir, f"{word}.wav")
        sf.write(output_path, segment, sample_rate)
        extracted += 1
    
    print(f"  Extracted: {extracted} words, Skipped: {skipped} words")
    return extracted, skipped


def main():
    """Extract word audio segments for all poems."""
    print("Extracting individual word audio segments...")
    print(f"Source audio: {AUDIO_SOURCE_DIR}")
    print(f"Output dir:   {AUDIO_OUTPUT_DIR}")
    print(f"Sample rate:  {SAMPLE_RATE} Hz")
    print(f"Padding:      ±{PADDING_MS} ms\n")
    
    total_extracted = 0
    total_skipped = 0
    
    for poem_key in POEM_KEYS:
        print(f"Processing {poem_key}...")
        
        # Load word onset times
        try:
            word_onsets = load_word_onsets(poem_key)
            print(f"  Found {len(word_onsets)} word onsets")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue
        
        # Find source audio file
        audio_path = os.path.join(AUDIO_SOURCE_DIR, f"{poem_key}.wav")
        if not os.path.exists(audio_path):
            print(f"  ERROR: Audio file not found: {audio_path}")
            continue
        
        # Extract segments
        output_dir = os.path.join(AUDIO_OUTPUT_DIR, poem_key)
        extracted, skipped = extract_word_segments(
            audio_path, word_onsets, output_dir,
            sample_rate=SAMPLE_RATE, padding_ms=PADDING_MS
        )
        
        total_extracted += extracted
        total_skipped += skipped
        print()
    
    print(f"Summary:")
    print(f"  Total extracted: {total_extracted}")
    print(f"  Total skipped:   {total_skipped}")
    print(f"  Output directory: {AUDIO_OUTPUT_DIR}")
    
    # Show directory structure
    if total_extracted > 0:
        print(f"\nDirectory structure:")
        for poem_key in POEM_KEYS:
            poem_dir = os.path.join(AUDIO_OUTPUT_DIR, poem_key)
            if os.path.exists(poem_dir):
                wav_files = [f for f in os.listdir(poem_dir) if f.endswith('.wav')]
                print(f"  {poem_dir}: {len(wav_files)} .wav files")
                if len(wav_files) <= 10:  # Show first few files as example
                    for f in wav_files[:5]:
                        print(f"    - {f}")
                    if len(wav_files) > 5:
                        print(f"    ... and {len(wav_files) - 5} more")


if __name__ == "__main__":
    main()