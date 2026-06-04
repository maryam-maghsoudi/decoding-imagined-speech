"""
test_audio_loading.py
=====================
Test that word audio files can be loaded correctly for Whisper/Wav2Vec2.
"""

import os
import json
from typing import Dict, List
import librosa
import torch

ONSET_DIR = "./onset_out"
AUDIO_DIR = "./audio"
POEM_KEYS = ["poem1", "poem2"]

def test_audio_loading():
    """Test loading individual word audio files."""
    print("Testing audio file loading...")
    
    # Build word list and audio paths like in the main script
    word_audio: Dict[str, str] = {}
    all_words = set()
    
    for poem_key in POEM_KEYS:
        onset_file = os.path.join(ONSET_DIR, f"{poem_key}_word_onsets.json")
        if os.path.exists(onset_file):
            with open(onset_file) as f:
                word_onsets = json.load(f)
            
            for w in word_onsets:
                word = w["word"].strip().lower()
                all_words.add(word)
                
                if word not in word_audio:
                    audio_path = os.path.join(AUDIO_DIR, poem_key, f"{word}.wav")
                    if os.path.exists(audio_path):
                        word_audio[word] = audio_path
    
    words = sorted(all_words)
    print(f"Found {len(words)} unique words")
    print(f"Found {len(word_audio)} words with audio files")
    
    # Test loading a few audio files
    print("\nTesting audio loading:")
    test_words = words[:5]  # Test first 5 words
    
    for word in test_words:
        audio_path = word_audio.get(word)
        if audio_path and os.path.exists(audio_path):
            try:
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                duration = len(audio) / sr
                print(f"  ✓ '{word}': {duration:.3f}s, {len(audio)} samples")
            except Exception as e:
                print(f"  ✗ '{word}': ERROR - {e}")
        else:
            print(f"  ✗ '{word}': No audio file found")
    
    # Check coverage
    missing = len(words) - len(word_audio)
    print(f"\nSummary:")
    print(f"  Total words: {len(words)}")
    print(f"  With audio:  {len(word_audio)}")
    print(f"  Missing:     {missing}")
    
    if missing > 0:
        missing_words = [w for w in words if w not in word_audio]
        print(f"  Missing words: {missing_words[:10]}{'...' if len(missing_words) > 10 else ''}")

if __name__ == "__main__":
    test_audio_loading()