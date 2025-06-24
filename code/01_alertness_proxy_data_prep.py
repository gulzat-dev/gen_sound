# 01_alertness_proxy_data_prep.py
import shutil
import sys
from collections import Counter

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

try:
    from gen_sound.config import (
        DATA_RAW_DIR, BASE_SEGMENTS_DIR, SAMPLE_RATE, DURATION_S,
        ESC50_MAP, URBANSOUND8K_MAP
    )
except ImportError:
    print("Error: config.py not found.")
    sys.exit(1)


def process_and_segment_audio(filepath, target_sr, duration_s):
    """Loads, resamples, converts to mono, and segments a single audio file."""
    try:
        y, sr = librosa.load(filepath, sr=None)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        if y.ndim > 1:
            y = librosa.to_mono(y)
        return librosa.util.fix_length(y, size=int(target_sr * duration_s))
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def main():
    """Creates the base dataset of segmented audio files without augmentation."""
    print("--- Starting Base Data Preparation (No Augmentation) ---")

    if BASE_SEGMENTS_DIR.exists():
        print(f"Cleaning previous base data in {BASE_SEGMENTS_DIR}")
        shutil.rmtree(BASE_SEGMENTS_DIR)

    proxy_classes = set(ESC50_MAP.values()) | set(URBANSOUND8K_MAP.values())
    for proxy_class in proxy_classes:
        (BASE_SEGMENTS_DIR / proxy_class).mkdir(parents=True, exist_ok=True)

    segment_counts = Counter()

    # Process ESC-50
    esc50_meta_path = DATA_RAW_DIR / "ESC-50-master" / "meta" / "esc50.csv"
    if esc50_meta_path.exists():
        df = pd.read_csv(esc50_meta_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="ESC-50 Base Prep"):
            if row['category'] in ESC50_MAP:
                proxy_label = ESC50_MAP[row['category']]
                audio_path = DATA_RAW_DIR / "ESC-50-master" / "audio" / row['filename']
                audio = process_and_segment_audio(audio_path, SAMPLE_RATE, DURATION_S)
                if audio is not None:
                    sf.write(BASE_SEGMENTS_DIR / proxy_label / f"esc50_{row['filename']}", audio, SAMPLE_RATE)
                    segment_counts[proxy_label] += 1

    # Process UrbanSound8K
    us8k_meta_path = DATA_RAW_DIR / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
    if us8k_meta_path.exists():
        df = pd.read_csv(us8k_meta_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="UrbanSound8K Base Prep"):
            if row['class'] in URBANSOUND8K_MAP:
                proxy_label = URBANSOUND8K_MAP[row['class']]
                path = DATA_RAW_DIR / "UrbanSound8K" / "audio" / f"fold{row['fold']}" / row['slice_file_name']
                audio = process_and_segment_audio(path, SAMPLE_RATE, DURATION_S)
                if audio is not None:
                    sf.write(BASE_SEGMENTS_DIR / proxy_label / f"us8k_{row['slice_file_name']}", audio, SAMPLE_RATE)
                    segment_counts[proxy_label] += 1

    print("\n--- Base Data Preparation Complete ---")
    print("Summary of generated base segments:")
    for category, count in sorted(segment_counts.items()):
        print(f"- {category}: {count} segments")
    print(f"\nTotal base segments created: {sum(segment_counts.values())}")


if __name__ == "__main__":
    main()