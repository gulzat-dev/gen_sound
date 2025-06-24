# 02_alertness_feature_extraction.py
import sys

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from gen_sound.config import (
        FINAL_AUDIO_DIR, FEATURES_DIR, SAMPLE_RATE, N_FFT, HOP_LENGTH_FFT,
        N_MFCC, N_MELS
    )
except ImportError:
    print("Error: config.py not found.")
    sys.exit(1)

CNN_FEATURES_DIR = FEATURES_DIR / "cnn_spectrograms"


def extract_features(audio_path):
    """Extracts aggregated features AND the full log-mel spectrogram for the CNN."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH_FFT)
        delta_mfccs = librosa.feature.delta(mfccs)
        mfcc_delta_aggregated = np.hstack(
            (np.mean(mfccs, axis=1), np.mean(delta_mfccs, axis=1), np.std(mfccs, axis=1), np.std(delta_mfccs, axis=1)))

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH_FFT, n_mels=N_MELS)
        log_mel_spec_2d = librosa.power_to_db(mel_spec, ref=np.max)

        log_mel_aggregated = np.hstack((np.mean(log_mel_spec_2d, axis=1), np.std(log_mel_spec_2d, axis=1)))

        return mfcc_delta_aggregated, log_mel_aggregated, log_mel_spec_2d
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None, None, None


def main():
    print("--- Starting Feature Extraction (Classic + CNN) ---")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    CNN_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if not FINAL_AUDIO_DIR.exists():
        print(f"Error: Final audio directory not found. Run previous scripts.")
        return

    features_mfcc, features_log_mel= [], []
    cnn_feature_metadata = []

    for audio_path in tqdm(list(FINAL_AUDIO_DIR.glob('*/*.wav')), desc="Extracting Features"):
        proxy_label = audio_path.parent.name
        mfcc, log_mel, log_mel_2d = extract_features(audio_path)

        if mfcc is not None:
            features_mfcc.append([audio_path.name, proxy_label] + list(mfcc))
            features_log_mel.append([audio_path.name, proxy_label] + list(log_mel))

            feature_filename = f"{audio_path.stem}.npy"
            save_path = CNN_FEATURES_DIR / feature_filename
            np.save(save_path, log_mel_2d)
            cnn_feature_metadata.append({'feature_path': str(save_path), 'label': proxy_label})

    # Save aggregated feature CSVs
    df_mfcc = pd.DataFrame(features_mfcc, columns=['filename', 'label'] + [f'mfcc_mean_{i}' for i in range(N_MFCC)] + [f'delta_mean_{i}' for i in range(N_MFCC)] + [f'mfcc_std_{i}' for i in range(N_MFCC)] + [f'delta_std_{i}' for i in range(N_MFCC)])
    df_mfcc.to_csv(FEATURES_DIR / "alertness_mfcc_delta_features.csv", index=False)

    df_log_mel = pd.DataFrame(features_log_mel, columns=['filename', 'label'] + [f'mel_mean_{i}' for i in range(N_MELS)] + [f'mel_std_{i}' for i in range(N_MELS)])
    df_log_mel.to_csv(FEATURES_DIR / "alertness_log_mel_features.csv", index=False)

    # Save CNN metadata
    df_cnn = pd.DataFrame(cnn_feature_metadata)
    df_cnn.to_csv(FEATURES_DIR / "cnn_metadata.csv", index=False)
    print(f"Saved CNN feature metadata to {FEATURES_DIR / 'cnn_metadata.csv'}")

    print("\n--- Feature Extraction Complete ---")


if __name__ == "__main__":
    main()