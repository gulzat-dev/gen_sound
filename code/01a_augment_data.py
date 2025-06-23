# 01a_augment_data.py
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
import shutil
import sys
from collections import Counter

try:
    from config import BASE_SEGMENTS_DIR, FINAL_AUDIO_DIR, SAMPLE_RATE
except ImportError:
    print("Error: config.py not found.")
    sys.exit(1)


# --- Define Augmentation Functions using Librosa ---

def add_noise(y, noise_factor=0.005):
    """Adds random Gaussian noise to the audio signal."""
    noise = np.random.randn(len(y))
    augmented_y = y + noise_factor * noise
    # Ensure the audio is still in the valid range [-1, 1]
    augmented_y = np.clip(augmented_y, -1., 1.)
    return augmented_y


def time_stretch(y, rate=1.0):
    """Stretches the time of an audio signal without changing pitch."""
    return librosa.effects.time_stretch(y=y, rate=rate)


def pitch_shift(y, sr, n_steps):
    """Shifts the pitch of an audio signal."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


# --- Configuration for Augmentation ---
# Define how many new augmented samples to create for each original sample
# in the specified minority classes.
AUGMENTATION_FACTORS = {
    "proxy_distress_discomfort": 5,  # Create 5 new versions per original
    "proxy_joy_excitement": 2,  # Create 2 new versions per original
}


def main():
    """Augments the base dataset using Librosa and creates the final dataset."""
    print("--- Starting Separate Audio Augmentation Step (using Librosa) ---")

    if not BASE_SEGMENTS_DIR.is_dir():
        print(f"Error: Base segments directory not found at {BASE_SEGMENTS_DIR}")
        print("Please run 01_alertness_proxy_data_prep.py first.")
        sys.exit(1)

    if FINAL_AUDIO_DIR.exists():
        print(f"Cleaning previous final data in {FINAL_AUDIO_DIR}")
        shutil.rmtree(FINAL_AUDIO_DIR)
    FINAL_AUDIO_DIR.mkdir(parents=True)

    source_files = list(BASE_SEGMENTS_DIR.glob('*/*.wav'))
    counts = Counter()

    for original_path in tqdm(source_files, desc="Augmenting Data with Librosa"):
        class_label = original_path.parent.name

        dest_class_dir = FINAL_AUDIO_DIR / class_label
        dest_class_dir.mkdir(exist_ok=True)

        shutil.copy(original_path, dest_class_dir)
        counts[class_label] += 1

        if class_label in AUGMENTATION_FACTORS:
            num_augmentations = AUGMENTATION_FACTORS[class_label]
            try:
                y, sr = librosa.load(original_path, sr=SAMPLE_RATE)

                for i in range(num_augmentations):
                    y_aug = y.copy()  # Start with a fresh copy

                    # Apply a random combination of augmentations
                    choice = np.random.randint(0, 3)
                    if choice == 0:
                        # Add noise
                        y_aug = add_noise(y_aug, noise_factor=np.random.uniform(0.003, 0.008))
                    elif choice == 1:
                        # Time stretch
                        stretch_rate = np.random.uniform(0.8, 1.2)
                        y_aug = time_stretch(y_aug, rate=stretch_rate)
                    elif choice == 2:
                        # Pitch shift
                        shift_steps = np.random.uniform(-4, 4)
                        y_aug = pitch_shift(y_aug, sr=sr, n_steps=shift_steps)

                    # The length might change due to time stretching, so we fix it
                    y_aug = librosa.util.fix_length(y_aug, size=len(y))

                    aug_filename = f"aug{i}_{original_path.name}"
                    sf.write(dest_class_dir / aug_filename, y_aug, sr)
                    counts[f"{class_label}_augmented"] += 1
            except Exception as e:
                print(f"Could not augment {original_path}: {e}")

    print("\n--- Audio Augmentation Complete ---")
    print("Summary of final dataset composition:")
    final_counts = Counter()
    for category, count in sorted(counts.items()):
        base_category = category.replace("_augmented", "")
        final_counts[base_category] += count

    for category, count in sorted(final_counts.items()):
        print(f"- {category}: {count} segments")
    print(f"\nTotal final segments created: {sum(final_counts.values())}")


if __name__ == "__main__":
    main()