# 02a_explore_feature_distributions.py

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from gen_sound.config import FEATURES_DIR, PLOTS_DIR
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    FEATURES_DIR = BASE_DIR / "features"
    PLOTS_DIR = BASE_DIR / "plots_alertness"

def plot_feature_distributions(feature_name: str, df: pd.DataFrame, features_to_plot: list):
    sns.set_theme(style="whitegrid", palette="viridis")

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f'{feature_name}: Density Distribution of Selected Features by Class', fontsize=22, y=0.98)

    axes = axes.flatten()

    for i, col_name in enumerate(features_to_plot):
        ax = axes[i]
        sns.kdeplot(
            data=df,
            x=col_name,
            hue='label',
            ax=ax,
            fill=True,
            alpha=0.6,
            common_norm=False,
            linewidth=1.5
        )
        ax.set_title(f'Distribution of {col_name}', fontsize=16)
        ax.set_xlabel(col_name, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plot_path = PLOTS_DIR / f"dist_{feature_name.lower()}_kde.png"
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved KDE plot to {plot_path}")


def main():
    """Loads feature sets and orchestrates the creation of distribution plots."""
    print("--- Starting Feature Distribution Exploration ---")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_files = {
        "MFCC_Delta": FEATURES_DIR / "alertness_mfcc_delta_features.csv",
        "Log-Mel": FEATURES_DIR / "alertness_log_mel_features.csv"
    }

    features_to_showcase = {
        "MFCC_Delta": [
            'mfcc_mean_1',
            'mfcc_mean_2',
            'mfcc_mean_3',
            'delta_mean_4'
        ],
        "Log-Mel": [
            'mel_mean_2',
            'mel_mean_20',
            'mel_mean_100',
            'mel_std_20'
        ]
    }

    for feature_name, filepath in feature_files.items():
        if not filepath.exists():
            print(f"Warning: Feature file not found at {filepath}. Skipping.")
            continue

        df = pd.read_csv(filepath)

        features_for_this_set = features_to_showcase.get(feature_name)
        if features_for_this_set:
            plot_feature_distributions(feature_name, df, features_for_this_set)

    print("\n--- Feature Distribution Exploration Complete ---")


if __name__ == "__main__":
    main()