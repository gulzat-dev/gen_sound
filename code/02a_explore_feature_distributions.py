# 02a_explore_feature_distributions.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

try:
    from config import FEATURES_DIR, PLOTS_DIR
except ImportError:
    print("Error: config.py not found.")
    sys.exit(1)


def main():
    """Loads feature sets and creates distribution plots."""
    print("--- Starting Feature Distribution Exploration ---")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_files = {
        "MFCC_Delta": "alertness_mfcc_delta_features.csv",
        "Log-Mel": "alertness_log_mel_features.csv"
    }

    # Define a small subset of features to plot for clarity
    features_to_plot = {
        "MFCC_Delta": ['mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'delta_mean_1'],
        "Log-Mel": ['mel_mean_10', 'mel_mean_20', 'mel_mean_30', 'mel_mean_40']
    }

    for feature_name, filename in feature_files.items():
        filepath = FEATURES_DIR / filename
        if not filepath.exists():
            print(f"Warning: Feature file not found at {filepath}. Skipping.")
            continue

        print(f"\nProcessing {feature_name} features...")
        df = pd.read_csv(filepath)

        # --- KDE (Density) Plots ---
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'{feature_name}: Density Distribution of Selected Features by Class', fontsize=16)
        for i, col in enumerate(features_to_plot[feature_name]):
            plt.subplot(2, 2, i + 1)
            sns.kdeplot(data=df, x=col, hue='label', fill=True, common_norm=False, palette='viridis')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Density')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = PLOTS_DIR / f"dist_{feature_name.lower()}_kde.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved KDE plot to {plot_path}")



if __name__ == "__main__":
    main()