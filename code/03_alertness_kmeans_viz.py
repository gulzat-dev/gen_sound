# 03_alertness_kmeans_viz.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import sys

try:
    from config import FEATURES_DIR, PLOTS_DIR
except ImportError:
    print("Error: config.py not found.")
    sys.exit(1)


def run_kmeans_analysis(df, feature_name):
    """Performs K-means clustering, dimensionality reduction, and visualization."""
    print(f"\n--- Analyzing {feature_name} ---")

    # Prepare data
    X = df.drop(columns=['filename', 'label']).values
    y_true = df['label'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means clustering
    n_clusters = len(np.unique(y_true))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred_kmeans = kmeans.fit_predict(X_scaled)

    # Evaluate clustering
    ari = adjusted_rand_score(y_true, y_pred_kmeans)
    nmi = normalized_mutual_info_score(y_true, y_pred_kmeans)
    silhouette = silhouette_score(X_scaled, y_pred_kmeans)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")

    # Dimensionality Reduction
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print("Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 2))
    X_tsne = tsne.fit_transform(X_scaled)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'K-means Clustering & Dimensionality Reduction for {feature_name}', fontsize=18)

    # PCA vs True Labels
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_true, palette='viridis', ax=axes[0, 0], s=20, alpha=0.7)
    axes[0, 0].set_title('PCA Projection colored by True Labels')
    axes[0, 0].legend(title='True Label')

    # PCA vs K-means Labels
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pred_kmeans, palette='viridis', ax=axes[0, 1], s=20, alpha=0.7)
    axes[0, 1].set_title('PCA Projection colored by K-means Clusters')
    axes[0, 1].legend(title='K-means Cluster')

    # t-SNE vs True Labels
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_true, palette='viridis', ax=axes[1, 0], s=20, alpha=0.7)
    axes[1, 0].set_title('t-SNE Projection colored by True Labels')
    axes[1, 0].legend(title='True Label')

    # t-SNE vs K-means Labels
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_pred_kmeans, palette='viridis', ax=axes[1, 1], s=20,
                    alpha=0.7)
    axes[1, 1].set_title('t-SNE Projection colored by K-means Clusters')
    axes[1, 1].legend(title='K-means Cluster')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = PLOTS_DIR / f"kmeans_viz_{feature_name.lower()}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved visualization to {plot_path}")

    return {'Feature Set': feature_name, 'ARI': ari, 'NMI': nmi, 'Silhouette Score': silhouette}


def main():
    print("--- Starting K-means Clustering and Visualization ---")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_files = {
        "MFCC_Delta": "alertness_mfcc_delta_features.csv",
        "Log-Mel": "alertness_log_mel_features.csv",
    }

    all_metrics = []
    for feature_name, filename in feature_files.items():
        filepath = FEATURES_DIR / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found. Skipping.")
            continue

        df = pd.read_csv(filepath)
        metrics = run_kmeans_analysis(df, feature_name)
        all_metrics.append(metrics)

    # Print summary table
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics).set_index('Feature Set')
        print("\n--- K-means Clustering Performance Summary ---")
        print(summary_df.to_string(float_format="%.4f"))

    print("\n--- K-means Analysis Complete ---")


if __name__ == "__main__":
    main()