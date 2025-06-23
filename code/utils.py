# utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import shutil
from pathlib import Path
from datetime import datetime

try:
    from config import RESULTS_FILE
except ImportError:
    RESULTS_FILE = Path("project_summary_results.txt")


def initialize_results_file():
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()

    header = f"Project Results Summary\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 70}\n"
    RESULTS_FILE.write_text(header)
    print(f"--- Initialized clean results file: {RESULTS_FILE.name} ---")


def log_and_save_df(title: str, df: pd.DataFrame, include_index: bool = True):
    """Prints a formatted DataFrame to the console AND appends it to the results file."""
    try:
        table_str = df.to_markdown(index=include_index, floatfmt=".4f")
    except (ImportError, AttributeError):
        table_str = df.to_string(float_format="%.4f")
    output_block = f"\n--- {title} ---\n{table_str}\n"
    print(output_block)
    try:
        with open(RESULTS_FILE, 'a') as f:
            f.write(output_block)
            f.write(f"\n{'=' * 70}\n")
        print(f"--- Results for '{title}' have been logged and saved. ---")
    except Exception as e:
        print(f"Error saving results to file: {e}")


def clean_and_recreate_dirs(dir_list: list[Path]):
    for dir_path in dir_list:
        if dir_path.exists():
            print(f"--- CLEANUP: Removing previous output directory: {dir_path} ---")
            shutil.rmtree(dir_path)
    for dir_path in dir_list:
        print(f"--- SETUP: Creating empty output directory: {dir_path} ---")
        dir_path.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    integer_labels = range(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=integer_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")