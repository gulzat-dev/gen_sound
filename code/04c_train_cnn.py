# 04c_train_cnn_pytorch_v2.py
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- Import from the central config file ---
try:
    from gen_sound.config import (
        FEATURES_DIR,
        MODELS_DIR,
        PLOTS_DIR,
        RANDOM_STATE,
        TEST_SIZE,
        BATCH_SIZE,
        NUM_EPOCHS,
        LEARNING_RATE
    )
except ImportError:
    print("Error: config.py not found. Please ensure it's in the project root or accessible.")
    sys.exit(1)

# --- Import from the utils file ---
try:
    from utils import plot_confusion_matrix, log_and_save_df
except ImportError:
    print("Warning: utils.py not found. Using dummy functions for logging and plotting.")


    def plot_confusion_matrix(y_true, y_pred, classes, title, path):
        print(f"DUMMY: Would plot confusion matrix '{title}' to {path}")


    def log_and_save_df(title, df, **kwargs):
        print(f"DUMMY: --- {title} ---")
        print(df.to_string())


class AudioCNN(nn.Module):
    """A simple 2D CNN for audio classification."""

    def __init__(self, num_classes, input_shape):
        super(AudioCNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding='same'),
            nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(kernel_size=(2, 2)),
        )
        with torch.no_grad():
            dummy_output = self.conv_stack(torch.zeros(1, *input_shape))
            flattened_size = dummy_output.flatten(1).shape[1]
        self.flatten = nn.Flatten()
        self.dense_stack = nn.Sequential(
            nn.Linear(in_features=flattened_size, out_features=256),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        return self.dense_stack(self.flatten(self.conv_stack(x)))


def train_model(model, loader, criterion, optimizer, device):
    """Trains the model for one epoch and returns loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def validate_epoch(model, loader, criterion, device):
    """Evaluates the model on a validation/test set for one epoch."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def evaluate_model(model, loader, device):
    """Evaluates the model on a given dataset for final reporting."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return np.array(labels), np.array(preds)


def main():
    """Main function to run the training and evaluation pipeline."""
    print("--- Starting CNN Model Training (PyTorch) ---")

    # This file name is specific to the CNN feature extraction script
    metadata_path = FEATURES_DIR / "cnn_metadata.csv"

    try:
        metadata = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        print("Please ensure you have run the feature extraction script first.")
        sys.exit(1)

    # The feature extraction script should save paths relative to the FEATURES_DIR
    X = np.array([np.load(FEATURES_DIR / p) for p in tqdm(metadata['feature_path'], desc="Loading Spectrograms")])[:,
        np.newaxis, :, :]

    le = LabelEncoder()
    y = le.fit_transform(metadata['label'].values)
    class_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    input_shape = (1, X_train.shape[2], X_train.shape[3])
    model = AudioCNN(num_classes=len(class_names), input_shape=input_shape).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\n--- Starting Training Loop ---")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Ensure directories from config exist
    MODELS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    torch.save(model.state_dict(), MODELS_DIR / "cnn_pytorch_model.pth")
    joblib.dump(le, MODELS_DIR / "cnn_pytorch_label_encoder.joblib")
    print(f"\nModel and label encoder saved to {MODELS_DIR}")

    # Plotting training history
    print("\n--- Generating Training History Plots ---")
    epochs_range = range(1, NUM_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs_range, history['train_loss'], label='Training Loss', marker='o')
    ax1.plot(epochs_range, history['val_loss'], label='Validation Loss', marker='o')
    ax1.set_title('CNN Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, history['train_acc'], label='Training Accuracy', marker='o')
    ax2.plot(epochs_range, history['val_acc'], label='Validation Accuracy', marker='o')
    ax2.set_title('CNN Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "cnn_training_history.png"
    plt.savefig(plot_path)
    print(f"Training history plot saved to: {plot_path}")

    # Final Evaluation
    print("\n--- CNN (PyTorch) Final Test Set Evaluation ---")
    y_true, y_pred = evaluate_model(model, test_loader, device)

    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report_str)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # <-- MODIFIED: Added precision and recall from the report dictionary
    results_df = pd.DataFrame([{
        'Model': 'CNN (PyTorch)',
        'Test Acc': report_dict['accuracy'],
        'Test F1-W': report_dict['weighted avg']['f1-score'],
        'Test Precision-W': report_dict['weighted avg']['precision'],
        'Test Recall-W': report_dict['weighted avg']['recall']
    }])

    log_and_save_df("CNN (PyTorch) Model Performance", results_df, include_index=False)

    plot_confusion_matrix(y_true, y_pred, class_names, "CNN (PyTorch) Confusion Matrix",
                          PLOTS_DIR / "cm_cnn_pytorch.png")

    log_file_path = PLOTS_DIR / "report_cnn_pytorch.txt"
    with open(log_file_path, 'w') as f:
        f.write("--- CNN (PyTorch) Test Set Evaluation ---\n\n")
        f.write(report_str)
        f.write("\n\n--- CNN (PyTorch) Model Performance ---\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n")
    print(f"Detailed evaluation report saved to: {log_file_path}")

    print("\n--- CNN (PyTorch) Model Training Complete ---")


if __name__ == "__main__":
    main()