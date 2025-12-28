import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, top_k_accuracy_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path

from config import BATCH_SIZE, NUM_WORKERS, MODEL_DIR, OUTPUT_DIR, PLOTS_DIR
from dataset import ArtworkDataset, get_transforms
from model import load_model

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.cm.tab20.colors


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(epochs, history["train_loss"], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, history["train_acc"], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history["val_acc"], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    gap = np.array(history["train_acc"]) - np.array(history["val_acc"])
    axes[1, 0].bar(epochs, gap, color=['green' if g < 0.1 else 'orange' if g < 0.2 else 'red' for g in gap])
    axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', label='Mild overfit')
    axes[1, 0].axhline(y=0.2, color='red', linestyle='--', label='Severe overfit')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train - Val Accuracy')
    axes[1, 0].set_title('Overfitting Gap')
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history["lr"], 'g-', linewidth=2, marker='o', markersize=3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path, normalize=True):
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    n_classes = len(class_names)
    fig_size = max(12, n_classes * 0.5)
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(cm_display, annot=n_classes <= 20, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_metrics(report_df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    metrics = ['precision', 'recall', 'f1-score']
    colors = ['steelblue', 'darkorange', 'forestgreen']
    
    for ax, metric, color in zip(axes, metrics, colors):
        sorted_df = report_df.sort_values(metric, ascending=True)
        y_pos = range(len(sorted_df))
        ax.barh(y_pos, sorted_df[metric], color=color, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_df.index, fontsize=7)
        ax.set_xlabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} by Class')
        ax.axvline(x=sorted_df[metric].mean(), color='red', linestyle='--', 
                   label=f'Mean: {sorted_df[metric].mean():.2f}')
        ax.legend()
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_top_k_accuracy(targets, probs, class_names, save_path):
    max_k = min(10, len(class_names))
    k_values = range(1, max_k + 1)
    accuracies = [top_k_accuracy_score(targets, probs, k=k) for k in k_values]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(k_values, accuracies, color='steelblue', alpha=0.8)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('K (Top-K)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Top-K Accuracy', fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(k_values)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_prediction_confidence(probs, preds, targets, save_path):
    max_probs = np.max(probs, axis=1)
    correct = np.array(preds) == np.array(targets)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(max_probs[correct], bins=30, alpha=0.7, label='Correct', color='green', density=True)
    axes[0].hist(max_probs[~correct], bins=30, alpha=0.7, label='Incorrect', color='red', density=True)
    axes[0].set_xlabel('Confidence (Max Probability)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Prediction Confidence Distribution')
    axes[0].legend()
    
    thresholds = np.linspace(0, 1, 50)
    accuracies = []
    coverages = []
    for t in thresholds:
        mask = max_probs >= t
        if mask.sum() > 0:
            accuracies.append(correct[mask].mean())
            coverages.append(mask.mean())
        else:
            accuracies.append(np.nan)
            coverages.append(0)
    
    ax2 = axes[1].twinx()
    axes[1].plot(thresholds, accuracies, 'b-', linewidth=2, label='Accuracy')
    ax2.plot(thresholds, coverages, 'r--', linewidth=2, label='Coverage')
    axes[1].set_xlabel('Confidence Threshold')
    axes[1].set_ylabel('Accuracy', color='blue')
    ax2.set_ylabel('Coverage', color='red')
    axes[1].set_title('Accuracy vs Coverage at Different Thresholds')
    axes[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_misclassified_analysis(cm, class_names, save_path, top_n=15):
    mistakes = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                mistakes.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i, j]
                })
    
    mistakes_df = pd.DataFrame(mistakes).sort_values('count', ascending=False).head(top_n)
    
    if len(mistakes_df) == 0:
        return
    
    plt.figure(figsize=(12, 6))
    labels = [f"{row['true']} → {row['pred']}" for _, row in mistakes_df.iterrows()]
    plt.barh(range(len(labels)), mistakes_df['count'], color='coral')
    plt.yticks(range(len(labels)), labels, fontsize=9)
    plt.xlabel('Count')
    plt.title(f'Top {top_n} Most Common Misclassifications')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def evaluate_model(model, loader, device, class_names):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_targets.extend(lbls.numpy())

    return np.array(all_probs), np.array(all_preds), np.array(all_targets)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        print(f"Model not found at {model_path}. Train the model first.")
        return

    model, artist_names = load_model(model_path, device)
    print(f"Loaded model with {len(artist_names)} classes")

    test_split_path = OUTPUT_DIR / "test_split.json"
    if test_split_path.exists():
        with open(test_split_path) as f:
            data = json.load(f)
        test_paths, test_labels = data["paths"], data["labels"]
    else:
        print("Test split not found. Creating new test split from dataset...")
        from dataset import load_dataset
        from sklearn.model_selection import train_test_split
        from config import RANDOM_SEED
        
        image_paths, labels, _, _ = load_dataset()
        _, test_paths, _, test_labels = train_test_split(
            image_paths, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
        )
        _, test_paths, _, test_labels = train_test_split(
            test_paths, test_labels, test_size=0.5, stratify=test_labels, random_state=RANDOM_SEED
        )
        print(f"Using {len(test_paths)} test images")

    test_ds = ArtworkDataset(test_paths, test_labels, get_transforms(train=False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    probs, preds, targets = evaluate_model(model, test_loader, device, artist_names)

    accuracy = (preds == targets).mean()
    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Top-3 Accuracy: {top_k_accuracy_score(targets, probs, k=3):.4f}")
    print(f"Top-5 Accuracy: {top_k_accuracy_score(targets, probs, k=5):.4f}")

    try:
        roc_auc = roc_auc_score(targets, probs, multi_class="ovr", average="weighted")
        print(f"ROC-AUC (weighted): {roc_auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC computation skipped: {e}")

    report = classification_report(targets, preds, target_names=artist_names, output_dict=True)
    report_df = pd.DataFrame(report).T
    class_report_df = report_df.iloc[:-3]  # Exclude avg rows

    print(f"\n{'='*50}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*50}")
    print(classification_report(targets, preds, target_names=artist_names))

    report_df.to_csv(PLOTS_DIR / "classification_report.csv")
    print(f"Saved: {PLOTS_DIR / 'classification_report.csv'}")

    history_path = OUTPUT_DIR / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, PLOTS_DIR / "training_curves.png")

    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, artist_names, PLOTS_DIR / "confusion_matrix.png", normalize=True)
    plot_confusion_matrix(cm, artist_names, PLOTS_DIR / "confusion_matrix_counts.png", normalize=False)

    plot_per_class_metrics(class_report_df, PLOTS_DIR / "per_class_metrics.png")
    plot_top_k_accuracy(targets, probs, artist_names, PLOTS_DIR / "top_k_accuracy.png")
    plot_prediction_confidence(probs, preds, targets, PLOTS_DIR / "confidence_analysis.png")
    plot_misclassified_analysis(cm, artist_names, PLOTS_DIR / "misclassifications.png")

    print(f"\n{'='*50}")
    print(f"All plots saved to: {PLOTS_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
