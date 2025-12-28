import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

from config import (
    BATCH_SIZE, NUM_EPOCHS, PATIENCE, LEARNING_RATE, 
    WEIGHT_DECAY, NUM_WORKERS, RANDOM_SEED, MODEL_DIR, OUTPUT_DIR
)
from dataset import ArtworkDataset, load_dataset, get_transforms
from model import ArtworkCNN, save_model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for imgs, lbls in pbar:
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(dim=1) == lbls).sum().item()
        total += imgs.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct/total)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Validating", leave=False):
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lbls.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_paths, labels, artist_names, _ = load_dataset()
    num_classes = len(artist_names)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        test_paths, test_labels, test_size=0.5, stratify=test_labels, random_state=RANDOM_SEED
    )

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    train_ds = ArtworkDataset(train_paths, train_labels, get_transforms(train=True))
    val_ds = ArtworkDataset(val_paths, val_labels, get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = ArtworkCNN(num_classes=num_classes, variant="b1").to(device)

    for p in model.backbone.classifier.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW([
        {"params": model.backbone.features.parameters(), "lr": LEARNING_RATE},
        {"params": model.backbone.classifier.parameters(), "lr": LEARNING_RATE * 3},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        if epoch == 10:
            print("Unfreezing top backbone layers")
            model.unfreeze_top_layers()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        scheduler.step(val_acc)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_model(model, MODEL_DIR / "best_model.pth", epoch, val_acc, artist_names)
            print(f"  -> Best model saved (Val Acc = {best_val_acc:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    save_model(model, MODEL_DIR / "last_model.pth", epoch, val_acc, artist_names)

    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(OUTPUT_DIR / "test_split.json", "w") as f:
        json.dump({"paths": test_paths, "labels": test_labels, "artist_names": artist_names}, f)

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.3f}")
    print(f"Model saved to {MODEL_DIR / 'best_model.pth'}")
    print(f"Run eval.py to evaluate on test set")


if __name__ == "__main__":
    main()
