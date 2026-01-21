#!/usr/bin/env python3
"""
Train a spectrogram classifier (AI vs Human) using PyTorch + torchvision.

  python3 train_spectrogram_classifier.py \
      --data-root ./spectrograms \
      --model resnet18 \
      --epochs 15 \
      --batch-size 32 \
      --lr 1e-4 \
      --img-size 224 \
      --output-dir ./runs/exp1
      
"""

import argparse
import os
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms, models
import numpy as np


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)


def build_transforms(img_size: int):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), inplace=False),
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, eval_tf

def get_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Model '{name}' not supported.")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def get_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True)


def precision_recall_f1(preds: np.ndarray, targets: np.ndarray, positive_label: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Binary metrics; if positive_label is None, we infer the '1' class if available.
    """
    if positive_label is None:
        positive_label = 1 if 1 in np.unique(targets) else 0

    tp = np.sum((preds == positive_label) & (targets == positive_label))
    fp = np.sum((preds == positive_label) & (targets != positive_label))
    fn = np.sum((preds != positive_label) & (targets == positive_label))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    acc = np.mean(all_preds_np == all_targets_np)
    prec, rec, f1 = precision_recall_f1(all_preds_np, all_targets_np)
    return {"loss": avg_loss, "acc": acc, "precision": prec, "recall": rec, "f1": f1}


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def main():
    parser = argparse.ArgumentParser(description="Train spectrogram classifier (AI vs Human).")
    parser.add_argument("--data-root", type=str, required=True, help="Root folder containing train/ (and val/, test/ optional)")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4, help="Num workers for DataLoader")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-sampler", action="store_true", help="Use weighted random sampler for class imbalance")
    parser.add_argument("--output-dir", type=str, default="runs/exp", help="Directory to save model and logs")

    args = parser.parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train_log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    train_tf, eval_tf = build_transforms(args.img_size)

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")
    test_dir = os.path.join(args.data_root, "test")

    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    class_to_idx = full_train.class_to_idx
    print(f"[Info] Classes: {class_to_idx}")

    if os.path.isdir(val_dir):
        val_set = datasets.ImageFolder(val_dir, transform=eval_tf)
    else:
        print("[Info] No `val` directory found, splitting 20% from train set for validation.")
        val_len = int(len(full_train) * 0.2)
        train_len = len(full_train) - val_len
        full_train, val_set = random_split(full_train, [train_len, val_len],
                                           generator=torch.Generator().manual_seed(args.seed))

    test_set = datasets.ImageFolder(test_dir, transform=eval_tf) if os.path.isdir(test_dir) else None

    if args.balanced_sampler and isinstance(full_train, datasets.ImageFolder):
        sampler = get_weighted_sampler(full_train)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = (DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
                   if test_set is not None else None)

    num_classes = len(class_to_idx)
    model = get_model(args.model, num_classes)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    with open(log_file, "w") as f:
        header = "Epoch,Train_Loss,Val_Loss,Val_Acc,Val_F1,Val_Precision,Val_Recall\n"
        f.write(header)
        print(header.strip())

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate_model(model, val_loader, criterion, device)
            val_loss = val_metrics["loss"]
            
            log_line = (f"{epoch+1},{train_loss:.4f},{val_loss:.4f},"
                        f"{val_metrics['acc']:.4f},{val_metrics['f1']:.4f},"
                        f"{val_metrics['precision']:.4f},{val_metrics['recall']:.4f}")
            print(log_line)
            f.write(log_line + "\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
                print(f"  -> New best model saved with val_loss: {val_loss:.4f}")

    if test_loader:
        print("\n[Info] Running final evaluation on test set...")
        best_model = get_model(args.model, num_classes)
        best_model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
        best_model.to(device)

        test_metrics = evaluate_model(best_model, test_loader, criterion, device)
        test_log_line = (f"Test Results: Loss={test_metrics['loss']:.4f}, "
                         f"Acc={test_metrics['acc']:.4f}, F1={test_metrics['f1']:.4f}, "
                         f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}")
        print(test_log_line)
        with open(log_file, "a") as f:
            f.write(test_log_line + "\n")

    print("[Info] Training complete.")


if __name__ == "__main__":
    main()
