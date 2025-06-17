#!/usr/bin/env python

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights


def get_args():
    parser = argparse.ArgumentParser(description="Train ResNet-50 for real vs AI image classification")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root folder with 'train/real', 'train/ai', 'val/real', 'val/ai'")
    parser.add_argument("--epochs",    type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size",type=int, default=32, help="Batch size")
    parser.add_argument("--lr",        type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--wd",        type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--out-dir",   type=str, default="checkpoints", help="Where to save models")
    parser.add_argument("--num-workers",type=int,default=4, help="DataLoader workers")
    return parser.parse_args()


def build_dataloaders(root, batch_size, num_workers):
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = ImageFolder(os.path.join(root, "train"), transform=train_tf)
    val_ds   = ImageFolder(os.path.join(root,   "test"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_model(device):
    # to load pretrained weights (if you’ve already cached them locally):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for imgs, labels in tqdm(loader, desc="  Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    model    = build_model(device)
    criterion= nn.CrossEntropyLoss()
    optimizer= optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler= optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model,   val_loader,   criterion, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.out_dir, f"best_resnet50_epoch{epoch}_acc{val_acc:.4f}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved best model to: {ckpt_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
