"""
Train emotion classification model on FER2013 dataset.

Creates a ResNet18-based emotion classifier that can be used for emotion embedding extraction.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.config import MODELS_DIR, DATA_DIR, NUM_EMOTIONS
from src.utils.seed import set_seed


class EmotionDataset(Dataset):
    """
    Dataset wrapper for emotion classification.
    """
    
    def __init__(self, data_dir, transform=None):
        self.dataset = ImageFolder(str(data_dir), transform=transform)
        self.classes = self.dataset.classes
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class EmotionModel(nn.Module):
    """
    Emotion classification model based on ResNet18.
    """
    
    def __init__(self, num_classes: int = NUM_EMOTIONS, pretrained: bool = True):
        super(EmotionModel, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Replace final layer
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes)
        
        self.model = resnet
    
    def forward(self, x):
        return self.model(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100 * correct / total
        })
    
    return total_loss / len(dataloader), 100 * correct / total


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
    
    return total_loss / len(dataloader), 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train emotion classification model")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR / "faces" / "fer2013"),
                        help="Path to FER2013 dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default=str(MODELS_DIR),
                        help="Directory to save model")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "test"  # FER2013 uses 'test' for validation
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"Error: Dataset directories not found at {data_dir}")
        print("Please download FER2013 dataset first:")
        print("  python scripts/download_data.py --dataset fer2013")
        return
    
    train_dataset = EmotionDataset(train_dir, transform=train_transform)
    val_dataset = EmotionDataset(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create model
    model = EmotionModel(num_classes=len(train_dataset.classes), pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(args.output_dir) / "emotion_model.pth"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'classes': train_dataset.classes,
            }, output_path)
            print(f"✓ Saved best model to {output_path}")
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()

