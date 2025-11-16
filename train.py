import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np

DATA_ROOT = "binary_data"
MODEL_PATH = "models/cataract_resnet18_binary.pth"

def main():
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"Binary dataset not found at '{DATA_ROOT}'. Run extract_binary.py first.")

    # ============================
    # DATA TRANSFORMS
    # ============================
    img_size = 224

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=train_transform)
    print("Classes:", full_dataset.classes)  
    print("class_to_idx:", full_dataset.class_to_idx)

    # ============================
    # TRAIN / VAL SPLIT
    # ============================
    n_total = len(full_dataset)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # override val transform
    val_ds.dataset.transform = val_transform

    # ============================
    # CLASS IMBALANCE FIX
    # ============================
    targets = [full_dataset.imgs[i][1] for i in train_ds.indices]

    class_counts = np.bincount(targets)
    print("Class counts:", class_counts)

    # inverse frequency weight
    class_weights = 1.0 / class_counts
    print("Class weights:", class_weights)

    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # ============================
    # DATA LOADERS
    # ============================
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # ============================
    # MODEL
    # ============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)  
    model.to(device)

    # weighted loss (extra signal for cataract)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ============================
    # TRAINING LOOP
    # ============================
    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                outputs = model(images)
                loss = criterion(outputs, labels)

                if train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    NUM_EPOCHS = 12
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = run_epoch(train_loader, True)
        val_loss, val_acc = run_epoch(val_loader, False)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}"
        )

    # ============================
    # SAVE MODEL
    # ============================
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("💾 Model saved to:", os.path.abspath(MODEL_PATH))


if __name__ == "__main__":
    main()
