import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(script_dir, "splitData")
    IMG_SIZE = 224
    BATCH_SIZE = 8
    EPOCHS = 5
    NUM_WORKERS = 0  # safe for Mac

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, "val"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print dataset information
    print(f"Dataset loaded from: {DATA_PATH}")
    print(f"Number of categories: {len(train_dataset.classes)}")
    print(f"Categories: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Using device: {device}\n")
    model = mobilenet_v3_small(weights="IMAGENET1K_V1")

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    num_classes = len(train_dataset.classes)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = 100 * correct / total if total > 0 else 0.0
        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%\n")

    # Create model directory if it doesn't exist
    model_dir = os.path.join(script_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "mobilenetv3_recyclable_classifier.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}.")
    print(f"Model trained on {num_classes} categories: {train_dataset.classes}")

if __name__ == "__main__":
    main()