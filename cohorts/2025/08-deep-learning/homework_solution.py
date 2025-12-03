"""
Deep Learning Homework - Hair Type Classification
Machine Learning Zoomcamp 2025 - Module 8

This script implements a CNN for binary classification of hair types (straight vs curly)
using PyTorch, following the homework requirements exactly.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import wget
import zipfile
import statistics

# ================================
# REPRODUCIBILITY SETUP
# ================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ================================
# DATASET DOWNLOAD AND PREPARATION
# ================================
def download_dataset():
    """Download and extract the hair type dataset"""
    url = "https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip"
    
    if not os.path.exists("data.zip"):
        print("Downloading dataset...")
        wget.download(url, "data.zip")
        print("\nDownload completed!")
    
    if not os.path.exists("data"):
        print("Extracting dataset...")
        with zipfile.ZipFile("data.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction completed!")
    else:
        print("Dataset already exists!")

# ================================
# MODEL DEFINITION
# ================================
class HairCNN(nn.Module):
    """
    CNN for hair type classification following homework specifications:
    - Input: (3, 200, 200)
    - Conv2d: 32 filters, kernel_size=(3,3), padding=0, stride=1, ReLU
    - MaxPool2d: pool_size=(2,2)
    - Flatten
    - Linear: 64 neurons, ReLU
    - Linear: 1 neuron, Sigmoid (for binary classification)
    """
    
    def __init__(self):
        super(HairCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, 
                              kernel_size=3, padding=0, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after conv and pooling
        # Input: (3, 200, 200)
        # After conv1: (32, 198, 198)  # 200-3+1 = 198
        # After pool1: (32, 99, 99)    # 198/2 = 99
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 99 * 99, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        # Note: For BCEWithLogitsLoss, we don't apply sigmoid here
        return x

# ================================
# DATA TRANSFORMATIONS
# ================================
# Initial transforms (without augmentation)
train_transforms_initial = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )  # ImageNet normalization
])

# Transforms with data augmentation
train_transforms_augmented = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Test transforms (no augmentation)
test_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================================
# TRAINING FUNCTION
# ================================
def train_model(model, train_loader, validation_loader, criterion, optimizer, 
                num_epochs, device, train_dataset, validation_dataset):
    """Train the model and return training history"""
    
    history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)  # Ensure labels are float and have shape (batch_size, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            # For binary classification with BCEWithLogitsLoss, apply sigmoid to outputs before thresholding for accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_train / total_train
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(validation_dataset)
        val_epoch_acc = correct_val / total_val
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
    
    return history

# ================================
# MAIN EXECUTION
# ================================
def main():
    # Download dataset
    download_dataset()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ===============================
    # QUESTION 1: LOSS FUNCTION
    # ===============================
    print("\n" + "="*50)
    print("QUESTION 1: Which loss function to use?")
    print("="*50)
    print("For binary classification, we should use:")
    print("✓ nn.BCEWithLogitsLoss() - Binary Cross Entropy with Logits")
    print("This is appropriate for binary classification tasks.")
    
    # ===============================
    # QUESTION 2: MODEL PARAMETERS
    # ===============================
    print("\n" + "="*50)
    print("QUESTION 2: Total number of parameters")
    print("="*50)
    
    # Create model
    model = HairCNN().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # Try to use torchsummary if available
    try:
        summary(model, input_size=(3, 200, 200))
    except ImportError:
        print("torchsummary not available, but total parameters calculated manually")
    
    # Manual calculation verification
    # Conv1: (3 * 3 * 3 + 1) * 32 = 28 * 32 = 896
    # FC1: (32 * 99 * 99 + 1) * 64 = 312769 * 64 = 20017216
    # FC2: (64 + 1) * 1 = 65
    # Total = 896 + 20017216 + 65 = 20018177
    
    print("\nManual calculation:")
    conv1_params = (3 * 3 * 3 + 1) * 32
    fc1_params = (32 * 99 * 99 + 1) * 64
    fc2_params = (64 + 1) * 1
    print(f"Conv1 parameters: {conv1_params}")
    print(f"FC1 parameters: {fc1_params}")
    print(f"FC2 parameters: {fc2_params}")
    print(f"Total calculated: {conv1_params + fc1_params + fc2_params}")
    
    # ===============================
    # INITIAL TRAINING SETUP
    # ===============================
    print("\n" + "="*50)
    print("INITIAL TRAINING (WITHOUT AUGMENTATION)")
    print("="*50)
    
    # Load datasets without augmentation
    train_dataset = datasets.ImageFolder('data/train', transform=train_transforms_initial)
    validation_dataset = datasets.ImageFolder('data/test', transform=test_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=20, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)
    
    # Train initial model
    print("\nTraining initial model for 10 epochs...")
    history_initial = train_model(model, train_loader, validation_loader, criterion, 
                                optimizer, num_epochs=10, device=device,
                                train_dataset=train_dataset, validation_dataset=validation_dataset)
    
    # ===============================
    # QUESTION 3: MEDIAN TRAINING ACCURACY
    # ===============================
    print("\n" + "="*50)
    print("QUESTION 3: Median of training accuracy")
    print("="*50)
    
    median_acc = statistics.median(history_initial['acc'])
    print(f"Training accuracies: {[f'{acc:.4f}' for acc in history_initial['acc']]}")
    print(f"Median training accuracy: {median_acc:.4f}")
    
    # ===============================
    # QUESTION 4: STANDARD DEVIATION OF TRAINING LOSS
    # ===============================
    print("\n" + "="*50)
    print("QUESTION 4: Standard deviation of training loss")
    print("="*50)
    
    std_loss = statistics.stdev(history_initial['loss'])
    print(f"Training losses: {[f'{loss:.4f}' for loss in history_initial['loss']]}")
    print(f"Standard deviation of training loss: {std_loss:.4f}")
    
    # ===============================
    # DATA AUGMENTATION TRAINING
    # ===============================
    print("\n" + "="*50)
    print("TRAINING WITH DATA AUGMENTATION")
    print("="*50)
    
    # Create new dataset with augmentation
    train_dataset_aug = datasets.ImageFolder('data/train', transform=train_transforms_augmented)
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=20, shuffle=True)
    
    # Continue training with augmentation (don't recreate model)
    print("Continuing training with data augmentation for 10 more epochs...")
    history_aug = train_model(model, train_loader_aug, validation_loader, criterion, 
                            optimizer, num_epochs=10, device=device,
                            train_dataset=train_dataset_aug, validation_dataset=validation_dataset)
    
    # ===============================
    # QUESTION 5: MEAN TEST LOSS WITH AUGMENTATION
    # ===============================
    print("\n" + "="*50)
    print("QUESTION 5: Mean test loss with augmentations")
    print("="*50)
    
    mean_val_loss = statistics.mean(history_aug['val_loss'])
    print(f"Validation losses with augmentation: {[f'{loss:.4f}' for loss in history_aug['val_loss']]}")
    print(f"Mean validation loss: {mean_val_loss:.4f}")
    
    # ===============================
    # QUESTION 6: AVERAGE TEST ACCURACY FOR LAST 5 EPOCHS
    # ===============================
    print("\n" + "="*50)
    print("QUESTION 6: Average test accuracy for last 5 epochs (6-10)")
    print("="*50)
    
    last_5_acc = history_aug['val_acc'][5:]  # epochs 6-10 (0-indexed: 5-9)
    avg_last_5_acc = statistics.mean(last_5_acc)
    print(f"Validation accuracies for epochs 6-10: {[f'{acc:.4f}' for acc in last_5_acc]}")
    print(f"Average validation accuracy for last 5 epochs: {avg_last_5_acc:.4f}")
    
    # ===============================
    # SUMMARY OF ANSWERS
    # ===============================
    print("\n" + "="*70)
    print("HOMEWORK ANSWERS SUMMARY")
    print("="*70)
    print(f"Question 1: nn.BCEWithLogitsLoss()")
    print(f"Question 2: {total_params}")
    print(f"Question 3: {median_acc:.4f}")
    print(f"Question 4: {std_loss:.4f}")
    print(f"Question 5: {mean_val_loss:.4f}")
    print(f"Question 6: {avg_last_5_acc:.4f}")
    print("="*70)
    
    return {
        'q1': 'nn.BCEWithLogitsLoss()',
        'q2': total_params,
        'q3': median_acc,
        'q4': std_loss,
        'q5': mean_val_loss,
        'q6': avg_last_5_acc
    }

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import torchsummary
    except ImportError:
        print("Installing torchsummary...")
        os.system("pip install torchsummary")
    
    try:
        import wget
    except ImportError:
        print("Installing wget...")
        os.system("pip install wget")
    
    answers = main()