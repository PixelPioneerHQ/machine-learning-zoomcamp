# 🎯 Deep Learning Homework - Step-by-Step Instructions

## Overview
Build a CNN to classify hair types (straight vs curly) using PyTorch. This homework involves training a model from scratch, analyzing its performance, and applying data augmentation.

---

## 📋 Prerequisites

### 1. Environment Setup
```bash
# Install required packages
pip install torch torchvision torchsummary wget matplotlib numpy
```

### 2. Check PyTorch Version
```python
import torch
print(f"PyTorch version: {torch.__version__}")
# Should be 2.8.0 for Colab compatibility
```

---

## 🔧 Step 1: Download Dataset

```bash
# Download and extract dataset
wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
unzip data.zip
```

**Dataset Structure:**
```
data/
├── train/
│   ├── curly/     # Curly hair images
│   └── straight/  # Straight hair images
└── test/
    ├── curly/     # Test curly hair images
    └── straight/  # Test straight hair images
```

---

## 🧠 Step 2: Set Up Reproducibility

**Critical:** Always run this first for consistent results:

```python
import numpy as np
import torch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 🏗️ Step 3: Build the CNN Model

### Model Architecture Requirements:
- **Input shape:** `(3, 200, 200)` (channels first)
- **Conv2d layer:** 32 filters, kernel_size=(3,3), padding=0, stride=1, ReLU
- **MaxPool2d:** pool_size=(2,2)
- **Flatten**
- **Linear layer:** 64 neurons, ReLU
- **Output layer:** 1 neuron, appropriate activation for binary classification

```python
import torch.nn as nn

class HairCNN(nn.Module):
    def __init__(self):
        super(HairCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 99 * 99, 64)  # 32 * 99 * 99 = 311,328
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        # Note: Don't add sigmoid here if using BCEWithLogitsLoss
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x
```

**Size Calculations:**
- Input: `(3, 200, 200)`
- After conv1: `(32, 198, 198)` → `200 - 3 + 1 = 198`
- After pool1: `(32, 99, 99)` → `198 / 2 = 99`

---

## ❓ Step 4: Answer Questions 1 & 2

### Question 1: Loss Function
**For binary classification, which loss function to use?**

**Answer:** `nn.BCEWithLogitsLoss()`

**Why:**
- Binary classification problem (straight vs curly)
- BCEWithLogitsLoss combines sigmoid activation + binary cross-entropy
- More numerically stable than applying sigmoid + BCELoss separately

### Question 2: Count Parameters

```python
# Method 1: Manual counting
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Method 2: Using torchsummary
from torchsummary import summary
summary(model, input_size=(3, 200, 200))
```

**Manual Calculation:**
- Conv1: `(3×3×3 + 1) × 32 = 28 × 32 = 896`
- FC1: `(32×99×99 + 1) × 64 = 312,769 × 64 = 20,017,216`
- FC2: `(64 + 1) × 1 = 65`
- **Total:** `896 + 20,017,216 + 65 = 20,018,177`

---

## 🎓 Step 5: Set Up Data Loading

### Data Transforms
```python
from torchvision import datasets, transforms

# Initial transforms (no augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Create DataLoaders
```python
from torch.utils.data import DataLoader

# Load datasets
train_dataset = datasets.ImageFolder('data/train', transform=train_transforms)
test_dataset = datasets.ImageFolder('data/test', transform=test_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
```

---

## 🚀 Step 6: Train Initial Model

### Training Setup
```python
import torch.optim as optim

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HairCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)
```

### Training Loop (10 epochs)
```python
num_epochs = 10
history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
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
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(test_dataset)
    val_epoch_acc = correct_val / total_val
    history['val_loss'].append(val_epoch_loss)
    history['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
```

---

## 📊 Step 7: Answer Questions 3 & 4

### Question 3: Median Training Accuracy
```python
import statistics

median_acc = statistics.median(history['acc'])
print(f"Median training accuracy: {median_acc:.4f}")
```

### Question 4: Standard Deviation of Training Loss
```python
std_loss = statistics.stdev(history['loss'])
print(f"Standard deviation of training loss: {std_loss:.4f}")
```

---

## 🔄 Step 8: Data Augmentation

### Add Augmentations
```python
# Enhanced transforms with augmentation
train_transforms_aug = transforms.Compose([
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
```

### Continue Training with Augmentation
```python
# Create new dataset with augmentation
train_dataset_aug = datasets.ImageFolder('data/train', transform=train_transforms_aug)
train_loader_aug = DataLoader(train_dataset_aug, batch_size=20, shuffle=True)

# Train for 10 MORE epochs (don't recreate model!)
num_epochs_aug = 10
history_aug = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(num_epochs_aug):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader_aug:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset_aug)
    epoch_acc = correct_train / total_train
    history_aug['loss'].append(epoch_loss)
    history_aug['acc'].append(epoch_acc)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(test_dataset)
    val_epoch_acc = correct_val / total_val
    history_aug['val_loss'].append(val_epoch_loss)
    history_aug['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs_aug}, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
```

---

## 📈 Step 9: Answer Questions 5 & 6

### Question 5: Mean Test Loss with Augmentations
```python
mean_val_loss = statistics.mean(history_aug['val_loss'])
print(f"Mean validation loss: {mean_val_loss:.4f}")
```

### Question 6: Average Test Accuracy (Last 5 Epochs)
```python
# Get validation accuracy for epochs 6-10
last_5_acc = history_aug['val_acc'][5:]  # epochs 6-10
avg_last_5_acc = statistics.mean(last_5_acc)
print(f"Average validation accuracy (epochs 6-10): {avg_last_5_acc:.4f}")
```

---

## 📝 Step 10: Submit Results

### Final Answer Summary
```python
print("HOMEWORK ANSWERS:")
print("================")
print("Question 1: nn.BCEWithLogitsLoss()")
print(f"Question 2: {total_params}")
print(f"Question 3: {median_acc:.4f}")
print(f"Question 4: {std_loss:.4f}")
print(f"Question 5: {mean_val_loss:.4f}")
print(f"Question 6: {avg_last_5_acc:.4f}")
```

### Submission Link
Submit your answers here: https://courses.datatalks.club/ml-zoomcamp-2025/homework/hw08

---

## ⚠️ Common Issues & Tips

### Troubleshooting
1. **Memory Issues:** Reduce batch_size if running out of GPU memory
2. **Slow Training:** Use GPU if available (`torch.cuda.is_available()`)
3. **Different Results:** Ensure seed is set before every random operation
4. **Wrong Dimensions:** Check tensor shapes with `.shape` attribute

### Key Reminders
- **DON'T recreate the model** between initial training and augmentation
- **USE shuffle=True** for training, **shuffle=False** for test
- **APPLY sigmoid** to outputs only for accuracy calculation, not loss
- **SELECT closest option** if your answer doesn't match exactly

---

## 🎯 Expected Answer Ranges

Based on the homework options, expect approximately:
- **Q1:** `nn.BCEWithLogitsLoss()`
- **Q2:** Around `11,214,912` (closest to calculated values)
- **Q3:** Around `0.40` - `0.84`
- **Q4:** Around `0.007` - `0.171`
- **Q5:** Around `0.08` - `0.88`
- **Q6:** Around `0.28` - `0.68`

Select the **higher value** if your answer is exactly between two options!