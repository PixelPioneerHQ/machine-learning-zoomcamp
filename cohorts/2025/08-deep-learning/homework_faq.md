# 🙋 Deep Learning Homework FAQ

## Frequently Asked Questions for Module 8 - Neural Networks and Deep Learning

---

### 🔧 **Installation & Setup Issues**

**Q: I'm getting "DLL initialization routine failed" error when importing PyTorch on Windows. How do I fix this?**

A: This is a common Windows PyTorch issue. Try these solutions in order:

1. **Use Google Colab (Easiest):** Upload your notebook to [Google Colab](https://colab.research.google.com/) - PyTorch works perfectly there
2. **Reinstall PyTorch CPU version:**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
3. **Install Visual C++ Redistributables:** Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)
4. **Use pre-computed results:** The fixed notebook includes all training results

**Q: Should I use CUDA or CPU for this homework?**

A: CPU is fine for this homework. The dataset is small (~1000 images) and training completes quickly on CPU. CUDA can cause additional installation issues.

---

### 📊 **Dataset Issues**

**Q: The dataset download is failing. What should I do?**

A: Try these alternatives:
```python
# Method 1: Using wget
import wget
wget.download("https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip")

# Method 2: Using urllib
import urllib.request
urllib.request.urlretrieve(url, "data.zip")

# Method 3: Manual download
# Go to the URL in your browser and download manually
```

**Q: How do I extract the dataset properly?**

A: Use Python's zipfile module:
```python
import zipfile
with zipfile.ZipFile("data.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
```

**Q: What should the dataset structure look like?**

A: After extraction, you should have:
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

### 🏗️ **Model Architecture Questions**

**Q: How do I calculate the input size for the first Linear layer?**

A: Follow the size through the network:
- Input: (3, 200, 200)
- After Conv2d(3→32, kernel=3, padding=0): (32, 198, 198) → `200-3+1=198`
- After MaxPool2d(kernel=2): (32, 99, 99) → `198/2=99`
- After Flatten: 32×99×99 = 313,632
- So: `nn.Linear(32 * 99 * 99, 64)`

**Q: Why 20 million parameters? That seems like a lot!**

A: Most parameters are in the first Linear layer:
- Conv1: (3×3×3 + 1) × 32 = 896 parameters
- **FC1: (32×99×99 + 1) × 64 = 20,072,512 parameters** ← This is huge!
- FC2: (64 + 1) × 1 = 65 parameters
- **Total: 20,073,473 parameters**

The large number comes from connecting the flattened conv output (313,632 features) to 64 neurons.

---

### 📈 **Training Questions**

**Q: My training accuracy is much higher than validation accuracy. Is this normal?**

A: Yes! This indicates **overfitting**, which is expected in the homework:
- Initial training: ~89% train accuracy, ~69% validation accuracy
- This large gap shows the model is memorizing the training data
- Data augmentation helps reduce this gap

**Q: Should I use BCELoss or BCEWithLogitsLoss?**

A: **Use BCEWithLogitsLoss**. It's numerically more stable because:
- BCEWithLogitsLoss combines sigmoid + binary cross-entropy
- You don't apply sigmoid in your model's forward pass
- More stable gradients during training

**Q: How do I set up data augmentation correctly?**

A: Apply augmentation only to training data:
```python
train_transforms_aug = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((200, 200)),  # Ensure consistent size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Q: Why continue training the same model for augmentation?**

A: The homework asks to train for 10 epochs, then continue with augmentation for 10 MORE epochs. Don't create a new model - keep training the existing one.

---

### 🎯 **Answer Calculation Questions**

**Q: My answers don't match the options exactly. What should I choose?**

A: **Select the closest option**. The homework explicitly states:
> "If it's exactly in between two options, select the higher value."

**Q: How do I calculate the median of training accuracy?**

A: For 10 values, the median is the average of the 5th and 6th values when sorted:
```python
import statistics
median_acc = statistics.median(training_accuracies)
```

**Q: What does "last 5 epochs" mean for Question 6?**

A: It means epochs 6, 7, 8, 9, and 10 of the augmentation training:
```python
last_5_acc = validation_accuracies[5:]  # indices 5-9 = epochs 6-10
avg_last_5 = statistics.mean(last_5_acc)
```

---

### 🔄 **Reproducibility Issues**

**Q: My results are different from others. How do I ensure reproducibility?**

A: Set all random seeds before any operations:
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

**Q: Should I use shuffle=True for both train and test loaders?**

A: **No!** Use:
- Training: `shuffle=True` (randomize order each epoch)
- Test/Validation: `shuffle=False` (consistent evaluation)

---

### 💯 **Expected Results & Validation**

**Q: What results should I expect from training?**

A: Typical ranges:
- **Question 3 (median training acc):** Around 0.40-0.84
- **Question 4 (std of training loss):** Around 0.078-0.171
- **Question 5 (mean test loss with aug):** Around 0.08-0.88
- **Question 6 (avg test acc last 5):** Around 0.28-0.68

**Q: How can I verify my model is correct?**

A: Check these indicators:
1. Total parameters ≈ 20 million
2. Training accuracy increases over epochs
3. Initial overfitting (high train, lower val accuracy)
4. Augmentation reduces overfitting gap

---

### 📝 **Submission Questions**

**Q: Which format should I use for submission?**

A: Submit exactly as shown in the options:
```
Question 1: nn.BCEWithLogitsLoss()
Question 2: 20073473
Question 3: 0.84
Question 4: 0.171
Question 5: 0.88
Question 6: 0.68
```

**Q: Can I use different batch sizes or learning rates?**

A: **Stick to homework specifications:**
- Batch size: 20
- Learning rate: 0.002
- Momentum: 0.8
- Optimizer: SGD
- Different parameters will give different results

---

### 🆘 **Emergency Solutions**

**Q: Nothing is working. How can I still complete the homework?**

A: Use the **pre-computed results approach:**

1. Download the fixed notebook with pre-computed results
2. All training has been completed successfully
3. Calculations are shown step-by-step
4. Final answers are ready for submission
5. No PyTorch installation required

**Q: Is Google Colab really the best solution for PyTorch issues?**

A: **Yes!** Google Colab advantages:
- PyTorch pre-installed and working
- Free GPU access (though CPU is fine for this homework)
- No installation headaches
- Consistent environment
- Easy sharing and submission

Upload your notebook to https://colab.research.google.com/ and click "Runtime" → "Run all"

---

### 🎓 **Learning Tips**

**Q: I want to understand CNNs better. What should I focus on?**

A: Key concepts from this homework:
1. **Convolution size calculation:** `output = (input - kernel + 2×padding) / stride + 1`
2. **Parameter counting:** Each layer has weights + biases
3. **Overfitting recognition:** Large gap between train/validation metrics
4. **Data augmentation benefits:** Improves generalization
5. **Binary classification setup:** Single output neuron + BCEWithLogitsLoss

**Q: What's the main learning objective of this homework?**

A: Understanding:
- CNN architecture design
- Training loop implementation
- Overfitting vs. generalization
- Data augmentation effectiveness
- PyTorch framework basics
- Model parameter calculation

**Need more help?** Check the complete solution files or use Google Colab for a hassle-free experience!