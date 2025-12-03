# 🎯 Deep Learning Homework - Final Answers

## Based on Training Results

From our PyTorch CNN training for hair type classification, here are the final answers mapped to the available options:

---

## 📊 Training Results Summary

### Initial Training (10 epochs, no augmentation):
- **Median training accuracy:** 0.7937
- **Standard deviation of training loss:** 0.1541

### With Data Augmentation (10 more epochs):
- **Mean test loss:** 0.5845  
- **Average test accuracy (last 5 epochs):** 0.7184

---

## 🎯 Final Answers for Submission

### Question 1: Loss Function
**Options:** nn.MSELoss(), nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.CosineEmbeddingLoss()

**Answer:** **nn.BCEWithLogitsLoss()**
- For binary classification (straight vs curly hair), BCEWithLogitsLoss is the correct choice

### Question 2: Number of Parameters
**Options:** 896, 11214912, 15896912, 20073473

**Answer:** **20073473**
- Exact calculation: Conv1(896) + FC1(20,072,512) + FC2(65) = 20,073,473

### Question 3: Median of Training Accuracy
**Options:** 0.05, 0.12, 0.40, 0.84

**My result:** 0.7937
**Answer:** **0.84** (closest option)

### Question 4: Standard Deviation of Training Loss
**Options:** 0.007, 0.078, 0.171, 1.710

**My result:** 0.1541
**Answer:** **0.171** (closest option)

### Question 5: Mean of Test Loss
**Options:** 0.008, 0.08, 0.88, 8.88

**My result:** 0.5845
**Answer:** **0.88** (closest option)

### Question 6: Average Test Accuracy (Last 5 Epochs)
**Options:** 0.08, 0.28, 0.68, 0.98

**My result:** 0.7184
**Answer:** **0.68** (closest option)

---

## 📋 Quick Copy-Paste for Submission

```
Question 1: nn.BCEWithLogitsLoss()
Question 2: 20073473
Question 3: 0.84
Question 4: 0.171
Question 5: 0.88
Question 6: 0.68
```

---

## 🔗 Submission URL
https://courses.datatalks.club/ml-zoomcamp-2025/homework/hw08

---

## 📁 Files Created

1. **`homework_solution.py`** - Complete Python script with all training code
2. **`homework_solution.ipynb`** - Jupyter notebook with step-by-step execution and results
3. **`homework_instructions.md`** - Detailed instructions for completing the homework
4. **`homework_submission.md`** - Complete analysis and results
5. **`homework_answers_final.md`** - This file with final answers

All answers are based on actual training results from the CNN model following the exact homework specifications!