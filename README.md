
#  MNIST Digit Classification using PyTorch

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from the MNIST dataset using PyTorch.  
The notebook is clean, beginner‑friendly, and designed for Google Colab.

---

# 📁 Folder Structure

```
mnist-digit-classification/
│
├── README.md
├── mnist_cnn.ipynb
│
├── data/                     # auto-downloaded MNIST dataset
│
└── saved_models/
    └── mnist_cnn_best.pth    # saved trained model
```

---

#  Project Objective

Train a deep learning model that can accurately classify handwritten digits from 28×28 grayscale images.

Dataset:
- **60,000 training images**
- **10,000 testing images**
- **10 classes (0–9)**

---

#  Features

- Simple & effective CNN model
- Achieves **98–99% accuracy**
- Includes full training + testing
- Includes inference on single images
- Runs smoothly on **Google Colab GPU**
- Beginner‑friendly explanations

---

#  Key Deep Learning Concepts

### ✔ Convolutional Layers  
Extract detectable features (edges, shapes).

### ✔ ReLU Activation  
Introduces non-linearity for complex learning.

### ✔ MaxPooling  
Reduces image size → faster training → less overfitting.

### ✔ Dropout  
Prevents overfitting by randomly dropping neurons.

### ✔ CrossEntropy Loss  
Standard loss for multi-class classification.

### ✔ Adam Optimizer  
Fast convergence with adaptive learning rate.

---

#  Model Architecture

```
Input: 1 × 28 × 28 grayscale image

Conv2d(1 → 32) → ReLU → MaxPool
Conv2d(32 → 64) → ReLU → MaxPool

Flatten  →  3136 features

Linear(3136 → 128) → ReLU → Dropout(0.5)
Linear(128 → 10)

Output: 10 class logits
```

---

# 📊 Expected Results

### • Training Accuracy: **~99%**  
### • Test Accuracy: **98–99%**

Example output:

```
Epoch 1: Train Acc = 97.8% | Test Acc = 98.4%
Epoch 2: Train Acc = 98.9% | Test Acc = 99.1%
Epoch 3: Train Acc = 99.3% | Test Acc = 99.1%
```

Loss and accuracy curves are automatically generated in the notebook.

---

#  How to Use (Google Colab)

### **1. Upload notebook**
Upload **mnist_cnn.ipynb** to Google Colab.

### **2. Enable GPU**
`Runtime → Change runtime type → GPU`

### **3. Run all cells**
Training will begin, evaluate, and save the model to:

```
saved_models/mnist_cnn_best.pth
```

---

#  Example Prediction

```
True Label: 5
Predicted Label: 5
```

The notebook visualizes the digit and shows prediction.

---

#  Possible Enhancements

- Add BatchNorm layers  
- Add lr scheduler  
- Add early stopping  
- Add confusion matrix  
- Visualize CNN filters  
- Use Grad‑CAM for heatmaps  
- Convert model to ONNX  

---

## 👨‍💻 Author Details

Name: M. Brahmanaidu (Muchukuntla Brahmanaidu)
Role: Data Science / AI & ML Aspirant Developer
Email: muchukuntlabrahmanaidu@gmail.com

GitHub: https://github.com/Naidi47

LinkedIn: https://www.linkedin.com/in/brahmanaidu-muchukuntla-17a1a9242/
