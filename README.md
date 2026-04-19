# Binarized Neural Networks (BNN) in Keras

This repository provides a clean and unified implementation of **Binarized Neural Networks (BNNs)** using Keras (TensorFlow backend), supporting both standard datasets and custom image datasets.

---

## 🚀 Overview

This project implements BNNs with:

* Binary weights and activations (±1)
* Straight-Through Estimator (STE)
* Fully-connected MLP architecture
* Reproducible training pipeline
* Automatic export of results

Two example applications are included:

| Script         | Description                        |
| -------------- | ---------------------------------- |
| `BNN_MNIST.py` | BNN for MNIST digit classification |
| `BNN_MIR_image.py`   | BNN for custom MIR image dataset   |

---

## 🧠 Model Highlights

* Binary activation: `sign(x)`
* Binary weights: `{-H, +H}`
* Loss function: **squared hinge loss**
* Optimizer: Adam
* Supports BatchNorm + Dropout (BNN-friendly)

---

## 📦 Outputs (Auto Generated)

After training, the following files are generated:

* `training_history_*.csv` → training logs
* `BNN_*_accuracy_curve.jpg` → accuracy curve
* `BNN_*_loss_curve.jpg` → loss curve
* `confusion_matrix_*.png` → confusion matrix
* `weights_*_csv/` → exported weights
* `.h5` model file

---

## 🖥️ Installation

Recommended environment:

```bash
Python 3.7
TensorFlow 1.15
Keras 2.2.x
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pillow
```

---

## ▶️ Usage

### MNIST Example

```bash
python BNN_MNIST.py
```

---

### MIR Dataset Example

```bash
python BNN_MIR_image.py --data "path/to/dataset"
```

---

## 📁 MIR Dataset Format

```
dataset_root/
    training_image/
        class1/
        class2/
    testing_image/
        class1/
        class2/

---

## ⚠️ Notes

* Code runs in **CPU-only mode by default**
* Compatible with **TensorFlow 1.x + Keras backend**
* GPU usage may require additional configuration

