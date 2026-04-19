# Binary-neural-network-for-photonics
This repository provides a clean and unified implementation of Binarized Neural Networks (BNNs) using Keras (TensorFlow backend), supporting both standard datasets and custom image datasets.
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
| `mnist_bnn.py` | BNN for MNIST digit classification |
| `mir_bnn.py`   | BNN for custom MIR image dataset   |

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
python mnist_bnn.py
```

---

### MIR Dataset Example

```bash
python mir_bnn.py --data "path/to/dataset"
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
```

---

## 📊 Example Results

The model typically achieves:

* MNIST accuracy: ~97–98% (depending on config)
* Fast convergence with binary constraints

---

## 🔬 Research Context

This implementation is designed for:

* Low-power neural networks
* Hardware-aware AI (e.g., photonic computing)
* Probabilistic computing with noise

---

## ⚠️ Notes

* Code runs in **CPU-only mode by default**
* Compatible with **TensorFlow 1.x + Keras backend**
* GPU usage may require additional configuration

---

## 📌 Future Work

* Convolutional BNN support
* FPGA / photonic hardware mapping
* Integration with probabilistic neural networks (PBNN)

---

## 📄 License

MIT License

---

## 🙋 Contact

Author: Your Name
Affiliation: National University of Singapore (NUS)
Email: [your_email@nus.edu.sg](mailto:your_email@nus.edu.sg)
