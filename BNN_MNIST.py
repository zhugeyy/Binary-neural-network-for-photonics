# -*- coding: utf-8 -*-
"""
Binarized Neural Network (BNN) for MNIST classification
"""

# ============================================================
# Imports (ALL imports are collected here)
# ============================================================
import os

import numpy as np
import tensorflow as tf

import keras.backend as K
from keras import constraints, initializers
from keras.callbacks import LearningRateScheduler
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    InputSpec,
)
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import np_utils

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ============================================================
# Runtime settings
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"   # reduce TensorFlow logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only

np.random.seed(37)


# ============================================================
# 1) Binary Operations
# ============================================================
def round_through(x):
    """
    Straight-Through Estimator (STE):
    - Forward: outputs rounded(x)
    - Backward: uses gradient of x (identity) instead of rounding.
    """
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    """
    Piecewise linear approximation of sigmoid:
      x <= -1 -> 0
      -1 < x < 1 -> 0.5*x + 0.5
      x >= 1 -> 1
    """
    x = 0.5 * x + 0.5
    return K.clip(x, 0, 1)


def binary_tanh_op(x):
    """
    Binary tanh used in BNN:
    - Forward: outputs {-1, +1}
    - Backward: STE based on hard sigmoid
    """
    return 2 * round_through(_hard_sigmoid(x)) - 1


def binarize(W, H=1.0):
    """
    Binarize a tensor into {-H, +H}.
    """
    return H * binary_tanh_op(W / H)


# ============================================================
# 2) Binary Layers
# ============================================================
class Clip(constraints.Constraint):
    """Clip weights into [min_value, max_value]."""

    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value if max_value is not None else -min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


class BinaryDense(Dense):
    """
    Binary fully-connected layer.
    During forward pass, the kernel is binarized before dot-product.
    """

    def __init__(self, units, H=1.0, kernel_lr_multiplier="Glorot", bias_lr_multiplier=None, **kwargs):
        super(BinaryDense, self).__init__(units, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = int(input_shape[1])

        # Original repo uses a Glorot-like scaling for H and learning-rate multiplier.
        if self.H == "Glorot":
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
        if self.kernel_lr_multiplier == "Glorot":
            self.kernel_lr_multiplier = np.float32(1.0 / np.sqrt(1.5 / (input_dim + self.units)))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        binary_kernel = binarize(self.kernel, H=self.H)
        out = K.dot(inputs, binary_kernel)
        if self.use_bias:
            out = K.bias_add(out, self.bias)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def get_config(self):
        config = {
            "H": self.H,
            "kernel_lr_multiplier": self.kernel_lr_multiplier,
            "bias_lr_multiplier": self.bias_lr_multiplier,
        }
        base = super(BinaryDense, self).get_config()
        return dict(list(base.items()) + list(config.items()))


# ============================================================
# 3) Main Script (MNIST)
# ============================================================
class DropoutNoScale(Dropout):
    """
    Standard Keras Dropout scales remaining units by 1/(1-rate).
    For BNN, we avoid that scaling, and compensate by multiplying (1-rate).
    """

    def call(self, inputs, training=None):
        if 0.0 < self.rate < 1.0:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape, seed=self.seed) * (1 - self.rate)

            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs


def binary_tanh(x):
    """Keras-friendly wrapper around the binary activation function."""
    return binary_tanh_op(x)


# ----------------------------
# Dataset loader (MNIST)
# ----------------------------
def load_and_prepare_mnist(target_size=(20, 20)):
    """
    Load and preprocess the MNIST dataset.

    Processing steps:
    1) load the original 28x28 grayscale images,
    2) resize them to `target_size`,
    3) flatten each image into a 1D feature vector,
    4) normalize pixel values to [0, 1],
    5) convert labels to one-hot targets in {-1, +1} for squared hinge loss.

    Returns:
        X_train, Y_train, X_test, Y_test, class_names
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Resize with TensorFlow, then convert back to NumPy for TF1/Keras-style workflows.
    X_train = X_train.transpose(1, 2, 0)
    X_train = tf.image.resize(X_train, target_size)
    X_train = K.eval(X_train).transpose(2, 0, 1)

    X_test = X_test.transpose(1, 2, 0)
    X_test = tf.image.resize(X_test, target_size)
    X_test = K.eval(X_test).transpose(2, 0, 1)

    # Flatten each image into a feature vector for the MLP.
    X_train = X_train.reshape(X_train.shape[0], -1).astype("float32") / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype("float32") / 255.0

    # Convert labels to one-hot format and remap them to {-1, +1}.
    nb_classes = 10
    Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1
    Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1
    class_names = [str(i) for i in range(nb_classes)]

    print("Classes:", class_names)
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_test :", X_test.shape, "Y_test :", Y_test.shape)
    return X_train, Y_train, X_test, Y_test, class_names


# ----------------------------
# Model builder
# ----------------------------
def build_bnn_mlp(input_dim=400, nb_classes=10, num_hidden=1, num_unit=144,
                  H="Glorot", kernel_lr_multiplier="Glorot",
                  use_bias=False, drop_in=0.0, drop_hidden=0.0,
                  epsilon=1e-6, momentum=0.9):
    """
    Build the fully-connected BNN used for MNIST classification.

    The network is intentionally small and simple: dropout on the input,
    one or more hidden binary dense layers, batch normalization, binary
    activation, and a final binary output layer.
    """
    model = Sequential()
    model.add(DropoutNoScale(drop_in, input_shape=(input_dim,), name="drop0"))

    for i in range(num_hidden):
        model.add(BinaryDense(num_unit, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                              use_bias=use_bias, name=f"dense{i+1}"))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name=f"bn{i+1}"))
        model.add(Activation(binary_tanh, name=f"act{i+1}"))
        model.add(DropoutNoScale(drop_hidden, name=f"drop{i+1}"))

    model.add(BinaryDense(nb_classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                          use_bias=use_bias, name="dense"))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name="bn"))
    return model


# ----------------------------
# Train + export
# ----------------------------
def train_and_export(model, X_train, Y_train, X_test, Y_test,
                     lr_start=1e-3, lr_end=1e-5, epochs=50, batch_size=128,
                     history_csv="training_history_mnist.csv",
                     acc_fig="BNN_MNIST_accuracy_curve.jpg",
                     loss_fig="BNN_MNIST_loss_curve.jpg",
                     model_path="mlp_mnist_cpu.h5",
                     weights_dir="weights_mnist_csv",
                     cm_fig="confusion_matrix_MNIST.png",
                     class_names=None):
    """
    Train the model and export all main artifacts.

    Saved artifacts include:
    - CSV training history
    - accuracy and loss curves
    - serialized model file
    - per-layer weight CSV files
    - confusion matrix figure
    """
    lr_decay = (lr_end / lr_start) ** (1.0 / epochs)

    model.summary()
    model.compile(loss="squared_hinge", optimizer=Adam(lr=lr_start), metrics=["acc"])

    lr_scheduler = LearningRateScheduler(lambda e: lr_start * (lr_decay ** e))
    hist = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, Y_test),
        callbacks=[lr_scheduler],
    ).history

    # In legacy Keras / TF1 setups, accuracy keys are usually acc / val_acc.
    acc = hist.get("acc", [])
    val_acc = hist.get("val_acc", [])
    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])

    # Save the epoch-wise training history for later analysis or plotting.
    df = pd.DataFrame({
        "epoch": list(range(len(acc))),
        "accuracy": acc,
        "val_accuracy": val_acc,
        "loss": loss,
        "val_loss": val_loss,
    })
    df.to_csv(history_csv, index=False)

    # Plot and save the training/validation accuracy curves.
    plt.figure()
    plt.plot(df["epoch"], df["accuracy"], label="Training Accuracy")
    plt.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_fig, dpi=600)

    # Plot and save the training/validation loss curves.
    plt.figure()
    plt.plot(df["epoch"], df["loss"], label="Training Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_fig, dpi=600)
    plt.close()

    # Save and reload once so custom layers/constraints are verified correctly.
    model.save(model_path)
    print("Model saved:", model_path)
    model = load_model(
        model_path,
        custom_objects={
            "DropoutNoScale": DropoutNoScale,
            "BinaryDense": BinaryDense,
            "Clip": Clip,
            "binary_tanh": binary_tanh,
        },
    )

    # Export each layer's weights to CSV for inspection or hardware mapping.
    os.makedirs(weights_dir, exist_ok=True)
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for i, w in enumerate(weights):
                out_path = os.path.join(weights_dir, f"{layer.name}_weight_{i}.csv")
                pd.DataFrame(w).to_csv(out_path, index=False, header=False)
                print("Saved:", out_path)

    # Evaluate final performance on the test set.
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Generate and save the confusion matrix for class-wise inspection.
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(Y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        labels = [str(i) for i in range(cm.shape[0])]
    else:
        labels = list(class_names)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (MNIST)")
    plt.tight_layout()
    plt.savefig(cm_fig, dpi=300)
    plt.show()
    print("Confusion matrix saved:", cm_fig)



def main():
    # --------------------
    # Data
    # --------------------
    X_train, Y_train, X_test, Y_test, class_names = load_and_prepare_mnist(
        target_size=(20, 20),
    )

    # --------------------
    # Model
    # --------------------
    model = build_bnn_mlp(
        input_dim=400,
        nb_classes=len(class_names),
        num_hidden=1,
        num_unit=144,
        H="Glorot",
        kernel_lr_multiplier="Glorot",
        use_bias=False,
        drop_in=0.0,
        drop_hidden=0.0,
        epsilon=1e-6,
        momentum=0.9,
    )

    # --------------------
    # Train + export (unified)
    # --------------------
    train_and_export(
        model,
        X_train, Y_train,
        X_test, Y_test,
        lr_start=1e-3,
        lr_end=1e-5,
        epochs=50,
        batch_size=128,
        history_csv="training_history_mnist.csv",
        acc_fig="BNN_MNIST_accuracy_curve.jpg",
        loss_fig="BNN_MNIST_loss_curve.jpg",
        model_path="mlp_mnist_cpu.h5",
        weights_dir="weights_mnist_csv",
        cm_fig="confusion_matrix_MNIST.png",
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
