# Toyceptron: Pure Python Neural Network

A lightweight, ground-up implementation of a Multi-Layer Perceptron (MLP) built entirely in Python standard libraries. This project demonstrates the fundamental mathematics of deep learning (forward propagation, matrix multiplication simulation, and activation functions) without the overhead of frameworks like TensorFlow, PyTorch, or even NumPy.

## 🧠 Core Architecture

The network is constructed using a modular, object-oriented approach:

* **Neuron (`neuron.py`)**: The fundamental unit. Handles weight initialization (random/fixed), bias, and the dot product calculation $\sum (x \cdot w) + b$.
* **Layer (`layer.py`)**: A collection of neurons operating in parallel. Transforms an input vector $X$ into an output vector $Y$.
* **Network (`network.py`)**: The orchestration class. Manages dynamic architecture construction and sequential forward propagation through hidden layers.

## 🚀 Features

* **Dependency-Free**: Runs on pure Python 3.x.
* **Dynamic Architecture**: Supports any configuration of layers (e.g., `[2, 3, 4, 1]`).
* **Activation Functions**:
    * `Sigmoid` (for binary classification/probabilities)
    * `ReLU` (for deep networks/hidden layers)
    * `Threshold` (binary step)
    * `Identity` (linear)
* **Unit Tested**: Includes built-in verification for every module.

## 🛠️ Usage

To run the system check and verify the network logic:

```bash
python main.py

[Test 1] Initializing Sigmoid Network...
Network Architecture Summary
------------------------------
Layer 1: Input 2 -> Output 3 Neurons
Layer 2: Input 3 -> Output 1 Neurons
------------------------------
>> Status: PASS (Sigmoid behavior confirmed)