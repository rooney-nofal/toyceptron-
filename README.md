# Toyceptron: Pure Python Deep Learning Framework

A lightweight, ground-up implementation of a Neural Network built entirely in Python standard libraries. 

Unlike simple wrappers, this project implements the core mathematics of Deep Learning from scratch: **Matrix Multiplication**, **Activation Functions**, and **Backpropagation via Gradient Descent**.

## 🧠 Core Architecture

The network is constructed using a modular, object-oriented approach:

* **Neuron (`neuron.py`)**: The fundamental unit. Handles weights, bias, forward dot products, and **gradient updates**.
* **Layer (`layer.py`)**: Manages parallel processing and propagates error terms (deltas) backward.
* **Network (`network.py`)**: The orchestration class. Handles dynamic architecture, forward passes, and the **training loop**.

## 🚀 Features

* **Dependency-Free**: Runs on pure Python 3.x (No NumPy, No PyTorch).
* **Backpropagation Engine**: Implements the Chain Rule to calculate gradients and update weights.
* **Dynamic Architecture**: Supports any configuration (e.g., `[2, 4, 1]` for XOR).
* **Activation Functions**:
    * `Sigmoid` + Derivative (for probabilities)
    * `ReLU` + Derivative (for deep hidden layers)
* **XOR Solver**: Capable of learning non-linear patterns.

## 🛠️ Usage

### Training the Network (XOR Problem)
The `main.py` script demonstrates training the network to solve the classic XOR problem.

```bash
python main.py

TOYCEPTRON - TRAINING MODE
==========================
Initializing Network...
Training for 20000 epochs...

Epoch 2000: Total Error = 0.12507
Epoch 10000: Total Error = 0.04329
Epoch 20000: Total Error = 0.02906

Training Complete! Testing results:
------------------------------
Input: [0.0, 0.0] -> Prediction: 0.0035 -> Rounded: 0 (Target: 0.0)
Input: [0.0, 1.0] -> Prediction: 0.9929 -> Rounded: 1 (Target: 1.0)
Input: [1.0, 0.0] -> Prediction: 0.9927 -> Rounded: 1 (Target: 1.0)
Input: [1.0, 1.0] -> Prediction: 0.0111 -> Rounded: 0 (Target: 0.0)

SUCCESS: The Network has learned XOR!