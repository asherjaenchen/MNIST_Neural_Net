# =======================================================================================
#                               MNIST Neural Network Project
# =======================================================================================
# Features: - MNIST dataset images (handrawn digits 0-9)
#           - 60000 training images, 10000 test images
#           - Each image is 28 x 28 = 784 pixels for 784 features
# Labels: Respective number values

# Import path 
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import matplotlib.pyplot as plt

# Download once, then reuse the local file
path = Path("mnist.npz")
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

if not path.exists():
    urlretrieve(url, path)

with np.load(path, allow_pickle=False) as data:
    X_train = data["x_train"]
    y_train = data["y_train"]
    X_test = data["x_test"]
    y_test = data["y_test"]

# Convert to float32 so pixel values represent decimals
X_train = X_train.astype(np.float32)
y_test = X_test.astype(np.float32)

# Scale images to have a pixel brightness from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# Flatten images and return as arrays (features) with 784 columns
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Set aside last 10000 images for validation
X_val = X_train[50000:]
y_val = y_train[50000:]

# Keep the first 50000 images for training
X_train = X_train[:50000]
y_train = y_train[:50000]

# He initialization: for W1 draw standard-normal random numbers and multiply by the
# square root of 2 / 784; for W2 use scaling based on its 128 inputs from W1. Keeps
# signals at useful scale with ReLU.
rng = np.random.default_rng(42)
W1 = (rng.standard_normal((784, 128)) * np.sqrt(2 / 784)).astype(np.float32)
W2 = (rng.standard_normal((128, 10)) * np.sqrt(1 / 128)).astype(np.float32)

# Zero initialization for biases: create 128 hidden-layer biases and 10 output-layer
# biases.
b1 = np.zeros((1, 128), dtype=np.float32)
b2 = np.zeros((1, 10), dtype=np.float32)

# Hidden-layer forward pass. Take first 32 training images using slicing. Then calculate
# each respective neuron Z1 value which is the weighted sum of the images 784 features +
# its bias.
X_batch = X_train[:32]
Z1 = X_batch @ W1 + b1

# Apply ReLU to Z1.
A1 = np.maximum(0, Z1)

Z2 = A1 @ W2 + b2
print("Z2",Z2.shape)

