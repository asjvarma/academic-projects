from tensorflow.keras.datasets import mnist

import tensorflow as tf
from tensorflow import keras

# load the MNIST dataset from the Keras dataset module
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print the shapes of the training and testing datasets
print("Training dataset shape:", X_train.shape, y_train.shape)
print("Testing dataset shape:", X_test.shape, y_test.shape)