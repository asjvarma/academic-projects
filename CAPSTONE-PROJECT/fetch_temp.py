import requests
import os
import gzip
import numpy as np
from tensorflow.keras.datasets import mnist

def fetch_mnist(url):
    # specify the path to the MNIST dataset file in the '/tmp' directory
    file_path = os.path.join('/tmp', os.path.basename(url))

    # check if the file already exists in the '/tmp' directory
    if os.path.isfile(file_path):
        # if the file exists, read it from the '/tmp' directory
        with open(file_path, 'rb') as f:
            dat = f.read()
    else:
        # if the file does not exist, download it from the internet
        with open(file_path, 'wb') as f:
            dat = requests.get(url).content
            f.write(dat)

    # return the MNIST dataset as a numpy array
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

def test_mnist():
    # download the MNIST dataset from the internet
    url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    X_test = fetch_mnist(url)

    # load the MNIST dataset from the '/tmp' directory
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # check if the downloaded MNIST dataset is the same as the one loaded from the '/tmp' directory
    assert np.array_equal(X_test, X_test), "The downloaded MNIST dataset is not the same as the one loaded from the '/tmp' directory."
    assert np.array_equal(y_test, y_test), "The downloaded MNIST dataset is not the same as the one loaded from the '/tmp' directory."

    print("The downloaded MNIST dataset is working correctly.")

test_mnist()