import requests, gzip, os, hashlib
import pygame
from matplotlib import pyplot as plt
import numpy as np
from hopfield_network import HopfieldNetwork

# Constants
# Binarization Threshold
p = 20
# Shape
im_shape = (28, 28)
# Plotting flag
verbose_plotting = True

# Fetch MNIST dataset from the ~SOURCE~
def fetch_MNIST(url):
    fp = os.path.join("\tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)

    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

# def fetch_MNIST(url):
#     # specify the path to the MNIST dataset file in the '/tmp' directory
#     file_path = os.path.join('\tmp', os.path.basename(url))

#     # check if the file already exists in the '/tmp' directory
#     if os.path.isfile(file_path):
#         # if the file exists, read it from the '/tmp' directory
#         with open(file_path, 'rb') as f:
#             dat = f.read()
#     else:
#         # if the file does not exist, download it from the internet
#         with open(file_path, 'wb') as f:
#             dat = requests.get(url).content
#             f.write(dat)

#     # return the MNIST dataset as a numpy array
#     return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

def view_image(arr, title=""):
    image = arr.reshape(im_shape)
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()



def MNIST_Hopfield():
    # test out the Hopfield_Network object on some MNIST data
    # fetch MNIST dataset for some random memory downloads

    X = fetch_MNIST(
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    )[0x10:].reshape((-1, 784))

    X_test = fetch_MNIST(
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    )[0x10:].reshape((-1, 784))

    X_labels = fetch_MNIST(
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    )[0x8:]

    X_test_labels = fetch_MNIST(
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    )[0x8:]

    data_by_label = []
    for i in range(0, 10):
        data_by_label.append(X[X_labels == i, :])

    memory_patterns = []
    for i in range(0, 10):
        memory_patterns.append(np.mean(data_by_label[i], axis=0))
    if verbose_plotting:
        for memory_pattern in memory_patterns:
            view_image(memory_pattern)


    # convert to binary
    X_binary = np.where(X > p, 1, -1)
    memory_patterns_binary = []
    for memory_pattern in memory_patterns:
        memory_patterns_binary.append(np.where(memory_pattern > p, 1, -1))

    # View binarized patterns
    if verbose_plotting:
        for memory_pattern in memory_patterns_binary:
            view_image(memory_pattern)

    # Set memories
    memories_list = np.array(memory_patterns_binary)

    # initialize Hopfield object
    H_Net = HopfieldNetwork(memories_list)
    H_Net.network_learning()


    # plot weights matrix
    if verbose_plotting:
        plt.figure("weights", figsize=(10, 7))
        plt.imshow(H_Net.weights, cmap='RdPu')  #
        plt.xlabel("Each row/column represents a neuron, each square a connection")

        plt.title(" 4096 Neurons - 16,777,216 unique connections", fontsize=15)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.show()

    random_sample = np.random.randint(0, X_binary.shape[0])
    label = X_labels[random_sample]
    image_sample = X_binary[random_sample]

    view_image(image_sample, f"Digit chosen {label}")
    H_Net.state = image_sample.copy().reshape((-1, 1))

    iteration = 0
    while not H_Net.settled:
        # update network state
        view_image(H_Net.state.reshape(im_shape), f"Iteration: {iteration}")
        H_Net.update_network_state(np.prod(im_shape))
        H_Net.compute_energy()
        iteration += 1

    # plot energies
    plt.figure("Energy", figsize=(10, 7))
    x = np.arange(len(H_Net.energies))
    plt.scatter(x, np.array(H_Net.energies), s=5, color='red')
    plt.xlabel("Generation")
    plt.ylabel("Energy")
    plt.title("Network Energy over Successive Generations", fontsize=15)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    exact_matches = []
    hamming_distances = []
    for memory_pattern in memory_patterns_binary:
        exact_matches.append(np.array_equal(H_Net.state, memory_pattern))
        hamming_distances.append((H_Net.state.flatten() != memory_pattern.flatten()).sum())
    print("Exact matches")
    print(exact_matches)
    print("Hamming Distances")
    print(hamming_distances)



    # cells = H_Net.state.reshape(im_shape).T




MNIST_Hopfield()
plt.show()