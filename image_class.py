import tensorflow as tf
import numpy as np

import keras

import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_labels[5600])

plt.imshow(train_images[5600], cmap='gray', vmin=0, vmax=255)

plt.show()


