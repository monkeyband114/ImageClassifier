import tensorflow as tf
import numpy as np

import keras

import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_labels[1])

# plt.imshow(train_images[1], cmap='gray', vmin=0, vmax=255)

# plt.show()


model = keras.Sequential([
    
    keras.layers.Flatten(input_shape=(28, 28)),
    
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    
    keras.layers.Dense(units=10, activation =tf.nn.softmax)
])


model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# model.save('image_class.h5')

avg_loss = model.evaluate(test_images, test_labels)

prediction = model.predict(test_images)

print(test_labels[10])

plt.imshow(train_images[10], cmap='gray', vmin=0, vmax=255)

plt.show()

print(prediction[10])

print(list(prediction[10]).index(max(prediction[10])))