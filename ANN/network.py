import numpy as np
from keras import utils,layers,models, datasets,callbacks
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignores the warning about CPU Tensorflow configuration.

BATCH_SIZE = 32
EPOCHS = 3
CALLBACK = callbacks.EarlyStopping(monitor='val_loss', patience=2)


class NeuralNetwork:
    """ The Neural Network class. """

    def get_model(self):
        """
        Creates, trains and returns the Convolutional Neural Network.
        :return: Compiled CNN model. Model score on test set ~97.5%
        """

        model = models.Sequential([
                 layers.Conv2D(filters=64, kernel_size=3, input_shape = (28,28,1), padding='same', activation='relu'),
                 layers.MaxPooling2D(pool_size=(2,2), strides=2),
                 layers.Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu'),
                 layers.Flatten(),
                 layers.Dense(units = 128, activation='relu'),
                 layers.Dense(units=10, activation='softmax')
])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()               # loading MNIST dataset

train_shape, test_shape = train_images.shape[0], test_images.shape[0]
train_images = utils.normalize(train_images, axis=1).reshape(train_shape,28,28,1)                   # normalizing images between 0-1 range
test_images = utils.normalize(test_images, axis=1).reshape(test_shape,28,28,1)


model = NeuralNetwork().get_model()
model.fit(train_images, train_labels,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=[test_images, test_labels],
          callbacks=[CALLBACK])

try:
    with open('model.h5', 'wb') as file:
        pickle.dump(model, file)                # serialize the trained ANN
except:
    print('Serialization failed')
