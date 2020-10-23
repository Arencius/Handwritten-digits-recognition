from keras import *
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignores the warning about CPU Tensorflow configuration.


class NeuralNetwork:
    ''' The Neural Network class. '''

    def __init__(self, X, y):
        '''
        :param X: vector with training images data for the model to train on
        :param y: vector of labels of training images
        '''
        self.train_images = X
        self.train_labels = y

    def __str__(self):
        return self.get_model().layers

    def get_model(self):
        '''
        Creates, trains and returns the Neural Network.
        :return: Sequential object trained on the data passed to the class constructor. Model score on test set ~97.5%
        '''
        model = Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),  # 1st layer with 128 neurons and rectifier activation function
            layers.Dense(128, activation='relu'),  # 2nd layer with 128 neurons and rectifier activation function
            layers.Dense(64, activation='relu'),  # 3rd layer with 128 neurons and rectifier activation function
            layers.Dense(10, activation='softmax')  # softmax optimization function for probability distribution
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.train_images, self.train_labels, epochs=6)

        return model


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()  # loading MNIST dataset
train_images = utils.normalize(train_images, axis=1)  # normalizing images between 0-1 range
test_images = utils.normalize(test_images, axis=1)

model = NeuralNetwork(train_images, train_labels).get_model()

try:
    with open('model.h5', 'wb') as file:
        pickle.dump(model, file)  # serialize the trained ANN
except:
    print('Serialization failed')
