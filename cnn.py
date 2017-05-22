import tflearn
from tflearn.layers.normalization import local_response_normalization
from tflearn.metrics import Accuracy
from image_utils import data_augmentation, data_preprocessing


def cnn():
    # Building Convolutional Neural Network
    network = tflearn.input_data(shape=[None, 128, 128, 3])
    network = tflearn.conv_2d(network, 32, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = tflearn.conv_2d(network, 64, 3, activation='relu')
    network = tflearn.conv_2d(network, 64, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = tflearn.fully_connected(network, 512, activation='relu')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 2, activation='softmax')
    network = tflearn.regression(network, optimizer='adam',
                                 loss='categorical_crossentropy',
                                 metric=Accuracy(name="Accuracy"),
                                 learning_rate=0.001)
    return network
