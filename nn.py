import tflearn
from image_utils import data_augmentation, data_preprocessing


def nn():
    # building Deep Learning Network
    input_layer = tflearn.input_data(shape=[None, 128, 128, 3])
    network = tflearn.fully_connected(input_layer, 64, activation='tanh')
    network = tflearn.fully_connected(network, 2, activation='softmax')

    # regression using SGD with learning rate decay
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    network = tflearn.regression(network, optimizer=sgd, metric='accuracy',
                                 loss='categorical_crossentropy')
    return network
