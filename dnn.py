import tflearn
from image_utils import data_augmentation, data_preprocessing


def dnn():
    # building Deep Learning Network
    input_layer = tflearn.input_data(shape=[None, 128, 128, 3])
    dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout1 = tflearn.dropout(dense1, 0.8)
    dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout2 = tflearn.dropout(dense2, 0.8)
    softmax = tflearn.fully_connected(dropout2, 2, activation='softmax')

    # regression using SGD with learning rate decay
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    net = tflearn.regression(softmax, optimizer=sgd,
                             loss='categorical_crossentropy')
    return net
