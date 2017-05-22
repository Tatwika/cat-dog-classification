import tflearn
from image_utils import data_augmentation, data_preprocessing


def resnet():
    # residual blocks: 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
    n = 5

    # building Residual Network
    network = tflearn.input_data(shape=[None, 128, 128, 3])
    network = tflearn.conv_2d(network, 16, 3, regularizer='L2', weight_decay=0.0001)
    network = tflearn.residual_block(network, n, 16)
    network = tflearn.residual_block(network, 1, 32, downsample=True)
    network = tflearn.residual_block(network, n - 1, 32)
    network = tflearn.residual_block(network, 1, 64, downsample=True)
    network = tflearn.residual_block(network, n - 1, 64)
    network = tflearn.batch_normalization(network)
    network = tflearn.activation(network, 'relu')
    network = tflearn.global_avg_pool(network)

    # regression
    network = tflearn.fully_connected(network, 2, activation='softmax')
    network = tflearn.regression(network, optimizer='adam', loss='categorical_crossentropy')
    return network
