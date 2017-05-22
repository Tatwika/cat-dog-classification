import os
import glob

import tflearn
import tensorflow as tf
from sklearn.model_selection import train_test_split

from nn import nn
from cnn import cnn
from dnn import dnn
from resnet import resnet
from predict import predict
from image_utils import read_image

TRAIN_DATA = os.path.join(os.path.dirname(__file__), 'images')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, mode=0o755)
MODEL_NAME = 'dog_vs_cat.model'


def read_data():
    X = []
    Y = []
    # read cat and dog images respectively: 0 for cat, 1 for dog
    for f in glob.glob(TRAIN_DATA + '/cat/*.jpg'):
        label = 0
        image = read_image(f, [128, 128, 3])
        X.append(image)
        Y.append(label)
    for f in glob.glob(TRAIN_DATA + '/dog/*.jpg'):
        label = 1
        image = read_image(f, [128, 128, 3])
        X.append(image)
        Y.append(label)

    # split training data and validation set data
    X, X_test, y, y_test = train_test_split(X, Y,
                                            test_size=0.2,
                                            random_state=42)
    return (X, y), (X_test, y_test)


if __name__ == '__main__':
    (X, Y), (X_test, Y_test) = read_data()
    Y = tflearn.data_utils.to_categorical(Y, 2)
    Y_test = tflearn.data_utils.to_categorical(Y_test, 2)

    # training with simple neural network
    model_name_NN = 'nn_' + MODEL_NAME
    model_NN = tflearn.DNN(nn(), checkpoint_path='model_NN',
                           max_checkpoints=10, tensorboard_verbose=3)

    model_NN.fit(X, Y, n_epoch=50, validation_set=(X_test, Y_test),
                  show_metric=True, batch_size=32, shuffle=True,
                  run_id="nn_cat_vs_dog")
    model_NN.save(os.path.join(MODEL_PATH, model_name_NN))

    # predicting
    predict(model_NN, 'NN_result.csv')
    tf.reset_default_graph()

    # training with Deep Learning Network
    model_name_DNN = 'dnn_' + MODEL_NAME
    model_DNN = tflearn.DNN(dnn(), checkpoint_path='model_DNN',
                            max_checkpoints=10, tensorboard_verbose=3)
    model_DNN.fit(X, Y, n_epoch=50, validation_set=(X_test, Y_test),
                  show_metric=True, batch_size=32, shuffle=True,
                  run_id="dense_model_cat_vs_dog")
    model_DNN.save(os.path.join(MODEL_PATH, model_name_DNN))

    # predicting
    predict(model_DNN, 'DNN_result.csv')
    tf.reset_default_graph()

    # training with Convolutional Network
    model_name_CNN = 'cnn_' + MODEL_NAME
    model_CNN = tflearn.DNN(cnn(), checkpoint_path='model_CNN',
                            max_checkpoints=10, tensorboard_verbose=3)
    model_CNN.fit(X, Y, n_epoch=50, validation_set=(X_test, Y_test),
                  show_metric=True, batch_size=32, shuffle=True,
                  run_id="cnn_cat_vs_dog")
    model_CNN.save(os.path.join(MODEL_PATH, model_name_CNN))

    # predicting
    predict(model_CNN, 'CNN_result.csv')
    tf.reset_default_graph()

    # training with Residual Network
    model_name_RN = 'resnet_' + MODEL_NAME
    model_RN = tflearn.DNN(resnet(), checkpoint_path='model_resnet',
                           max_checkpoints=10, tensorboard_verbose=3,
                           clip_gradients=0.)
    model_RN.fit(X, Y, n_epoch=50, validation_set=(X_test, Y_test),
                 snapshot_epoch=False, snapshot_step=500,
                 show_metric=True, batch_size=32, shuffle=True,
                 run_id='resnet_cat_vs_dog')
    model_RN.save(os.path.join(MODEL_PATH, model_name_RN))

    # predicting
    predict(model_RN, 'RN_result.csv')
    tf.reset_default_graph()
