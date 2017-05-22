from skimage.io import imread
from scipy.misc import imresize
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


def read_image(path, dshape):
    image = imread(path)
    new_image = imresize(image, tuple(dshape))
    return new_image


def data_preprocessing():
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    return img_prep


def data_augmentation():
    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    return img_aug
