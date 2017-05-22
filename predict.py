import os
import glob
import tflearn
from image_utils import read_image

TEST_DATA = os.path.join(os.path.dirname(__file__), 'tests')
RESULT_NAME = 'result.csv'


def read_data():
    X = []
    filename = []
    # read cat and dog images respectively: 0 for cat, 1 for dog
    for f in glob.glob(TEST_DATA + '/*.jpg'):
        image = read_image(f, [128, 128, 3])
        X.append(image)
        filename.append(f)

    return X, filename


def predict(model, result_file):
    X, filename = read_data()
    result = model.predict(X)

    count = 1
    with open(result_file, 'w+') as f:
        f.write('id,label\n')
        for i in result:
            # label = 'cat' if i == 0 else 'dog'
            f.write('{},{}\n'.format(filename[count - 1], i))
            count += 1
        f.close()

