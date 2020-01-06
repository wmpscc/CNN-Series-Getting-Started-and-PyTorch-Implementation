'''
利用tf.Example创建TFRecords
'''
import tensorflow as tf
import numpy as np
import IPython.display as display
import cv2
import os


# The following functions can be used to convert a value to a type compatible with tf.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_label(folderName):
    label_dict = {
        'Sample001': 0,
        'Sample002': 1,
        'Sample003': 2,
        'Sample004': 3,
        'Sample005': 4,
        'Sample006': 5,
        'Sample007': 6,
        'Sample008': 7,
        'Sample009': 8,
        'Sample010': 9,
        'Sample011': 10,
    }
    return label_dict[folderName]


def image_example(image_string, label):
    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def make_tfrecords(HOME_PATH):
    trainPath = 'train.tfrecords'
    folderList = os.listdir(HOME_PATH)
    with tf.io.TFRecordWriter(trainPath) as writer:
        for folder in folderList:
            label = get_label(folder)
            current_folder = os.path.join(HOME_PATH, folder)
            imageList = os.listdir(current_folder)
            print(current_folder)
            for imgPath in imageList:
                img = cv2.imread(os.path.join(current_folder, imgPath))[:, :, ::-1]
                image_string = cv2.imencode('.jpg', img)[1].tostring()
                tf_example = image_example(image_string, label)
                writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    make_tfrecords('../dataset/Fnt10/')
