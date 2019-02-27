#!/usr/bin/env python
# _*_coding:utf-8_*_
import tensorflow as tf
import random
import os
from tqdm import tqdm
import threading

trainPaths = list()
testPaths = list()
classes = list()


def load_files():
    f = open("/media/data/oldcopy/PythonProject/Food101/food-101/meta/classes.txt")
    lines = f.readlines()
    for line in lines:
        classes.append(line.strip('\n'))
    f.close()
    f = open("/media/data/oldcopy/PythonProject/Food101/food-101/meta/train.txt")
    lines = f.readlines()
    for line in lines:
        trainPaths.append(line.strip('\n'))
    f.close()
    f = open("/media/data/oldcopy/PythonProject/Food101/food-101/meta/test.txt")
    lines = f.readlines()
    for line in lines:
        testPaths.append(line.strip('\n'))
    f.close()


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def transform_label(name):
    return classes.index(name)


def probuf(label, image_raw):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example.SerializeToString()


def writerRecord(save_path, HOME_PATH, coord, t_id):
    writerTrain = tf.python_io.TFRecordWriter(os.path.join(save_path, "train" + str(t_id) + ".tfrecords"))
    writerTest = tf.python_io.TFRecordWriter(os.path.join(save_path, "test" + str(t_id) + ".tfrecords"))
    randIndexTrain = random.sample(range(0, len(trainPaths)), len(trainPaths))

    size = int(len(randIndexTrain) / 4)
    randIndexTrain = randIndexTrain[t_id * size: (t_id + 1) * size]

    randIndexTest = random.sample(range(0, len(testPaths)), len(testPaths))
    size = int(len(randIndexTest) / 4)
    randIndexTest = randIndexTest[t_id * size: (t_id + 1) * size]

    sess = tf.Session()
    for i in tqdm(randIndexTrain, "thread" + str(t_id) + "Start write train tfrecords"):
        image_string = tf.read_file(os.path.join(HOME_PATH, trainPaths[i] + ".jpg"))
        image_string = sess.run(image_string)
        label = trainPaths[i].split("/")[0]
        label = transform_label(label)
        writerTrain.write(probuf(label, image_string))
    writerTrain.close()
    for i in tqdm(randIndexTest, "thread" + str(t_id) + "Start write test tfrecords"):
        image_string = tf.read_file(os.path.join(HOME_PATH, testPaths[i] + ".jpg"))
        image_string = sess.run(image_string)
        label = testPaths[i].split("/")[0]
        label = transform_label(label)
        writerTest.write(probuf(label, image_string))
    writerTest.close()
    coord.request_stop()


if __name__ == '__main__':
    # sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    load_files()
    threads = [threading.Thread(target=writerRecord, args=("/media/data/oldcopy/PythonProject/Food101/TFRecord",
                                                           "/media/data/oldcopy/PythonProject/Food101/food-101/images",
                                                           coord, i)) for i in range(4)]
    for t in threads:
        t.start()
    coord.join(threads)
