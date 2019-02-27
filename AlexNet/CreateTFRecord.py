#!/usr/bin/env python
# _*_coding:utf-8_*_
import tensorflow as tf
import random
import os
import cv2

classes = list()
f = open("/media/heolis/967EC257F5104FE6/oldcopy/PythonProject/Food101/food-101/meta/classes.txt")
lines = f.readlines()
for line in lines:
    classes.append(line.strip('\n'))


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def transform_label(name):
    return classes.index(name)

def creatData(HOME_PATH):
    totalFileList = list()
    filenameTrain = '/media/heolis/967EC257F5104FE6/oldcopy/PythonProject/Food101/TFRecord/train.tfrecords'
    filenameTest = '/media/heolis/967EC257F5104FE6/oldcopy/PythonProject/Food101/TFRecord/test.tfrecords'
    writerTrain = tf.python_io.TFRecordWriter(filenameTrain)
    writerTest = tf.python_io.TFRecordWriter(filenameTest)
    for folder in classes:
        filePath = os.path.join(HOME_PATH, folder)
        imageList = os.listdir(filePath)
        for imgName in imageList:
            imgPath = os.path.join(filePath, imgName)
            totalFileList.append(imgPath)
    randIndexList = random.sample(range(0, len(totalFileList)), len(totalFileList))
    t = 0
    for i in randIndexList:
        path = totalFileList[i]
        print(path)
        raw = cv2.imread(path)
        res = cv2.resize(raw, (227, 227), interpolation=cv2.INTER_CUBIC)
        image_raw = res.tostring()
        label = transform_label(path.split('/')[2])
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)
        }))
        if t < len(totalFileList) * 0.8:
            writerTrain.write(example.SerializeToString())
        else:
            writerTest.write(example.SerializeToString())
        t += 1
    writerTest.close()
    writerTrain.close()




if __name__ == '__main__':
    creatData('/media/heolis/967EC257F5104FE6/oldcopy/PythonProject/Food101/food-101/images/')