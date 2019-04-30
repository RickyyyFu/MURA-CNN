# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import random
import keras.backend as K


def load_path(root_path='../valid', size=512):
    '''
    load MURA data
    '''

    Path = []
    labels = []
    for root, dirs, files in os.walk(root_path):  # read all images, os.walk returns iterators genertor traverses all files
        for name in files:

            if root.split('_')[-1] == 'positive':  # positive label == 1；
                path_1 = os.path.join(root, name)
                Path.append(path_1)
                labels += [1]
            elif root.split('_')[-1] == 'negative': # negative label == 1；
                path_1 = os.path.join(root, name)
                Path.append(path_1)
                labels += [0]
            else:
                continue
    print (len(Path))
    labels = np.asarray(labels)
    return Path, labels


def load_image(Path='../valid', size=512):
    Images = []
    i = 1
    for path in Path:

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("null")
            continue
        else:
            print "loading image ", i
            image = cv2.resize(image, (size, size))   # resize images
            image = randome_rotation_flip(image, size)
            Images.append(image)
            i = i+1

    print (len(Images))
    Images = np.asarray(Images).astype('float32')

    print "Starting normalization......."
    mean = np.mean(Images[:, :, :])  # normalization
    std = np.std(Images[:, :, :])
    Images[:, :, :] = (Images[:, :, :] - mean) / std
    print "normalization finished"

    if K.image_data_format() == "channels_first":
        Images = np.expand_dims(Images, axis=1)  # extended dimension 1
    if K.image_data_format() == "channels_last":
        Images = np.expand_dims(Images, axis=3)  # extended dimension 3(usebackend tensorflow:aixs=3; theano:axixs=1)


    return Images


def randome_rotation_flip(image, size=512):
    if random.randint(0, 1):
        image = cv2.flip(image, 1)

    if random.randint(0, 1):
        angle = random.randint(-30, 30)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
        # third parameter: the transformed image size
        image = cv2.warpAffine(image, M, (size, size))
    return image


if __name__ == '__main__':
    load_path()
    load_image()
