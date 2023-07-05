import math
import os
import threading

import keras.models
import cv2.cv2
import numpy

import cnnmodel
import handgestureregconition
import loaddata.datapreprocessor
from matplotlib import pyplot as plt

from loaddata import labelfeaturemapping


def create_trained_model(epoch=20, input_shape=(64, 64, 3)):
    model = cnnmodel.create_model('model1.h5')
    train_ds, val_ds = loaddata.datapreprocessor.load_data_from_folder('data').create_data_set(input_shape=input_shape,
                                                                                               apply_augmentation=False)
    for i in range(epoch):
        print(f"epoch{i}")
        cnnmodel.train_model(model, train_ds, val_ds, epochs=1)


def create_img_data(label, img_num=50):
    handgestureregconition.recorde_hand_gesture(label, 'data', img_num)


def test(model_name):
    handgestureregconition.run(model_name)


def check_data(finger):
    trainds, valds = loaddata.datapreprocessor.load_data_from_folder('data').create_data_set_finger(finger)
    batch = next(iter(trainds))
    labels = batch[1]
    images = batch[0]
    n = len(images)
    ncol = int(math.ceil(math.sqrt(n)))

    fig, ax = plt.subplots(ncols=ncol, nrows=int(math.ceil(n / ncol)), figsize=(10, 10))
    for ix, img in enumerate(images):
        i = int(ix / ncol)
        j = int(ix % ncol)
        ax[i][j].imshow(img)
        ax[i][j].title.set_text(str(labels[ix].numpy()))
    fig.tight_layout(pad=3)
    plt.show()


def continue_training(epoch=20, input_shape=(64, 64, 3)):
    train_ds, val_ds = loaddata.datapreprocessor.load_data_from_folder('data').create_data_set(input_shape=input_shape)
    for i in range(epoch):
        print(f"epoch{i}")
        cnnmodel.continue_training_model('model1.h5', train_ds, val_ds, epoch=1)


def create_classifier_finger(finger, epoch, input_shape=(64, 64, 3)):
    model = cnnmodel.create_classifier(finger)
    train_ds, val_ds = loaddata.datapreprocessor.load_data_from_folder('data').create_data_set_finger(finger,
                                                                                                      input_shape=input_shape)
    batch = next(iter(train_ds))
    labels = batch[1]
    images = batch[0]
    n = len(images)
    ncol = int(math.ceil(math.sqrt(n)))

    fig, ax = plt.subplots(ncols=ncol, nrows=int(math.ceil(n / ncol)), figsize=(10, 10))
    for ix, img in enumerate(images):
        i = int(ix / ncol)
        j = int(ix % ncol)
        ax[i][j].imshow(img)
        ax[i][j].title.set_text(str(labels[ix].numpy()))
    fig.tight_layout(pad=3)
    plt.show()


    for i in range(epoch):
        print(f"epoch{i}")
        cnnmodel.continue_training_classifier(finger, train_ds, val_ds)


def continue_training_classifier(finger, epoch, input_shape=(64, 64, 3)):
    train_ds, val_ds = loaddata.datapreprocessor.load_data_from_folder('data').create_data_set_finger(finger,input_shape=input_shape)
    for i in range(epoch):
        print(f"epoch{i}")
        cnnmodel.continue_training_classifier(finger, train_ds, val_ds,epoch=1 )

from keras.layers import *
import tensorflow as tf

if __name__ == "__main__":
    # threads = [None]*6
    # for i in range(5):
    #     threads[i] = threading.Thread(target = continue_training_classifier,args=(i,100))
    #     threads[i].start()
    for i in range(0,5):
        continue_training_classifier(i,50)
    # continue_training_classifier(0,100)
    # create_img_data('11111',100)
    # test('0.h5')


    # check_data(0)

    pass
