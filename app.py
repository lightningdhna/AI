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


def create_img_data(label, img_num=50, data_dir='test'):
    handgestureregconition.recorde_hand_gesture(label, data_dir, img_num)


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
    train_ds, val_ds = loaddata.datapreprocessor.load_data_from_folder('data').create_data_set_finger(finger,
                                                                                                      input_shape=input_shape)
    for i in range(epoch):
        print(f"epoch{i}")
        cnnmodel.continue_training_classifier(finger, train_ds, val_ds, epoch=1)


from keras.layers import *
import tensorflow as tf


def test_model():
    acc = []
    for finger in range(0, 5):
        img_count = 0
        right_ans = 0;
        model = cnnmodel.load_classifier(finger)
        for label in os.listdir('test'):
            s = label.split(sep=' ')
            y = float(s[finger])
            for img in os.listdir(os.path.join('test', label)):
                img_count += 1
                yhat = cnnmodel.predict(model, img)
                if (yhat - 0.5) * (y - 0.5) > 0:
                    right_ans += 1

        acc.append(right_ans / img_count)


def test_model2():
    n = 16
    ncol = 4
    models = [cnnmodel.load_classifier(finger) for finger in range(0, 5)]
    fig, ax = plt.subplots(ncols=ncol, nrows=int(math.ceil(n / ncol)), figsize=(10, 10))

    for ix, img in enumerate(os.listdir('test2')):
        i = int(ix / ncol)
        j = int(ix % ncol)
        ax[i][j].imshow(img)
        yhats = [cnnmodel.predict_image(model, img) for model in models]
        output = []
        for yhat in yhats:
            output.append("{:1.1".format(yhat))
            pass
        ax[i][j].title.set_text(str(output))
    fig.tight_layout(pad=3)
    plt.show()
    pass


if __name__ == "__main__":
    # threads = [None]*6
    # create_classifier_finger(0,30)
    # continue_training_classifier(0,20)
    for i in range(0, 5):
        create_classifier_finger(i, 20)
    # continue_training_classifier(0,100)
    # create_img_data('10111',1,data_dir='test2')
    # test('0.h5')

    # model = cnnmodel.load_model('2.h5')
    # cnnmodel.predict(model,os.path.join('data/01110','100.jpeg'))

    # check_data(2)

    pass
