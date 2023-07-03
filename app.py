import math
import os

import keras.models
from cv2 import cv2

import cnnmodel
import handgestureregconition
import loaddata.datapreprocessor
from matplotlib import pyplot as plt


def create_trained_model(epoch=20, input_shape=(64, 64, 3)):
    model = cnnmodel.create_model('model1.h5')
    train_ds, val_ds = loaddata.datapreprocessor.load_data_from_folder('data').create_data_set(input_shape=input_shape)
    for i in range(epoch):
        cnnmodel.train_model(model, train_ds, val_ds, epochs=1)


def create_img_data(label, img_num=50):
    handgestureregconition.recorde_hand_gesture(label, 'data', img_num)


def test():
    handgestureregconition.run()


def check_data():
    trainds, valds = loaddata.datapreprocessor.load_data_from_folder('data').create_data_set()
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
        cnnmodel.continue_training_model('model1.h5', train_ds, val_ds, epoch=1)


if __name__ == "__main__":
    # create_img_data('one',500)

    # create_trained_model(epoch=100)
    # model = cnnmodel.load_model('model1.h5')
    # cnnmodel.predict(model,os.path.join('data/one','2.jpeg'))
    # continue_training(200)

    # test()
    # check_data()

    pass
