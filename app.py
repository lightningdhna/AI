import math
import os

import cv2.cv2
import numpy as np
from matplotlib import pyplot as plt

import cnnmodel
import resnet_model
import handgestureregconition
import loaddata.datapreprocessor


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
    # model = resnet_model.create_resnet_classifier(finger, input_shape)
    # model = inception_model.create_inception_classifier(finger, input_shape)
    train_ds, val_ds = loaddata.datapreprocessor.load_data_from_folder('data').create_data_set_finger(finger,
                                                                                                      input_shape=input_shape)
    # batch = next(iter(train_ds))
    # labels = batch[1]
    # images = batch[0]
    # n = len(images)
    # ncol = int(math.ceil(math.sqrt(n)))
    #
    # fig, ax = plt.subplots(ncols=ncol, nrows=int(math.ceil(n / ncol)), figsize=(10, 10))
    # for ix, img in enumerate(images):
    #     i = int(ix / ncol)
    #     j = int(ix % ncol)
    #     ax[i][j].imshow(img)
    #     ax[i][j].title.set_text(str(labels[ix].numpy()))
    # fig.tight_layout(pad=3)
    # plt.show()

    # for i in range(epoch):
    #     print(f"epoch{i}")

    cnnmodel.continue_training_classifier(finger, train_ds, val_ds, epoch)
    # resnet_model.continue_training_classifier(finger, train_ds, val_ds, epoch)
    # inception_model.continue_training_classifier(finger,train_ds, val_ds, epoch)


def continue_training_classifier(finger, epoch, input_shape=(64, 64, 3)):
    train_ds, val_ds = loaddata.datapreprocessor.load_data_from_folder('data').create_data_set_finger(finger,
                                                                                                      input_shape=input_shape)
    for i in range(epoch):
        print(f"epoch{i}")
        resnet_model.continue_training_classifier(finger, train_ds, val_ds, epoch=1)


def test_model(test_data_dir = 'test'):
    acc = []
    for finger in range(0, 5):
        img_count = 0
        right_ans = 0
        model = cnnmodel.load_classifier(finger, model_type='resnet')
        for label in os.listdir(test_data_dir):
            s = label.split(sep=' ')
            y = float(label[finger])
            for img in os.listdir(os.path.join(test_data_dir, label)):
                img_count += 1
                yhat = cnnmodel.predict_image(model, cv2.cvtColor(cv2.imread(os.path.join(test_data_dir, label, img)),
                                                                  cv2.COLOR_BGR2RGB))
                if (yhat - 0.5) * (y - 0.5) > 0:
                    right_ans += 1

        acc.append(right_ans / img_count)
    print(acc)


def run_model():
    handgestureregconition.run_2()


def test_model2():
    n = 16
    ncol = 4
    models = [cnnmodel.load_classifier(finger) for finger in range(0, 5)]
    fig, ax = plt.subplots(ncols=ncol, nrows=int(math.ceil(n / ncol)), figsize=(10, 10))

    for ix, img in enumerate(os.listdir('test2')):
        i = int(ix / ncol)
        j = int(ix % ncol)
        img = cv2.imread(os.path.join('test2', img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i][j].imshow(img)
        yhats = [np.squeeze(cnnmodel.predict_image(model, img)) for model in models]
        # print(str(yhats))
        output = []
        for yhat in yhats:
            output.append("{:1.1}".format(yhat))
            pass
        ax[i][j].title.set_text(str(output))
    fig.tight_layout(pad=5)
    plt.show()
    pass


if __name__ == "__main__":
    # create_classifier_finger(0,10)
    # threads = [None]*6
    # create_classifier_finger(0,30)
    # continue_training_classifier(0,20)
    # for i in range(0,5):
    # check_data(0)
    # for i in range(0, 5):
    #     create_classifier_finger(i, 20)
    # continue_training_classifier(0,100)
    # create_img_data('11111', 100, data_dir='seen_test')
    # test('resnet0.h5')

    # model = cnnmodel.load_model('2.h5')
    # cnnmodel.predict(model,os.path.join('data1/01110','100.jpeg'))

    # check_data(2)
    # test_model2()
    # print('test seen data')
    # test_model()
    # print('test unseen data: ')
    # test_model('seen_test')

    run_model()
    pass
