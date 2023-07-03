import os

from keras.losses import Loss
from matplotlib import pyplot as plt
from cv2 import cv2
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow as tf
import keras
from keras.metrics import Precision, Recall, BinaryAccuracy

from loaddata import labelfeaturemapping

output_dim_num = len(labelfeaturemapping.get_feature_by_num(0))


def create_model(model_name, input_shape=(64, 64, 3)):
    model = Sequential()

    # block1
    model.add(Conv2D(32, (4, 4), activation='relu', padding='same',input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())

    # block2

    model.add(Conv2D(32, (4, 4), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    # block 3

    model.add(Conv2D(64, (4, 4), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    # FCL
    model.add(Flatten())

    # model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.4))
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dense(256, activation=tf.nn.tanh))
    # model.add(Dropout(0.4))
    model.add(Dense(output_dim_num, activation=tf.nn.tanh))

    model.compile(optimizer='adam', loss=tf.losses.MeanSquaredError(), metrics=['accuracy'])
    model.summary()

    model.save(os.path.join('models', model_name))

    return model


def CustomLoss(Loss):
    def __init__(self):
        super()

    pass


def train_model(model, train_data, val_data, epochs=20):
    log_dir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[tensorboard_callback])
    model.save(os.path.join('models', 'model1.h5'))


def load_model(model_name):
    model = keras.models.load_model(os.path.join('models', model_name))
    return model


def test(test_data):
    model = create_model()
    precision = Precision()
    recall = Recall()
    binary_accuracy = BinaryAccuracy()
    for batch in test_data:
        x, y = batch
        yhat = model.predict(x)
        precision.update_state(y, yhat)
        recall.update_state(y, yhat)
        binary_accuracy.update_state(y, yhat)
        # print(f'Precision: {precision.result().nunpy()}')
        # print(f'Recall: {recall.result().nunpy()}')
        # print(f'Binary Accuracy: {binary_accuracy.result().nunpy()}')


def predict_image(model, image, input_shape=(64, 64, 3)):
    resize = tf.image.resize(image, (input_shape[0], input_shape[1]))
    return model.predict(np.expand_dims(resize / 255.0, 0))


def predict(model, img_path, input_shape=(64, 64, 3)):
    image = cv2.imread(img_path)
    resize = tf.image.resize(image, (input_shape[0], input_shape[1]))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    yhat = model.predict(np.expand_dims(resize / 255.0, 0))
    print(yhat)
    plt.title(np.squeeze(yhat))
    # print(tf.losses.CategoricalHinge().call([[1, 1, 0]], yhat))
    # yhat = model.predict(np.expand_dims(resize, 0))
    plt.show()


def predict_using_trained_model(model_name, img_path):
    model = load_model(model_name)
    predict(model, img_path)


def continue_training_model(model_name, train_data, val_data, epoch=20):
    model = load_model(model_name)
    train_model(model, train_data, val_data, epoch)
