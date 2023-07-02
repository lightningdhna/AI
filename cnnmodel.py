import os

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


def create_model(model_name):
    model = Sequential()

    #block1
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())

    #block2
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    #block 3
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    #block 4

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    #block 5

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    #FCL
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim_num, activation=tf.nn.sigmoid))

    model.compile(optimizer='sgd', loss=tf.losses.Huber(), metrics=['accuracy'])
    model.summary()

    model.save(os.path.join('models', model_name))
    return model


def train_model(model, train_data, val_data, epochs=20):
    log_dir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[tensorboard_callback])


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


def predict_image(model, image):
    resize = tf.image.resize(image, (256, 256))
    return model.predict(np.expand_dims(resize / 255.0, 0))


def predict(model, img_path):
    image = cv2.imread(img_path)
    resize = tf.image.resize(image, (256, 256))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    yhat = model.predict(np.expand_dims(resize / 255.0, 0))
    print(yhat)
    plt.title(np.squeeze(yhat))
    print(tf.losses.CategoricalHinge().call([[1, 1, 0]], yhat))
    # yhat = model.predict(np.expand_dims(resize, 0))
    plt.show()


def load_model(model_name):
    return keras.models.load_model(os.path.join('models', model_name))


def predict_using_trained_model(model_name, img_path):
    model = load_model(model_name)
    predict(model, img_path)


def continue_training_model(model_name, train_data, val_data, epoch=20):
    model = load_model(model_name)
    train_model(model, train_data, val_data, epoch)
