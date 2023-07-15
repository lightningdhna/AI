import os

from keras.losses import Loss
from matplotlib import pyplot as plt
from cv2 import cv2
import numpy as np
from keras import Sequential, layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow as tf
import keras
from keras.metrics import Precision, Recall, BinaryAccuracy

from loaddata import labelfeaturemapping

output_dim_num = 5

def create_classifier(finger, input_shape=(64, 64, 3)):
    model = Sequential()

    # block 1
    model.add(Conv2D(8, (3, 3), activation=tf.nn.relu, padding='same', input_shape=input_shape))

    model.add(Conv2D(8, (3, 3), activation=tf.nn.relu, padding='same'))
    # model.add(Conv2D(8, (3, 3), activation=tf.nn.relu, padding='same'))
    # model.add(Conv2D(8, (3, 3), activation=tf.nn.relu, padding='same'))

    model.add(layers.AveragePooling2D())
    # model.add(Conv2D(16,(7,7),strides=(2,2),padding='same', activation=tf.nn.relu))
    # model.add(MaxPooling2D())

    # block 2
    model.add(Conv2D(16, (3, 3), activation=tf.nn.relu, padding='same'))
    model.add(Conv2D(16, (3, 3), activation=tf.nn.relu, padding='same'))
    model.add(Conv2D(16, (3, 3), activation=tf.nn.relu, padding='same'))
    model.add(Conv2D(16, (3, 3), activation=tf.nn.relu, padding='same'))
    # model.add(Conv2D(16, (3, 3), activation=tf.nn.relu, padding='same'))

    model.add(layers.AveragePooling2D())

    # model.add(Conv2D(64,(2,2),strides=(2,2),padding='same'))
    # model.add(Conv2D(32,(5,5),strides=(2,2),padding='same', activation=tf.nn.relu))
    # model.add(MaxPooling2D())

    # block 3
    model.add(Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same'))
    model.add(Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same'))
    model.add(Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same'))
    model.add(Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same'))
    model.add(Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same'))

    model.add(layers.AveragePooling2D())
    # model.add(Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same'))
    # model.add(Conv2D(64,(3,3),strides=(2,2),padding='same', activation='relu'))

    # block 4và t
    model.add(Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same'))
    model.add(Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same'))
    model.add(Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same'))
    model.add(Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same'))
    model.add(Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same'))

    model.add(layers.AveragePooling2D())

    # model.add(Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same'))
    # model.add(Conv2D(64,(3,3),strides=(2,2),padding='same', activation='relu'))

    # FCL
    model.add(Flatten())

    model.add(Dense(64, activation=tf.nn.tanh))
    model.add(Dense(64, activation=tf.nn.tanh))
    model.add(Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer='adam', loss=tf.losses.MeanSquaredError(), metrics=['accuracy'])

    model.summary()
    model.save(os.path.join('models', str(finger) + '.h5'))

    return model


def load_classifier(finger):
    model_name = 'resnet' + str(finger) + '.h5'
    print(f"loading{model_name}")
    return load_model(model_name)


def continue_training_classifier(finger, train_data, val_data, epoch=1):
    model_name = str(finger) + '.h5'
    return continue_training_model(model_name, train_data, val_data)


def create_model(model_name, input_shape=(64, 64, 3)):
    model = Sequential()

    # block 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D())

    # block 2

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D())

    # block 3

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D())

    # block 4

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D())

    # block 4

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D())

    # FCL
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(512, activation=tf.nn.relu))
    # model.add(Dropout(0.5))
    model.add(Dense(512, activation=tf.nn.relu))
    # model.add(Dropout(0.5))
    model.add(Dense(output_dim_num, activation=tf.nn.tanh))

    model.compile(optimizer='sgd', loss=tf.losses.Huber(), metrics=['accuracy'])
    model.summary()

    model.save(os.path.join('models', model_name))

    return model


def CustomLoss(Loss):
    def __init__(self):
        super()

    pass


# def train_model(model, train_data, val_data, epochs=1):
#     log_dir = 'logs'
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
#     history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[tensorboard_callback])
#     model.save(os.path.join('models', 'model1.h5'))


def load_model(model_name):
    model = keras.models.load_model(os.path.join('models', model_name))
    print(f"load model {model_name}")
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


def continue_training_model(model_name, train_data, val_data, epoch):
    print(f"training model{model_name}")
    model = load_model(model_name)
    log_dir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    history = model.fit(train_data, epochs=epoch, validation_data=val_data, callbacks=[tensorboard_callback])

    model.save(os.path.join('models', model_name))
    print(f"save to {os.path.join('models',model_name)}")

    fig, ax  = plt.subplots(nrows=2,ncols=1,figsize= (10,6))

    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].title.set_text('model loss')
    ax[0].set(xlabel = 'epoch', ylabel = 'loss')
    # ax[0].xlabel('epoch')
    ax[0].legend(['train', 'val'], loc='upper left')

    ax[1].plot(history.history['accuracy'])
    ax[1].plot(history.history['val_accuracy'])
    ax[1].title.set_text('model accuracy')
    # ax[1].ylabel('accuracy')
    # ax[1].xlabel('epoch')
    ax[1].set(xlabel = 'epoch', ylabel = 'accuracy', yticks =[0,0.25,0.5,0.75,1])
    ax[1].legend(['train', 'val'], loc='upper left')
    plt.tight_layout(pad=5)
    plt.savefig(f"training_{model_name}.jpeg")
    # plt.show()
