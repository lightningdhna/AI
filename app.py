import tensorflow as tf
import os
import cv2
import imghdr
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Convolution2D, ZeroPadding2D
from matplotlib import pyplot as plt
from keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np
from keras.models import load_model

data_dir = 'data'
labels = os.listdir(data_dir)
image_extensions = ['jpeg', 'jpg', 'bmp', 'png']


def clean_source():
    for image_class in labels:
        for i, image in enumerate(os.listdir(os.path.join(data_dir, image_class))):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_extensions:
                    print('not an Image: {}'.format(image_path))
                    os.remove(image_path)
                else:
                    os.rename(image_path, os.path.join(data_dir, image_class, str(i) + '.' + tip))
            except Exception as e:
                os.remove(image_path)
                print('Issue with Image: {}'.format(image_path))
                print(e)


def create_data_set():
    batch_size = 64
    image_size = (256, 256)
    shuffle = True

    data = tf.keras.utils.image_dataset_from_directory('data', batch_size=batch_size, image_size=image_size,
                                                       shuffle=shuffle, crop_to_aspect_ratio=False)
    data = data.map(lambda x, y: (x / 255.0, y))

    data_size = len(data)
    train_size = int(data_size * 0.7)
    val_size = int(data_size * 0.2)
    test_size = data_size - train_size - val_size

    train_data = data.take(train_size)
    val_data = data.skip(train_size).take(val_size)
    test_data = data.skip(train_size + val_size).take(test_size)
    return train_data, val_data, test_data


def load_saved_model(model_name):
    try:
        model = load_model(os.path.join('models', model_name + '.h5'))
        return model
    except Exception as e:
        pass


def create_model(model_name):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    model.compile(optimizer='SGD', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    model.save(os.path.join('models', model_name + '.h5'))
    return model


def load_and_train_model(model_name):
    try:
        model = load_model(os.path.join('models', model_name + '.h5'))
        train_data, val_data, test_data = create_data_set()
        train_model(model, train_data, val_data)
    except Exception as e:
        print(e)


def train_model(model, train_data, val_data):
    log_dir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    history = model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[tensorboard_callback])

    # fig = plt.figure()
    # plt.plot(history.history()['loss'], color='yellow', label='loss')
    # plt.plot(history.history()['val_loss'], color='blue', label='val_loss')
    # fig.subtitle('Loss', fontsize=20)
    # plt.legend(loc='upper left')
    # plt.show()


# test


def test():
    train_data, val_data, test_data = create_data_set()
    model = create_model()
    train_model(model, train_data, val_data)

    precision = Precision()
    recall = Recall()
    binary_accuracy = BinaryAccuracy()
    train, val, test_data = create_data_set()
    for batch in test_data:
        x, y = batch
        yhat = model.predict(x)
        precision.update_state(y, yhat)
        recall.update_state(y, yhat)
        binary_accuracy.update_state(y, yhat)
        # print(f'Precision: {precision.result().nunpy()}')
        # print(f'Recall: {recall.result().nunpy()}')
        # print(f'Binary Accuracy: {binary_accuracy.result().nunpy()}')


def test_predict(model, img_path):
    image = cv2.imread(img_path)
    resize = tf.image.resize(image, (256, 256))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    yhat = model.predict(np.expand_dims(resize / 255.0, 0))
    # yhat = model.predict(np.expand_dims(resize, 0))
    print(yhat)

def show_image(img_path):
    image = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.show()

def main():
    model_name = 'model2.h5'
    train_data, val_data, test_data = create_data_set()
    if not load_saved_model(model_name):
        model = create_model(model_name)
        train_model(model, train_data, val_data)
    else:
        model = load_saved_model(model_name)

    test_predict(model, 'img.png')


if __name__ == "__main__":
    # clean_source()
    # main()
    pass