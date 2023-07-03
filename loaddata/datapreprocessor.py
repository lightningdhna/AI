import pathlib

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
import os
import imghdr
import numpy
import time
import tensorflow as tf
import math
from PIL import Image

AUTOTUNE = tf.data.AUTOTUNE


class DataLoader:
    image_extensions = ['jpeg', 'jpg', 'bmp', 'png']

    def __init__(self):
        self.input_shape = None
        self.batch_size = None
        self.data_dir = None
        self.source_dir = "/"
        self.labels = []

    def assign_path(self, data_dir):
        self.data_dir = data_dir
        self.source_dir = self.data_dir
        self.labels = os.listdir(self.data_dir)

    def clean_source(self):
        for image_class in self.labels:
            for i, image in enumerate(os.listdir(os.path.join(self.source_dir, image_class))):
                image_path = os.path.join(self.source_dir, image_class, image)
                try:
                    img = cv2.imread(image_path)
                    tip = imghdr.what(image_path)
                    if tip not in self.image_extensions:
                        print('not an Image: {}'.format(image_path))
                        os.remove(image_path)
                    else:
                        os.rename(image_path, os.path.join(self.source_dir, image_class, str(i) + '.' + tip))
                except Exception as e:
                    os.remove(image_path)
                    print('Issue with Image: {}'.format(image_path))
                    print(e)

    def create_data_set(self, input_shape = (64,64,3)):
        val_rate = 0.2
        # img_height = input_shape[0]
        # img_width = input_shape[1]
        self.batch_size = 32
        image_size = (input_shape[0], input_shape[1])
        shuffle = True
        self.input_shape = input_shape

        data = tf.keras.utils.image_dataset_from_directory('data', batch_size=None,
                                                           image_size=image_size,
                                                           shuffle=shuffle, crop_to_aspect_ratio=False,
                                                           label_mode="int")

        labels = []
        images = []
        for image, label in data:
            labels.append(label.numpy())
            images.append(image.numpy())

        # labels = np.concatenate([y.numpy() for x, y in data])
        # images = np.concatenate([x.numpy() for x, y in data])



        # fig, ax = plt.subplots(ncols=6, nrows=6, figsize=(10, 10))
        # for i in range(0,6):
        #     for j in range(0,6):
        #         idx = int(i*6+j)
        #         image = images[idx] /255
        #         label = labels[idx]
        #         ax[i][j].imshow(image)
        #         ax[i][j].title.set_text(label)
        #
        # fig.tight_layout(pad=3)
        # plt.show()
        # plt.show()

        from loaddata.labelfeaturemapping import get_feature_by_num
        attributes = [get_feature_by_num(label) for label in labels]

        data_path = pathlib.Path(self.data_dir)
        image_count = len(list(data_path.glob('*/*')))

        list_ds = tf.data.Dataset.from_tensor_slices((images, attributes))


        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        list_ds = list_ds.map(lambda x, y: (normalization_layer(x), y))

        val_size = int(image_count * val_rate)
        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)

        def configure_for_performance(ds):
            ds = ds.cache()
            ds = ds.shuffle(buffer_size=1000)
            ds = ds.batch(self.batch_size)
            ds = ds.prefetch(buffer_size=AUTOTUNE)
            return ds

        train_ds = configure_for_performance(train_ds)
        val_ds = configure_for_performance(val_ds)

        for image, label in train_ds:
            print(label[0])

        return train_ds, val_ds

    @staticmethod
    def show_images(images, label):
        n = len(images)
        ncol = int(math.ceil(math.sqrt(n)))

        fig, ax = plt.subplots(ncols=ncol, nrows=int(math.ceil(n / ncol)), figsize=(10, 10))
        for ix, img in enumerate(images):
            ax[int(ix / ncol)][ix % ncol].imshow(img)
            ax[int(ix / ncol)][ix % ncol].title.set_text(label[ix])
        fig.tight_layout(pad=3)
        plt.show()

    def show_batch(self, batch):
        self.show_images(batch[0], batch[1])

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


loader = DataLoader()


def load_data_from_folder(_data_dir, batch_size=32):
    loader.set_batch_size(batch_size)
    loader.assign_path(_data_dir)
    return loader
