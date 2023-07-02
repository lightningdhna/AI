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

    # def create_data_set():
    #     batch_size = 32
    #     image_size = (256, 256)
    #     shuffle = True
    #
    # data = tf.keras.utils.image_dataset_from_directory('data', batch_size=batch_size, image_size=image_size,
    # shuffle=shuffle, crop_to_aspect_ratio=False, label_mode="int")
    #
    #     labels = np.concatenate([y for x, y in data])
    #     images = np.concatenate([x for x, y in data])
    #
    #     # labels = tf.constant(labels)
    #     # images = tf.constant(images)
    #
    #     dataset = tf.data.Dataset.from_tensor_slices((images,labels))
    #
    #     cnt= 0
    #     for x, y in dataset:
    #         cnt+=1
    #     print(cnt)
    #
    #
    #     data = data.map(lambda x,y: (x / 255.0,y ))
    #
    #     data_size = len(data)
    #     train_size = int(data_size * 0.7)
    #     val_size = int(data_size * 0.2)
    #     test_size = data_size - train_size - val_size
    #
    #     train_data = data.take(train_size)
    #     val_data = data.skip(train_size).take(val_size)
    #     test_data = data.skip(train_size + val_size).take(test_size)
    #
    #     return train_data, val_data, test_data

    def create_data_set(self):
        val_rate = 0.2
        img_height = 256
        img_width = 256
        batch_size = 64
        image_size = (256, 256)
        shuffle = True

        data = tf.keras.utils.image_dataset_from_directory('data', batch_size=batch_size,
                                                           image_size=image_size,
                                                           shuffle=shuffle, crop_to_aspect_ratio=False,
                                                           label_mode="int")

        labels = np.concatenate([y for x, y in data])
        images = np.concatenate([x for x, y in data])
        from loaddata.labelfeaturemapping import get_feature_by_num
        attribute = [get_feature_by_num(label) for label in labels]

        data_path = pathlib.Path(self.data_dir)
        image_count = len(list(data_path.glob('*/*')))

        list_ds = tf.data.Dataset.from_tensor_slices((images, attribute))
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
        # list_ds = tf.data.Dataset.list_files(str(data_path / '*/*'), shuffle=False)

        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        list_ds = list_ds.map(lambda x, y: (normalization_layer(x), y))

        val_size = int(image_count * val_rate)
        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)

        def configure_for_performance(ds):
            ds = ds.cache()
            ds = ds.shuffle(buffer_size=1000)
            ds = ds.batch(batch_size)
            ds = ds.prefetch(buffer_size=AUTOTUNE)
            return ds

        train_ds = configure_for_performance(train_ds)
        val_ds = configure_for_performance(val_ds)

        for image, label in train_ds:
            print(image[0])
            print(label[0])

        batch = next(iter(train_ds))
        print(batch[1].shape)

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


loader = DataLoader()


def load_data_from_foler(_data_dir):
    loader.assign_path(_data_dir)
    return loader
