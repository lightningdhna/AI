from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from keras.losses import Loss
from matplotlib import pyplot as plt
from cv2 import cv2
import numpy as np
from keras import Sequential, layers, backend
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, ZeroPadding2D, \
    GlobalAveragePooling2D
import tensorflow as tf
import keras
from keras.metrics import Precision, Recall, BinaryAccuracy
import os
import warnings

import cnnmodel


# backend = None
# layers = None
# models = None
# keras_utils = None


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet50(input_shape=(64, 64, 3)):
    # Determine proper input shape
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    inputs = keras.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3))(inputs)
    x = layers.Conv2D(8, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    #block 1
    x = conv_block(x, 3, [8, 8, 32], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [8, 8, 32], stage=2, block='b')
    x = identity_block(x, 3, [8, 8, 32], stage=2, block='c')
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='d')

    #Block 2
    x = conv_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = identity_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = identity_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = identity_block(x, 3, [16, 16, 64], stage=3, block='d')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='e')

    #Block 3
    x = conv_block(x, 3, [32, 32 ,128], stage=4, block='a')
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='f')

    #Block 4
    x = conv_block(x, 3, [64, 64, 256], stage=5, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=5, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=5, block='c')
    # x = identity_block(x, 3, [64, 64, 256], stage=5, block='e')


    x = GlobalAveragePooling2D()(x)
    x = Dense(256,activation=tf.nn.relu)(x)
    # x = layers.Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, x)
    model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()
    return model


def create_resnet_classifier(finger, input_shape=(64, 64, 3)):
    model = ResNet50(input_shape)
    model.save(os.path.join('models', 'resnet' + str(finger) + '.h5'))
    return model


def continue_training_classifier(finger, train_ds, val_ds, epoch=1):
    model_name = 'resnet' + str(finger) + '.h5'
    cnnmodel.continue_training_model(model_name, train_ds, val_ds, epoch)
