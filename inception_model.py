import os

import keras
from keras.applications import imagenet_utils

import cnnmodel

import os

from keras import layers, backend, models
import tensorflow as tf


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name)(x)
    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 4, 1)
        branch_1 = conv2d_bn(x, 4, 1)
        branch_1 = conv2d_bn(branch_1, 4, 3)
        branch_2 = conv2d_bn(x, 4, 1)
        branch_2 = conv2d_bn(branch_2, 6, 3)
        branch_2 = conv2d_bn(branch_2, 8, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 24, 1)
        branch_1 = conv2d_bn(x, 16, 1)
        branch_1 = conv2d_bn(branch_1, 20, [1, 7])
        branch_1 = conv2d_bn(branch_1, 24, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 24, 1)
        branch_1 = conv2d_bn(x, 24, 1)
        branch_1 = conv2d_bn(branch_1, 28, [1, 3])
        branch_1 = conv2d_bn(branch_1, 32, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   backend.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x


def InceptionResNetV2(input_shape):
    img_input = keras.Input(shape=input_shape)
    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 4, 3, strides=2, padding='valid')
    # x = conv2d_bn(x, 4, 3, padding='valid')
    # x = conv2d_bn(x, 8, 3)
    # x = layers.AveragePooling2D(4, strides=2, padding='same')(x)
    # x = conv2d_bn(x, 10, 1, padding='valid')
    # x = conv2d_bn(x, 24, 3, padding='valid')
    # x = layers.MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 12, 1)
    branch_1 = conv2d_bn(x, 6, 1)
    branch_1 = conv2d_bn(branch_1, 8, 5)
    branch_2 = conv2d_bn(x, 8, 1)
    branch_2 = conv2d_bn(branch_2, 12, 3)
    branch_2 = conv2d_bn(branch_2, 12, 3)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 8, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    icb_nums=[4,8,4]
    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(0,icb_nums[0]):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 48, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 32, 1)
    branch_1 = conv2d_bn(branch_1, 32, 3)
    branch_1 = conv2d_bn(branch_1, 48, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)
    print(x.shape)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(0, icb_nums[1]):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    print(x.shape)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 32, 1)
    branch_0 = conv2d_bn(branch_0, filters=48, kernel_size=3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 32, 1)
    branch_1 = conv2d_bn(branch_1, 36, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn(x, 32, 1)
    branch_2 = conv2d_bn(branch_2, 36, 3)
    branch_2 = conv2d_bn(branch_2, 40, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(0,icb_nums[0]):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 192, 1, name='conv_7b')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(32, activation='tanh')(x)
    x = layers.Dense(1, activation='sigmoid', name='predictions')(x)

    # Create model.
    model = models.Model(img_input, x, name='inception_resnet_v2')
    model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    return model


def create_inception_classifier(finger, input_shape=(64, 64, 3)):
    model = InceptionResNetV2(input_shape)
    model.save(os.path.join('models', 'inception_' + str(finger) + '.h5'))
    return model


def continue_training_classifier(finger, train_ds, val_ds, epoch=1):
    model_name = 'inception_' + str(finger) + '.h5'
    cnnmodel.continue_training_model(model_name, train_ds, val_ds, epoch)
