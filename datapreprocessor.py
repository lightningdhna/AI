import cv2
from matplotlib import pyplot as plt
import os
import imghdr
import numpy
import time
import tensorflow as tf
import math
from PIL import Image
data_source_dir = 'source'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
classification_class = os.listdir(data_source_dir)


### clean dữ liệu
def clean_source_data():
    for image_class in classification_class:
        for image in os.listdir(os.path.join(data_source_dir, image_class)):
            image_path = os.path.join(data_source_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print('Issue with image {}'.format(image_path))




def get_image(image_name, image_class='0', data_directory=source_data_dir):
    try:
        image_path = os.path.join(data_directory, image_class, image_name)
        print(image_path)
        img = cv2.imread(image_path)
        tip = imghdr.what(image_path)

        return img

    except Exception as e:
        print(e)


def open_image(image_name, image_class='0', data_directory=source_data_dir):
    try:
        img = get_image(image_name, image_class, data_directory)
        print(img.shape)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    except Exception as e:
        print(e)

#
# data = tf.keras.utils.image_dataset_from_directory(source_data_dir,
#                                                    batch_size=16,
#                                                    image_size=(256, 256),
#                                                    shuffle=True)
# batch_iterator = data.as_numpy_iterator()
#
#
# def get_batch():
#     batch = batch_iterator.next()
#     return batch
#
#
# # todo
#
# def show_batch(batch):
#     print(batch[0].shape)
#     a, size, size, dep = batch[0].shape
#     row = int(math.sqrt(a))
#     col = math.ceil(a / row)
#     fig, ax = plt.subplots(ncols=col, nrows=row)
#     for idx, img in enumerate(batch[0]):
#         ax[idx].imshow(img.astype(int))
#         ax[idx].title.set_text(class_name[batch[1][idx]])
#     plt.show()
#
#
# data = data.map(lambda x, y: (x / 255, y))
# data_iterator = data.as_numpy_iterator()
#
#
# def get_training_batch():
#     return data_iterator.next()


# tạo dữ liệu traning, test
# làm sạch dữ liệu, các file k đúng định dạng

def run():
    clean_source_data()
    #tao training data
    #todo
    return
