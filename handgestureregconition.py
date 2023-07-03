import os.path
import pathlib
import time

import numpy
from cv2 import cv2

import cnnmodel

cam = cv2.VideoCapture(0)

# model = None
# model = cnnmodel.load_model('model1.h5')


def yhat_tostring(yhat):
    return numpy.array2string(yhat)


def set_prediction(window_name, yhat):
    cv2.setWindowTitle(window_name, yhat_tostring(yhat))
    pass


def run():
    model = cnnmodel.load_model('model1.h5')
    while True:
        retval, frame = cam.read()
        if not retval: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yhat = cnnmodel.predict_image(model, frame_rgb)
        window_name = 'test'
        cv2.imshow(window_name, frame)
        set_prediction(window_name, yhat)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def recorde_hand_gesture(label_name, data_dir, img_num):
    frame_step = 2
    img_folder_path = os.path.join(data_dir, label_name)
    if not os.path.isdir(img_folder_path):
        os.mkdir(img_folder_path)
    os.chdir(img_folder_path)
    frame_counter = int(0)

    # data_path = pathlib.Path(img_folder_path)
    # image_counter = int(len(list(data_path.glob('*'))))
    image_counter = int(len(os.listdir()))

    start_recording = False
    change_title = False
    cv2.setWindowTitle('window', 'Preparing')
    while img_num > 0:
        retval, frame = cam.read()
        cv2.imshow('window', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            start_recording = True
        if not retval:
            break
        if not start_recording:
            continue
        if not change_title:
            cv2.setWindowTitle('window', 'recoding label for class ' + label_name)
            change_title = True
        if frame_counter % frame_step == 0:
            image_counter += 1
            img_num -= 1
            print(f'saving image {image_counter}.jpeg')
            cv2.imwrite(str(image_counter) + '.jpeg', frame)
        frame_counter += 1

    cam.release()
    cv2.destroyAllWindows()
