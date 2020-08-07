import tensorflow as tf
from tensorflow.contrib import slim
from tools.adjust_brightness import adjust_brightness_from_src_to_dst, read_img
import os,cv2
import numpy as np


def load_test_data(image_path, size):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img,size)
    img = np.expand_dims(img, axis=0)
    return img

def preprocessing(img, size):
    h, w = img.shape[:2]
    if h <= size[0]:
        h = size[0]
    else:
        x = h % 32
        h = h - x

    if w < size[1]:
        w = size[1]
    else:
        y = w % 32
        w = w - y
    # the cv2 resize func : dsize format is (W ,H)
    img = cv2.resize(img, (w, h))
    return img/127.5 - 1.0


def save_images(images, image_path, photo_path = None):
    fake = inverse_transform(images.squeeze())
    if photo_path:
        return imsave(adjust_brightness_from_src_to_dst(fake, read_img(photo_path)),  image_path)
    else:
        return imsave(fake, image_path)

def inverse_transform(images):
    images = (images + 1.) / 2 * 255
    # The calculation of floating-point numbers is inaccurate, 
    # and the range of pixel values must be limited to the boundary, 
    # otherwise, image distortion or artifacts will appear during display.
    images = np.clip(images, 0, 255)
    return images.astype(np.uint8)


def imsave(images, path):
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

def random_crop(img1, img2, crop_H, crop_W):

    assert  img1.shape ==  img2.shape
    h, w = img1.shape[:2]

    # The crop width cannot exceed the original image crop width
    if crop_W > w:
        crop_W = w
    
    # Crop height
    if crop_H > h:
        crop_H = h

    # Randomly generate the position of the upper left corner
    x0 = np.random.randint(0, w - crop_W + 1)
    y0 = np.random.randint(0, h - crop_H + 1)

    crop_1 = crop_image(img1, x0, y0, crop_W, crop_H)
    crop_2 = crop_image(img2, x0, y0, crop_W, crop_H)
    return crop_1,crop_2


def show_all_variables():
    model_vars = tf.trainable_variables()
    # slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    print('G:')
    slim.model_analyzer.analyze_vars([var for var in tf.trainable_variables() if var.name.startswith('generator')], print_info=True)
    # print('D:')
    # slim.model_analyzer.analyze_vars([var for var in tf.trainable_variables() if var.name.startswith('discriminator')], print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')


