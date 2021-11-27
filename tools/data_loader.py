import os
import tensorflow as tf
import cv2,random
import numpy as np

class ImageGenerator(object):

    def __init__(self, image_dir,size, batch_size, num_cpus = 16):
        self.paths = self.get_image_paths_train(image_dir)
        self.num_images = len(self.paths)
        self.num_cpus = num_cpus
        self.size = size
        self.batch_size = batch_size

    def get_image_paths_train(self, image_dir):
        paths = []
        for path in os.listdir(image_dir):
            # Check extensions of filename
            if path.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
                continue
            # Construct complete path to anime image
            path_full = os.path.join(image_dir, path)

            # Validate if colorized image exists
            if not os.path.isfile(path_full):
                continue

            paths.append(path_full)
        return paths

    def read_image(self, img_path1):

        if 'style' in img_path1.decode() or 'smooth' in img_path1.decode():
            # color image1
            image1 = cv2.imread(img_path1.decode()).astype(np.float32)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

            # gray image2
            image2 = cv2.imread(img_path1.decode(),cv2.IMREAD_GRAYSCALE).astype(np.float32)
            image2 = np.asarray([image2,image2,image2])
            image2= np.transpose(image2,(1,2,0))

        else:
            # color image1
            image1 = cv2.imread(img_path1.decode()).astype(np.float32)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

            image2 = np.zeros(image1.shape).astype(np.float32)

        return image1, image2

    def load_image(self, img1 ):
        image1, image2 = self.read_image(img1)
        processing_image1 = image1/ 127.5 - 1.0
        processing_image2 = image2/ 127.5 - 1.0
        return (processing_image1,processing_image2)

    def load_images(self):

        dataset = tf.data.Dataset.from_tensor_slices(self.paths)

        # Repeat indefinitely
        dataset = dataset.repeat()

        # Unform shuffle
        dataset = dataset.shuffle(buffer_size=len(self.paths))

        # Map path to image 
        dataset = dataset.map(lambda img: tf.py_func(
            self.load_image, [img], [tf.float32,tf.float32]),
                              self.num_cpus)

        dataset = dataset.batch(self.batch_size)

        img1,img2 = dataset.make_one_shot_iterator().get_next()

        return img1,img2
