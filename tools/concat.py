# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from tqdm import tqdm

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def preprocessing(img, size=[256,256]):
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
    return img

if __name__ == '__main__':

    for dirpath, dirnames, filenames in os.walk('../dataset/test/HR_photo'):
        style = '../results/Hayao/concat/'
        check_folder(style)
        c1 = len(filenames)
        i = 0
        print(c1)
        for filepath in tqdm(filenames):
            i += 1
            img_path1 = os.path.join(dirpath, filepath)
            img_path2 = os.path.join(dirpath.replace('dataset/test/','results/Hayao/'), filepath)

            img1 = cv2.imread(img_path1)
            img1 = preprocessing(img1)
            img2 = cv2.imread(img_path2)

            assert img1.shape == img2.shape

            h,w, c= img1.shape

            cut1 = np.ones((h, 7, 3), dtype='u8') * 255

            im_A1 = np.concatenate([img1, cut1], 1)
            im_AB = np.concatenate([im_A1, img2], 1)

            cv2.imwrite(style + str(i) + '.jpg', im_AB)



