'''
   made by @finnkso (github)
   2020.04.09
   tensorflow-gpu==1.15.0  : tf.compat.v1
   if tensorflow-gpu==1.8.0, please replayce tf.compat.v1 to tf
'''
import argparse
import os
import tkinter as tk
from tkinter import filedialog
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from net import generator,generator_lite
from tools.utils import preprocessing, check_folder
from tools.adjust_brightness import adjust_brightness_from_src_to_dst

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    desc = "Tensorflow implementation of AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--video', type=str, default='video/input/'+ '2.mp4',
                        help='video file or number for webcam')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoint/generator_Paprika_weight',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--output', type=str, default='video/output/' + 'Paprika',
                        help='output path')
    parser.add_argument('--output_format', type=str, default='MP4V',
                        help='codec used in VideoWriter when saving video to file')
    """
    output_format: xxx.mp4('MP4V'), xxx.mkv('FMP4'), xxx.flv('FLV1'), xxx.avi('XIVD')
    ps. ffmpeg -i xxx.mkv -c:v libx264 -strict -2 xxxx.mp4, this command can convert mkv to mp4, which has small size.
    """

    return parser.parse_args()

def getfileloc(initialdir='/', method='open', title='Please select a file', filetypes=(("video files", ".mkv .avi .mp4"), ("all files","*.*"))):
    root = tk.Tk()
    if method == 'open':
        fileloc = filedialog.askopenfilename(parent=root, initialdir=initialdir, title=title, filetypes=filetypes)
    elif method == 'save':
        fileloc = filedialog.asksaveasfilename(parent=root, initialdir=initialdir, initialfile='out.avi', title=title, filetypes=filetypes)
    root.withdraw()
    return fileloc

def convert_image(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = np.asarray(img)
    return img

def inverse_image(img):
    img = (img.squeeze()+1.) / 2 * 255
    img = img.astype(np.uint8)
    return img

def cvt2anime_video(video, output, checkpoint_dir, output_format='MP4V', img_size=(256,256)):
    '''
    output_format: 4-letter code that specify codec to use for specific video type. e.g. for mp4 support use "H264", "MP4V", or "X264"
    '''
    gpu_stat = bool(len(tf.config.experimental.list_physical_devices('GPU')))
    if gpu_stat:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=gpu_stat)

    test_real = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='test')

    with tf.compat.v1.variable_scope("generator", reuse=False):
        if 'lite' in checkpoint_dir:
            test_generated = generator_lite.G_net(test_real).fake
        else:
            test_generated = generator.G_net(test_real).fake

    # load video
    vid = cv2.VideoCapture(video)
    vid_name = os.path.basename(video)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    # codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    codec = cv2.VideoWriter_fourcc(*output_format)

    tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.compat.v1.Session(config=tfconfig) as sess:
        # tf.global_variables_initializer().run()
        # load model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information
        saver = tf.compat.v1.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
        else:
            print(" [*] Failed to find a checkpoint")
            return

        # determine output width and height
        ret, img = vid.read()
        if img is None:
            print('Error! Failed to determine frame size: frame empty.')
            return
        img = preprocessing(img, img_size)
        height, width = img.shape[:2]
        # out = cv2.VideoWriter(os.path.join(output, vid_name.replace('mp4','mkv')), codec, fps, (width, height))
        out = cv2.VideoWriter(os.path.join(output, vid_name), codec, fps, (width, height))

        pbar = tqdm(total=total)
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while ret:
            ret, frame = vid.read()
            if frame is None:
                print('Warning: got empty frame.')
                continue

            img = convert_image(frame, img_size)
            fake_img = sess.run(test_generated, feed_dict={test_real: img})
            fake_img = inverse_image(fake_img)
            fake_img = adjust_brightness_from_src_to_dst(fake_img, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out.write(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB))
            pbar.update(1)

        pbar.close()
        vid.release()
        # cv2.destroyAllWindows()
        return os.path.join(output, vid_name)


if __name__ == '__main__':
    arg = parse_args()
    if not arg.video:
        arg.video = getfileloc(initialdir='input/')
    else:
        arg.video = os.path.join(os.path.dirname(os.path.dirname(__file__)), arg.video)
    if not arg.output:
        arg.output = getfileloc(initialdir='output/', method='save')
    else:
        arg.output = os.path.join(os.path.dirname(os.path.dirname(__file__)), arg.output)
    check_folder(arg.output)
    info = cvt2anime_video(arg.video, arg.output, arg.checkpoint_dir, output_format=arg.output_format)
    print(f'output video: {info}')