import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from net import generator

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


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def process_image(img, x32=True):
    h, w = img.shape[:2]
    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
    return img

def post_precess(img, wh):
    img = (img.squeeze()+1.) / 2 * 255
    img = img.astype(np.uint8)
    img = cv2.resize(img, (wh[0], wh[1]))
    return img

def cvt2anime_video(video, output, checkpoint_dir, output_format='MP4V'):
    '''
    output_format: 4-letter code that specify codec to use for specific video type. e.g. for mp4 support use "H264", "MP4V", or "X264"
    '''
    gpu_stat = bool(len(tf.config.experimental.list_physical_devices('GPU')))
    if gpu_stat:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tf.GPUOptions(allow_growth=gpu_stat)

    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')
    with tf.variable_scope("generator", reuse=False):
        test_generated = generator.G_net(test_real).fake
         
    saver = tf.train.Saver()

    # load video
    vid = cv2.VideoCapture(video)
    vid_name = os.path.basename(video)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*output_format)

    tfconfig = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=tfconfig) as sess:
        # tf.global_variables_initializer().run()
        # load model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
        else:
            print(" [*] Failed to find a checkpoint")
            return
         
        video_out = cv2.VideoWriter(os.path.join(output, vid_name.rsplit('.', 1)[0] + "_AnimeGANv2.mp4"), codec, fps, (width, height))

        pbar = tqdm(total=total, ncols=80)
        pbar.set_description(f"Making: {os.path.basename(video).rsplit('.', 1)[0] + '_AnimeGANv2.mp4'}")
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            frame = np.asarray(np.expand_dims(process_image(frame),0))
            fake_img = sess.run(test_generated, feed_dict={test_real: frame})
            fake_img = post_precess(fake_img, (width, height))
            video_out.write(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB))
            pbar.update(1)

        pbar.close()
        vid.release()
        video_out.release()
        return os.path.join(output, vid_name.rsplit('.', 1)[0] + "_AnimeGANv2.mp4")

if __name__ == '__main__':
    arg = parse_args()
    check_folder(arg.output)
    info = cvt2anime_video(arg.video, arg.output, arg.checkpoint_dir)
    print(f'output video: {info}')
