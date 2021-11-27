import argparse
from tools.utils import *
import os
from net import generator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoint/' + 'AnimeGANv2_Hayao_lsgan_300_300_1_2_10_1',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--style_name', type=str, default='Hayao',
                        help='what style you want to get')

    return parser.parse_args()

def save(saver, sess, checkpoint_dir, model_name):
    save_path = os.path.join(checkpoint_dir, model_name + '.ckpt')
    saver.save(sess, save_path, write_meta_graph=True)
    return  save_path

def main(checkpoint_dir, style_name):
    ckpt_dir = '../checkpoint/' + 'generator_' + style_name + '_weight'
    check_folder(ckpt_dir)

    placeholder = tf.placeholder(tf.float32, [1, None, None, 3], name='generator_input')
    with tf.variable_scope("generator", reuse=False):
        _ = generator.G_net(placeholder).fake

    generator_var = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
    saver = tf.train.Saver(generator_var)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        # load model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = ckpt_name.split('-')[-1]
            print(" [*] Success to read {}".format(ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")
            return
        info = save(saver, sess, ckpt_dir, style_name+'-'+counter)

        print(f'save over : {info} ')



if __name__ == '__main__':
    arg = parse_args()
    print(arg.checkpoint_dir)
    main(arg.checkpoint_dir, arg.style_name)
