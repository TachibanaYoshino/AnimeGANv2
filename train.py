from AnimeGANv2 import AnimeGANv2
import argparse
from tools.utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""parsing and configuration"""

def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Hayao', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=101, help='The number of epochs to run')
    parser.add_argument('--init_epoch', type=int, default=10, help='The number of epochs for weight initialization')
    parser.add_argument('--batch_size', type=int, default=12, help='The size of batch size') # if light : batch_size = 20
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')

    parser.add_argument('--init_lr', type=float, default=2e-4, help='The learning rate')
    parser.add_argument('--g_lr', type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--d_lr', type=float, default=4e-5, help='The learning rate')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')

    parser.add_argument('--g_adv_weight', type=float, default=300.0, help='Weight about GAN')
    parser.add_argument('--d_adv_weight', type=float, default=300.0, help='Weight about GAN')
    parser.add_argument('--con_weight', type=float, default=1.5, help='Weight about VGG19')# 1.5 for Hayao, 2.0 for Paprika, 1.2 for Shinkai
    # ------ the follow weight used in AnimeGAN
    parser.add_argument('--sty_weight', type=float, default=2.5, help='Weight about style')# 2.5 for Hayao, 0.6 for Paprika, 2.0 for Shinkai
    parser.add_argument('--color_weight', type=float, default=10., help='Weight about color') # 15. for Hayao, 50. for Paprika, 10. for Shinkai
    parser.add_argument('--tv_weight', type=float, default=1., help='Weight about tv')# 1. for Hayao, 0.1 for Paprika, 1. for Shinkai
    # ---------------------------------------------
    parser.add_argument('--training_rate', type=int, default=1, help='training rate about G & D')
    parser.add_argument('--gan_type', type=str, default='lsgan', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge')

    parser.add_argument('--img_size', type=list, default=[256,256], help='The size of image: H and W')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')


    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --log_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,inter_op_parallelism_threads=8,
                               intra_op_parallelism_threads=8,gpu_options=gpu_options)) as sess:
        gan = AnimeGANv2(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        gan.train()
        print(" [*] Training finished!")


if __name__ == '__main__':
    main()
