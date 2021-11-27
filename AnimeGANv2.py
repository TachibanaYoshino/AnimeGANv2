from tools.ops import *
from tools.utils import *
from glob import glob
import time
import numpy as np
from net import generator
from net.discriminator import D_net
from tools.data_loader import ImageGenerator
from tools.vgg19 import Vgg19

class AnimeGANv2(object) :
    def __init__(self, sess, args):
        self.model_name = 'AnimeGANv2'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset

        self.epoch = args.epoch
        self.init_epoch = args.init_epoch # args.epoch // 20

        self.gan_type = args.gan_type
        self.batch_size = args.batch_size
        self.save_freq = args.save_freq

        self.init_lr = args.init_lr
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr

        """ Weight """
        self.g_adv_weight = args.g_adv_weight
        self.d_adv_weight = args.d_adv_weight
        self.con_weight = args.con_weight
        self.sty_weight = args.sty_weight
        self.color_weight = args.color_weight
        self.tv_weight = args.tv_weight

        self.training_rate = args.training_rate
        self.ld = args.ld

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        """ Discriminator """
        self.n_dis = args.n_dis
        self.ch = args.ch
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.real = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='real_A')
        self.anime = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='anime_A')
        self.anime_smooth = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='anime_smooth_A')
        self.test_real = tf.placeholder(tf.float32, [1, None, None, self.img_ch], name='test_input')

        self.anime_gray = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch],name='anime_B')


        self.real_image_generator = ImageGenerator('./dataset/train_photo', self.img_size, self.batch_size)
        self.anime_image_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/style'), self.img_size, self.batch_size)
        self.anime_smooth_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/smooth'), self.img_size, self.batch_size)
        self.dataset_num = max(self.real_image_generator.num_images, self.anime_image_generator.num_images)

        self.vgg = Vgg19()

        print()
        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# init_epoch : ", self.init_epoch)
        print("# training image size [H, W] : ", self.img_size)
        print("# g_adv_weight,d_adv_weight,con_weight,sty_weight,color_weight,tv_weight : ", self.g_adv_weight,self.d_adv_weight,self.con_weight,self.sty_weight,self.color_weight,self.tv_weight)
        print("# init_lr,g_lr,d_lr : ", self.init_lr,self.g_lr,self.d_lr)
        print(f"# training_rate G -- D: {self.training_rate} : 1" )
        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_init, reuse=False, scope="generator"):
        with tf.variable_scope(scope, reuse=reuse):
            G = generator.G_net(x_init)
            return G.fake

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
            D = D_net(x_init, self.ch, self.n_dis, self.sn, reuse=reuse, scope=scope)
            return D

    ##################################################################################
    # Model
    ##################################################################################
    def gradient_panalty(self, real, fake, scope="discriminator"):
        if self.gan_type.__contains__('dragan') :
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logit, _= self.discriminator(interpolated, reuse=True, scope=scope)

        grad = tf.gradients(logit, interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

        GP = 0
        # WGAN - LP
        if self.gan_type.__contains__('lp'):
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type.__contains__('gp') or self.gan_type == 'dragan' :
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    def build_model(self):

        """ Define Generator, Discriminator """
        self.generated = self.generator(self.real)
        self.test_generated = self.generator(self.test_real, reuse=True)


        anime_logit = self.discriminator(self.anime)
        anime_gray_logit = self.discriminator(self.anime_gray, reuse=True)

        generated_logit = self.discriminator(self.generated, reuse=True)
        smooth_logit = self.discriminator(self.anime_smooth, reuse=True)

        """ Define Loss """
        if self.gan_type.__contains__('gp') or self.gan_type.__contains__('lp') or self.gan_type.__contains__('dragan') :
            GP = self.gradient_panalty(real=self.anime, fake=self.generated)
        else :
            GP = 0.0

        # init pharse
        init_c_loss = con_loss(self.vgg, self.real, self.generated)
        init_loss = self.con_weight * init_c_loss
        
        self.init_loss = init_loss

        # gan
        c_loss, s_loss = con_sty_loss(self.vgg, self.real, self.anime_gray, self.generated)
        tv_loss = self.tv_weight * total_variation_loss(self.generated)
        t_loss = self.con_weight * c_loss + self.sty_weight * s_loss + color_loss(self.real,self.generated) * self.color_weight + tv_loss

        g_loss = self.g_adv_weight * generator_loss(self.gan_type, generated_logit)
        d_loss = self.d_adv_weight * discriminator_loss(self.gan_type, anime_logit, anime_gray_logit, generated_logit, smooth_logit) + GP

        self.Generator_loss =  t_loss + g_loss
        self.Discriminator_loss = d_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        self.init_optim = tf.train.AdamOptimizer(self.init_lr, beta1=0.5, beta2=0.999).minimize(self.init_loss, var_list=G_vars)
        self.G_optim = tf.train.AdamOptimizer(self.g_lr , beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.d_lr , beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

        self.G_gan = tf.summary.scalar("G_gan", g_loss)
        self.G_vgg = tf.summary.scalar("G_vgg", t_loss)
        self.G_init_loss = tf.summary.scalar("G_init", init_loss)

        self.V_loss_merge = tf.summary.merge([self.G_init_loss])
        self.G_loss_merge = tf.summary.merge([self.G_loss, self.G_gan, self.G_vgg, self.G_init_loss])
        self.D_loss_merge = tf.summary.merge([self.D_loss])

    def train(self):
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=self.epoch)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        """ Input Image"""
        real_img_op, anime_img_op, anime_smooth_op  = self.real_image_generator.load_images(), self.anime_image_generator.load_images(), self.anime_smooth_generator.load_images()


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = checkpoint_counter + 1

            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0

            print(" [!] Load failed...")

        # loop for epoch
        init_mean_loss = []
        mean_loss = []
        # training times , G : D = self.training_rate : 1
        j = self.training_rate
        for epoch in range(start_epoch, self.epoch):
            for idx in range(int(self.dataset_num / self.batch_size)):
                anime, anime_smooth, real = self.sess.run([anime_img_op, anime_smooth_op, real_img_op])
                train_feed_dict = {
                    self.real:real[0],
                    self.anime:anime[0],
                    self.anime_gray:anime[1],
                    self.anime_smooth:anime_smooth[1]
                }

                if epoch < self.init_epoch :
                    # Init G
                    start_time = time.time()

                    real_images, generator_images, _, v_loss, summary_str = self.sess.run([self.real, self.generated,
                                                                             self.init_optim,
                                                                             self.init_loss, self.V_loss_merge], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, epoch)
                    init_mean_loss.append(v_loss)

                    print("Epoch: %3d Step: %5d / %5d  time: %f s init_v_loss: %.8f  mean_v_loss: %.8f" % (epoch, idx,int(self.dataset_num / self.batch_size), time.time() - start_time, v_loss, np.mean(init_mean_loss)))
                    if (idx+1)%200 ==0:
                        init_mean_loss.clear()
                else :
                    start_time = time.time()

                    if j == self.training_rate:
                        # Update D
                        _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss_merge],
                                                            feed_dict=train_feed_dict)
                        self.writer.add_summary(summary_str, epoch)

                    # Update G
                    real_images, generator_images, _, g_loss, summary_str = self.sess.run([self.real, self.generated,self.G_optim,
                                                                                              self.Generator_loss, self.G_loss_merge], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, epoch)

                    mean_loss.append([d_loss, g_loss])
                    if j == self.training_rate:

                        print(
                            "Epoch: %3d Step: %5d / %5d  time: %f s d_loss: %.8f, g_loss: %.8f -- mean_d_loss: %.8f, mean_g_loss: %.8f" % (
                                epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, d_loss, g_loss, np.mean(mean_loss, axis=0)[0],
                                np.mean(mean_loss, axis=0)[1]))
                    else:
                        print(
                            "Epoch: %3d Step: %5d / %5d time: %f s , g_loss: %.8f --  mean_g_loss: %.8f" % (
                                epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, g_loss, np.mean(mean_loss, axis=0)[1]))

                    if (idx + 1) % 200 == 0:
                        mean_loss.clear()

                    j = j - 1
                    if j < 1:
                        j = self.training_rate


            if (epoch + 1) >= self.init_epoch and np.mod(epoch + 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, epoch)

            if epoch >= self.init_epoch -1:
                """ Result Image """
                val_files = glob('./dataset/{}/*.*'.format('val'))
                save_path = './{}/{:03d}/'.format(self.sample_dir, epoch)
                check_folder(save_path)
                for i, sample_file in enumerate(val_files):
                    print('val: '+ str(i) + sample_file)
                    sample_image = np.asarray(load_test_data(sample_file, self.img_size))
                    test_real,test_generated = self.sess.run([self.test_real,self.test_generated],feed_dict = {self.test_real:sample_image} )
                    save_images(test_real, save_path+'{:03d}_a.jpg'.format(i), None)
                    save_images(test_generated, save_path+'{:03d}_b.jpg'.format(i), None)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                                          self.gan_type,
                                                          int(self.g_adv_weight), int(self.d_adv_weight),
                                                          int(self.con_weight), int(self.sty_weight),
                                                          int(self.color_weight), int(self.tv_weight))


    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir) # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path) # first line
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
