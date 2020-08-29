
from tools.ops import *

def D_net(x_init,ch, n_dis,sn, scope, reuse):
    channel = ch // 2
    with tf.variable_scope(scope, reuse=reuse):
        x = conv(x_init, channel, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='conv_0')
        x = lrelu(x, 0.2)

        for i in range(1, n_dis):
            x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=False, sn=sn, scope='conv_s2_' + str(i))
            x = lrelu(x, 0.2)

            x = conv(x, channel * 4, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='conv_s1_' + str(i))
            x = layer_norm(x, scope='1_norm_' + str(i))
            x = lrelu(x, 0.2)

            channel = channel * 2

        x = conv(x, channel * 2, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='last_conv')
        x = layer_norm(x, scope='2_ins_norm')
        x = lrelu(x, 0.2)

        x = conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='D_logit')

        return x

