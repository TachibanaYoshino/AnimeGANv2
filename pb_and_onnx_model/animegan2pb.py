# -*- coding: utf-8 -*-
# @Time    : 2021/8/31 18:51
# @Author  : Xin Chen
# @File    : animegan2pb.py
# @Software: PyCharm

import os
import tensorflow as tf
from tensorflow.python.framework import graph_util

def freeze_graph(model_folder, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB save dir
    :return:
    '''
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    print(checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path

    # input node and output node from the network ( AnimeGANv2 generator)
    # input_op = 'generator_input:0'
    # output_op = 'generator/G_MODEL/out_layer/Tanh:0'

    output_node_names = "generator/G_MODEL/out_layer/Tanh"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,  # :sess.graph_def
            output_node_names=output_node_names.split(","))

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        for op in graph.get_operations():
            print(op.name, op.values())

if __name__ == '__main__':
    model_folder = "/media/ada/0009B35A000DC852/a2/checkpoint/generator_Shinkai_weight"
    pb_save_path = "Shinkai_53.pb"
    freeze_graph(model_folder, pb_save_path)


    """ pb model 2 onnx command"""
    cmd = "python -m tf2onnx.convert --input Shinkai_53.pb --inputs generator_input:0  --outputs generator/G_MODEL/out_layer/Tanh:0  --output Shinkai_53.onnx"
    res = os.system(cmd)
    print(res)
