import tensorflow as tf
import os,argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    desc = "AnimeGANv2 for pb"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--trained_checkpoint_prefix', type=str, default='checkpoint/' + 'generator_Hayao_weight/' + 'Hayao-64.ckpt',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--export_dir', type=str, default='pb_model_Hayao-64',
                        help='path of test photo')
    parser.add_argument('--out_path', type=str, default='pb_demo_results',
                        help='what style you want to get')

    return parser.parse_args()

if __name__ == '__main__':
    arg = parse_args()

    trained_checkpoint_prefix = arg.trained_checkpoint_prefix
    export_dir = arg.export_dir #  savedPath

    # input node and output node from the network
    input_op = 'generator_input:0'
    output_op = 'generator/G_MODEL/out_layer/Tanh:0'

    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session(graph=graph, config=config) as sess:
        # Restore from checkpoint
        loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
        loader.restore(sess, trained_checkpoint_prefix)

        # the input and output in the ckpt
        x = tf.get_default_graph().get_tensor_by_name(input_op)
        y = tf.get_default_graph().get_tensor_by_name(output_op)

        # Export checkpoint to SavedModel
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

        # custom settings of the input and output in the pb
        inputs = {'input': tf.saved_model.utils.build_tensor_info(x)}
        outputs = {'output': tf.saved_model.utils.build_tensor_info(y)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'AnimeGANv2')

        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],{'custom_signature':signature})
        builder.save()

        """
        This will save your protobuf ('saved_model.pb') in the said folder ('models' here) 
        which can then be loaded by Use_pb.py.
        the output file structure as below
        
        └── pb_model_Hayao-64
        ···├── saved_model.pb
        ···└── variables
        ·········├── variables.data-00000-of-00001
        ·········└── variables.index
        """



