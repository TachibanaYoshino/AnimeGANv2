import os
import tensorflow as tf
import cv2
import numpy as np
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    desc = "AnimeGANv2 for pb"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--pb_path', type=str, default='pb_model_Hayao-64',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--img_path', type=str, default='dataset/test/HR_photo/21.jpg',
                        help='path of test photo')
    parser.add_argument('--out_path', type=str, default='pb_demo_results',
                        help='what style you want to get')

    return parser.parse_args()

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
    return img/127.5 - 1.0

def load_test_data(image_path, size=[256,256]):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img,size)
    img = np.expand_dims(img, axis=0)
    return img

def img_save(img, path):
    """
    Brightness adjustment is not used here. If you feel necessary, you can add it yourself like test.py
    """
    image = (img.squeeze() + 1.) / 2 * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

signature_key = 'custom_signature'
input_key = 'input'
output_key = 'output'

if __name__ =='__main__':

    arg = parse_args()
    pb_path = arg.pb_path
    img_path = arg.img_path
    out_path = os.path.join(check_folder(arg.out_path), os.path.basename(img_path))

    test_img = load_test_data(img_path)

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        '''
        You can provide 'tags' when saving a model,
        in my case I provided, tf.saved_model.tag_constants.SERVING tag 
        '''

        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], pb_path)
        # get custom_signature from meta_graph_def
        signature = meta_graph_def.signature_def
        # Get alias of input and output
        x = signature[signature_key].inputs[input_key].name
        y = signature[signature_key].outputs[output_key].name

        graph = tf.get_default_graph()
        """ print your graph's ops, if needed """
        # i = 0
        # for op in graph.get_operations():
        #     print(i, ' : ', op.name, op.values())
        #     i += 1

        '''
        In my case, I named my input and output tensors as
        input:0 and output:0 respectively in convert_ckpt_to_savermodel.py
        '''
        input_ = sess.graph.get_tensor_by_name(x)
        output_ = sess.graph.get_tensor_by_name(y)

        #  if you know the input op(tensor) and output op(tensor), you can also make it like this:
        # input_ = sess.graph.get_tensor_by_name('generator_input:0')
        # output_ = sess.graph.get_tensor_by_name('generator/G_MODEL/out_layer/Tanh:0')

        y_pred = sess.run(output_, feed_dict={input_: test_img})
        img_save(y_pred,out_path)


