import tensorflow as tf
import os
from gray_centre_sample import *
import numpy as np
def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=name)

def get_trained_model(subfolder, global_step):
    model_path = './model/%s/my_model-%s.meta'%(subfolder,global_step)
    if not os.path.exists(model_path):
        print "No trained model! Please train firstly."
        return
    sess = tf.Session()
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, tf.train.latest_checkpoint('./model/%s/'%subfolder))
    graph = tf.get_default_graph()
    # Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("accuracy:0")
    x = graph.get_tensor_by_name("input_image:0")
    y_ = graph.get_tensor_by_name("input_label:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    pred_label = graph.get_tensor_by_name("pred_label:0")
    def predict(Input):
        feed = {x:Input, y_:np.zeros((81,9)), keep_prob:1.0} #watch
        prediction = pred_label.eval(session=sess, feed_dict=feed)
        return prediction
    return predict

if __name__ ==  "__main__":
    train_gray_cntre_model()
