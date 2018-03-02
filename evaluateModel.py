import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from numpy import random
from trainModel import get_image
from trainModel import evaluate_test_data
from trainModel import print_confuMat

import sys
import cv2

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

tfrecords_dir = 'tfrecords-output'
export_dir = 'saved-model'
model_name = 'optimized_ealc_tensorflow_Final.pb'


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

# params
bath_size = 50
num_classes = 4

with tf.Session() as sess:
    ## Load Graph
    load_graph(os.path.join(export_dir , model_name))
    graph = sess.graph
    print('The content of the graph')
    for op in graph.get_operations():
        print(op.name)

    # Read graph nodes
    x = graph.get_tensor_by_name('input:0')
    y = graph.get_tensor_by_name('add_4:0')
    y_ = graph.get_tensor_by_name('y_label:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    correct_prediction = graph.get_tensor_by_name('correct_prediction:0')
        
    # Read Data
    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'test')
    test_data_files = tf.gfile.Glob(tf_record_pattern)
    tlabel, timage = get_image(test_data_files, num_classes)

    # Do the same for the testing data.
    timage_batch, tlabel_batch = tf.train.batch(
    [timage, tlabel], batch_size=bath_size,
            capacity=2000)
        
    # get accuracy and confusion matrix
    # adding these 2 lines fixed the hang forever problem
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    accuracy_percent, accu_confusionMat =  evaluate_test_data(sess,x,y,y_,keep_prob,
                                                test_data_files, bath_size, correct_prediction, 
                                                num_classes, timage_batch, tlabel_batch, False) 
    # Stop queue threads and close session.
    coord.request_stop()
    coord.join(threads)
    sess.close()
# python evaluateModel.py    
