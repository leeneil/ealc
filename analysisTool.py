import pickle
import os
import glob

import tensorflow as tf
import sys
from tensorflow.python.platform import gfile
from trainModel import ModelLog
''' This code load the log of training. The log is saved as a .pickle file.  
variables in log
train_loss:      array of float
train_accu:      array of float
train_confuMat:  array of confusion matrix
test_accu:       array of float
test_confuMat:   array of confusion matrix
test_misLabeled: array of (image, correct label, prediction label)
'''

def get_pickle():
    # Get the "lastest" pickle file
    dir_pkl = 'saved-model'
    file_pkl = glob.glob(os.path.join(dir_pkl, '*.pickle'))
    file_pkl.sort(key=os.path.getmtime)
    filename = file_pkl[-1]
    return filename

# [It can be used outside]
def get_training_log(filename = get_pickle()):
    # Open the pickle file, output as a dictionary
    with open(filename, 'rb') as handle:
        log = pickle.load(handle)
        print('Opened the pickle file: ',filename)
    return log.__dict__
       


def create_graph(modelFullPath):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

# [It can be used outside]
def get_tfModel_params(filepath):
    """ Get the weigths, bias, filters in the neural network """
    # GRAPH_DIR = os.path.join('saved-model','frozen_ealc_tensorflow.pb')
    create_graph(filepath)
    constant_values = {}
    with tf.Session() as sess:
      constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
      for constant_op in constant_ops:
        constant_values[constant_op.name] = sess.run(constant_op.outputs[0])       
    return constant_values

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')