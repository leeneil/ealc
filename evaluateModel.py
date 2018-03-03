import os
import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from numpy import random
from trainModel import get_image
from trainModel import evaluate_test_data
from trainModel import print_confuMat


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def evalModel(model_dir, model_name, tfrecords_dir, bath_size, num_classes, image_size):
    # params
    #bath_size = 25
    #num_classes = 4
    #image_size = 128     
        
    with tf.Session() as sess:
        ## Load Graph
        load_graph(os.path.join(model_dir , model_name))
        graph = sess.graph
        #print('The content of the graph')
        #for op in graph.get_operations():
        #    print(op.name)

        # Read graph nodes
        x = graph.get_tensor_by_name('input:0')
        y = graph.get_tensor_by_name('add_4:0')
        y_ = graph.get_tensor_by_name('y_label:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        correct_prediction = graph.get_tensor_by_name('correct_prediction:0')
            
        # Read Data
        tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'test')
        test_data_files = tf.gfile.Glob(tf_record_pattern)
        tlabel, timage = get_image(test_data_files, num_classes, image_size)

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
    return accuracy_percent, accu_confusionMat

def print_confuMat_to_file(file, confuMat):
    for row in confuMat:
        toPrint = ''
        for col in row:
            toPrint += "{:0.3f}".format(col) + "\t"
        file.write(toPrint)
        file.write("\n")   
    
def main(tfrecords_dir, model_dir, model_root, model_num ,bath_size, num_classes, image_size):
    # Use GPU
    with tf.device("/gpu:0"):
        pass       
            
    ##
    model_name = model_root + model_num + ".pb"
    print("========== Eval Model: " + os.path.join(model_dir, model_name)+" ==========")
    accuracy_percent, accu_confusionMat = evalModel(model_dir, model_name, tfrecords_dir, bath_size, num_classes, image_size)
    #print('accuracy_percent = ', accuracy_percent)
    #print(accu_confusionMat)
    #print_confuMat(accu_confusionMat)
    
    # write to file 
    file = open(os.path.join(model_dir, "evalManyModel"+ model_num +".txt"),"w")
    file.write("Model at the step:  " + model_num + "\n") 
    file.write("Test Set Accuracy: = {:0.5} \n".format(accuracy_percent))
    file.write("Confusion Matrix:  \n")
    print_confuMat_to_file(file, accu_confusionMat)
    file.write("\n")
    file.close()
    print("======================================================================")    
    
##   
#name_List = [str(i) for i in range(50000,200000,10000)] 
#name_List.append("Final")
# name_List = ['10', '30', 'Final']

## params
bath_size = 128
num_classes = 4
image_size = 128

##
model_dir = os.path.join('saved-model')
model_root = "optimized_ealc_tensorflow_"
tfrecords_dir = "tfrecords-output"
  
parser = argparse.ArgumentParser()
parser.add_argument('--tfrecords-dir', type=str, dest='tfrecords_dir',
                    default=tfrecords_dir)
parser.add_argument('--model-dir', type=str, dest='model_dir',
                    default=model_dir)
parser.add_argument('--model-root', type=str, dest='model_root',
                    default=model_root)
parser.add_argument('--model-num', type=str, dest='model_num',
                    default="100000")
parser.add_argument('--bath-size', type=str, dest='bath_size',
                    default=bath_size)
parser.add_argument('--num-class', type=str, dest='num_classes',
                    default=num_classes)
parser.add_argument('--image-size', type=str, dest='image_size',
                    default=image_size) 
args = parser.parse_args()
main(args.tfrecords_dir, args.model_dir, args.model_root, str(args.model_num), int(args.bath_size), int(args.num_classes), int(args.image_size))                    