#!/usr/bin/env python
# Reference: https://github.com/IBM/tensorflow-hangul-recognition
import argparse
import io
import os
import pickle
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default paths.
DEFAULT_NUM_CLASS = 4 
DEFAULT_TFRECORDS_DIR = os.path.join(SCRIPT_PATH, 'tfrecords-output')
DEFAULT_SAVE_NAME = 'saved-model'
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, DEFAULT_SAVE_NAME)

MODEL_NAME = 'ealc_tensorflow'
DEFAULT_IMAGE_SIZE = 128
#IMAGE_WIDTH = DEFAULT_IMAGE_SIZE
#IMAGE_HEIGHT = DEFAULT_IMAGE_SIZE

DEFAULT_NUM_TRAIN_STEPS = 2000
DEFAULT_PRINT_STEPS = 100
DEFAULT_EVAL_STEPS = 200
BATCH_SIZE = 100

def get_image(files, num_classes, image_size):
    """This method defines the retrieval image examples from TFRecords files.

    Here we will define how the images will be represented (grayscale,
    flattened, floating point arrays) and how labels will be represented
    (one-hot vectors).
    """

    # Convert filenames to a queue for an input pipeline.
    file_queue = tf.train.string_input_producer(files)

    # Create object to read TFRecords.
    reader = tf.TFRecordReader()

    # Read the full set of features for a single example.
    key, example = reader.read(file_queue)

    # Parse the example to get a dict mapping feature keys to tensors.
    # image/class/label: integer denoting the index in a classification layer.
    # image/encoded: string containing JPEG encoded image
    features = tf.parse_single_example(
        example,
        features={
            'data/label': tf.FixedLenFeature([], dtype=tf.int64),
            'data/image': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })
    label = features['data/label']
    image_encoded = features['data/image']
    
    # Decode the PNG.
    image = tf.image.decode_png(image_encoded, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [image_size*image_size])
    
    # Represent the label as a one hot vector.
    label = tf.stack(tf.one_hot(label, num_classes))
    
    return label, image


def export_model(model_output_dir, input_node_names, output_node_name, step):
    """Export the model so we can use it later.

    This will create two Protocol Buffer files in the model output directory.
    These files represent a serialized version of our model with all the
    learned weights and biases. One of the ProtoBuf files is a version
    optimized for inference-only usage.
    """

    name_base = os.path.join(model_output_dir, MODEL_NAME)
    frozen_graph_file = os.path.join(model_output_dir,
                                     'frozen_' + MODEL_NAME + '.pb')
    freeze_graph.freeze_graph(
        name_base + '.pbtxt', None, False, name_base + '.chkp',
        output_node_name, "save/restore_all", "save/Const:0",
        frozen_graph_file, True, ""
    )

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(frozen_graph_file, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    optimized_graph_file = os.path.join(model_output_dir,
                                        'optimized_' + MODEL_NAME + '_' + str(step) + '.pb')
    with tf.gfile.FastGFile(optimized_graph_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Inference optimized graph saved at: " + optimized_graph_file)

class ModelLog():
    """Log the training and test results."""
    def __init__(self):
        self.print_steps = 100
        self.train_loss = []
        self.train_accu = []        
        self.train_confuMat = []
        self.test_accu = []        
        self.test_confuMat = []
        self.test_misLabeled = []
        self.train_overall_accu = []
        self.train_overall_confuMat = []
        self.train_overall_misLabeled = []
        self.print_steps = 0

def weight_variable(shape):
    """Generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weight')


def bias_variable(shape):
    """Generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')
    
def print_confuMat(confuMat):
    for row in confuMat:
        toPrint = ''
        for col in row:
            toPrint += "{:0.3f}".format(col) + "\t"
        print(toPrint)       

def evaluate_train_data(sess,x,y,y_,keep_prob,train_data_files, bath_size, correct_prediction, num_classes, image_batch, label_batch):     
    # Calculate the overall training accuracy and confusion matrix    
    # 1. Get number of samples in training set.
    sample_count = 0
    for f in train_data_files:
        sample_count += sum(1 for _ in tf.python_io.tf_record_iterator(f))
    # 2. See how model did by running the training set through the model.
    print('Evaluating the whole train set...')
    # 3. We will run the train set through batches and sum the total number of correct predictions.
    num_batches = int(sample_count/bath_size) or 1
    total_correct_preds = 0
    # 4. Define a different tensor operation for summing the correct predictions.
    accuracy2 = tf.reduce_sum(correct_prediction)
    accu_confusionMat = np.zeros((num_classes,num_classes))
    for s in range(num_batches):
        # 4A. Accuracy for Train Set
        image_batch2, label_batch2 = sess.run([image_batch, label_batch])
        acc = sess.run(accuracy2, feed_dict={x: image_batch2,
                                             y_: label_batch2,
                                             keep_prob: 1.0})
        total_correct_preds += acc
        # 4B. Confusion Matrix for Train Set
        train_labels = label_batch2
        train_predictions = sess.run(y, feed_dict={x: image_batch2, y_: label_batch2, keep_prob: 1.0})
        nature_labels =   np.argmax(train_labels, axis=1)
        nature_predicts = np.argmax(train_predictions, axis=1)
        curr_confusionMat = tf.confusion_matrix(labels=nature_labels, predictions=nature_predicts, num_classes=num_classes)
        accu_confusionMat += sess.run(curr_confusionMat)           
           
    # 5. Record the overall accuracy
    accuracy_percent = total_correct_preds/(num_batches*bath_size)
    print("Overall Training Accuracy {}".format(accuracy_percent))        
    # 6. Record the overall confusion matrix
    accu_confusionMat = accu_confusionMat/np.sum(accu_confusionMat,axis=1,keepdims=True)
    print('Overall Training Confusion Matrix:')
    print_confuMat(accu_confusionMat)
    return (accuracy_percent, accu_confusionMat)     

def evaluate_test_data(sess,x,y,y_,keep_prob,test_data_files, bath_size, correct_prediction, num_classes, timage_batch, tlabel_batch, save_mislabeled=False):
    # 1. Get number of samples in test set.
    sample_count = 0
    for f in test_data_files:
        sample_count += sum(1 for _ in tf.python_io.tf_record_iterator(f))
    # 2. See how model did by running the testing set through the model.
    print('Testing model by test set...')
    # 3. We will run the test set through batches and sum the total number of correct predictions.
    num_batches = int(sample_count/bath_size) or 1
    total_correct_preds = 0
    # 4. Define a different tensor operation for summing the correct predictions.
    accuracy2 = tf.reduce_sum(correct_prediction)
    accu_confusionMat = np.zeros((num_classes,num_classes))
    mislabeled = []
    for s in range(num_batches):
        if s%500==0:
            print("{:%I:%M %p} Status: ".format(datetime.datetime.today()) + str(s) + " / " + str(num_batches))
        # 4.a Accuracy for Test Set
        image_batch2, label_batch2 = sess.run([timage_batch, tlabel_batch])
        acc = sess.run(accuracy2, feed_dict={x: image_batch2,
                                             y_: label_batch2,
                                             keep_prob: 1.0})
        total_correct_preds += acc   

        
        # 4.b Confusion Matrix for Test Set
        test_labels = label_batch2
        test_predictions = sess.run(y, feed_dict={x: image_batch2, y_: label_batch2, keep_prob: 1.0})
        nature_labels =   tf.argmax(test_labels, axis=1)
        nature_predicts = tf.argmax(test_predictions, axis=1)
        curr_confusionMat = tf.confusion_matrix(labels=nature_labels, predictions=nature_predicts, num_classes=num_classes)
        accu_confusionMat += sess.run(curr_confusionMat)      
        # 4.c record the mislabeled predictions to "mislabeled"            
        #if save_mislabeled:
        #    for i in range(len(nature_labels)):
        #        mislabeled.append((image_batch2[i],nature_labels[i],nature_predicts[i]))   # log              
    
    accuracy_percent = total_correct_preds/(num_batches*bath_size)
    print("Testing Accuracy {}".format(accuracy_percent))
    print('Confusion Matrix:')
    accu_confusionMat = accu_confusionMat/np.sum(accu_confusionMat,axis=1,keepdims=True)
    print_confuMat(accu_confusionMat)    
    # 5. record the results of testing to trainLog and save mislabeled
    #if save_mislabeled:
    #    logName = "Mislabeled{:%m%d%H%M}".format(datetime.datetime.today())    
    #    with open(os.path.join(DEFAULT_SAVE_NAME, logName+'.pickle'), 'wb') as handle:
    #        pickle.dump(mislabeled, handle, protocol=pickle.HIGHEST_PROTOCOL) # save mislabeled
    return (accuracy_percent, accu_confusionMat)
 
 
def main(tfrecords_dir, model_output_dir, num_train_steps, bath_size, print_steps, evaluate_steps, image_size):
    """Perform graph definition and model training.
    Here we will first create our input pipeline for reading in TFRecords
    files and producing random batches of images and labels.
    Next, a convolutional neural network is defined, and training is performed.
    After training, the model is exported to be used in applications.
    """
    num_classes = DEFAULT_NUM_CLASS
    np.set_printoptions(precision=3)

    # Use GPU
    with tf.device('/gpu:0'):
        pass   
        
    # Define names so we can later reference specific nodes for when we use
    # the model for inference later.
    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'
    correct_prediction_node_name = 'correct_prediction'

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    
    # #############################################
    # Read the Data!                              #
    # #############################################
    print('Processing data...')

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'train')
    train_data_files = tf.gfile.Glob(tf_record_pattern)
    label, image = get_image(train_data_files, num_classes, image_size)

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'test')
    test_data_files = tf.gfile.Glob(tf_record_pattern)
    tlabel, timage = get_image(test_data_files, num_classes, image_size)

    # Associate objects with a randomly selected batch of labels and images.
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=bath_size,
        capacity=2000, min_after_dequeue=4)
        
    # Do the same for the testing data.
    timage_batch, tlabel_batch = tf.train.batch(
        [timage, tlabel], batch_size=bath_size,
        capacity=2000)
    
    # #############################################
    # Create the model!                           #
    # #############################################
    # Placeholder to feed in image data.
    x = tf.placeholder(tf.float32, [None, image_size*image_size],
                       name=input_node_name)
    # Placeholder to feed in label data. Labels are represented as one_hot
    # vectors.
    y_ = tf.placeholder(tf.float32, [None, num_classes],'y_label')  #, name=label_node_name

    # Reshape the image back into two dimensions so we can perform convolution.
    x_image = tf.reshape(x, [-1, image_size, image_size, 1])

    # First convolutional layer. 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv1 = tf.nn.relu(x_conv1 + b_conv1)

    # Max-pooling.
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
                             
    # Second convolutional layer. 64 feature maps.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    x_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv2 = tf.nn.relu(x_conv2 + b_conv2)
    
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
                             
    # Third convolutional layer. 128 feature maps.
    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    x_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv3 = tf.nn.relu(x_conv3 + b_conv3)
    
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
                             
    # Fully connected layer. Here we choose to have 1024 neurons in this layer.
    h_pool_flat = tf.reshape(h_pool3, [-1, int(image_size/8*image_size/8*128)])
    W_fc1 = weight_variable([int(image_size/8*image_size/8*128), 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    # Dropout layer. This helps fight overfitting.
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Classification layer.
    W_fc2 = weight_variable([1024, num_classes]) 
    b_fc2 = bias_variable([num_classes]) 
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    # This isn't used for training, but for when using the saved model.
    y2 = tf.nn.softmax(y, name=output_node_name)

    # Define our loss.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
    )

    # Define our optimizer for minimizing our loss. Here we choose a learning
    # rate of 0.0001 with AdamOptimizer. This utilizes someting
    # called the Adam algorithm, and utilizes adaptive learning rates and
    # momentum to get past saddle points.
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    # Define accuracy.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32, name=correct_prediction_node_name)
    accuracy = tf.reduce_mean(correct_prediction)
    saver = tf.train.Saver()
    
    # initialize the Model Log
    trainLog = ModelLog()
    trainLog.print_steps = print_steps

    with tf.Session() as sess: 
        # #############################################
        # Training Steps!                             #
        # #############################################
        
        # Initialize the variables.
        sess.run(tf.global_variables_initializer())

        # Initialize the queue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        checkpoint_file = os.path.join(model_output_dir, MODEL_NAME + '.chkp')

        # Save the graph definition to a file.
        tf.train.write_graph(sess.graph_def, model_output_dir,
                             MODEL_NAME + '.pbtxt', True)
        
        for step in range(num_train_steps):
            # Get a random batch of images and labels.
            train_images, train_labels = sess.run([image_batch, label_batch])
            
            # Perform the training step, feeding in the batches.
            sess.run(train_step, feed_dict={x: train_images, y_: train_labels,
                                            keep_prob: 0.5})
            
            # Every 100 iterations, we print the training accuracy.
            if step % print_steps == 0:
                train_accuracy, train_loss = sess.run(
                    [accuracy, cross_entropy],
                    feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0}
                )
                print("Step %d, Training Accuracy %g" %
                      (step, float(train_accuracy)))
                
                # Print the confusion matrix 
                train_predictions = sess.run(y, feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
                nature_labels =   np.argmax(train_labels, axis=1)
                nature_predicts = np.argmax(train_predictions, axis=1)
                curr_confusionMat = tf.confusion_matrix(labels=nature_labels, predictions=nature_predicts, num_classes=num_classes)
                curr_confusionMat = sess.run(curr_confusionMat)
                print('Confusion Matrix:')
                print_confuMat(curr_confusionMat/np.sum(curr_confusionMat,axis=1,keepdims=True))
                # record the results of this training step to trainLog
                trainLog.train_accu.append(train_accuracy)          # log
                trainLog.train_loss.append(train_loss)              # log
                trainLog.train_confuMat = curr_confusionMat         # log
                
                # Save the .pickle log in case the program carshes early. 
                logName = "printLog{:%m%d}".format(datetime.datetime.today())
                with open(os.path.join('saved-model',logName+'.pickle'), 'wb') as handle:
                    pickle.dump(trainLog, handle, protocol=pickle.HIGHEST_PROTOCOL) # save trainLog 
            
            # Every 5000 iterations, we evalue train set and save a model.
            if step % evaluate_steps == 0 and step>0:
                print('--------- Eval step ---------')
                # save model
                saver.save(sess, checkpoint_file)
                export_model(model_output_dir, [input_node_name, keep_prob_node_name],
                    correct_prediction_node_name, step)
                # save pickles
                # accuracy_percent, accu_confusionMat =  evaluate_test_data(sess,x,y,y_,keep_prob,
                #                                 test_data_files, bath_size, correct_prediction, 
                #                                 num_classes, timage_batch, tlabel_batch, False)
                # trainLog.test_accu =     accuracy_percent    # log
                # trainLog.test_confuMat=  accu_confusionMat   # log                
                # logName = "evaluateLog{:%m%d}".format(datetime.datetime.today()) + "_step{:06}".format(step)
                # with open(os.path.join(DEFAULT_SAVE_NAME, logName+'.pickle'), 'wb') as handle:
                #     pickle.dump(trainLog, handle, protocol=pickle.HIGHEST_PROTOCOL) # save temporary trainLog with test set
                print('-----------------------------')    
            
            # Every 10,000 iterations, we save a checkpoint of the model.
            #if step % 10000 == 0 and step>0:
            #    saver.save(sess, checkpoint_file, global_step=step)
            

        # Save a checkpoint after training has completed.
        saver.save(sess, checkpoint_file)
        #export_model(model_output_dir, [input_node_name, keep_prob_node_name],
        #             output_node_name, 'Final')
        export_model(model_output_dir, [input_node_name, keep_prob_node_name],
                     correct_prediction_node_name, 'Final')             
        print('The model has been exported to a .pb file!')
        
        print('========= Training Report =========') 
        # #############################################
        # Evaluate the whole training set!          #
        # #############################################
        accuracy_percent, accu_confusionMat = evaluate_train_data(sess,x,y,y_,keep_prob,
                                                train_data_files, bath_size, correct_prediction, 
                                                num_classes, image_batch, label_batch)
        trainLog.train_overall_accu = accuracy_percent        # log        
        trainLog.train_overall_confuMat = accu_confusionMat   # log
        
        
        # #############################################
        # Evaluate Testing Set!                       #
        # #############################################  
        accuracy_percent, accu_confusionMat =  evaluate_test_data(sess,x,y,y_,keep_prob,
                                                test_data_files, bath_size, correct_prediction, 
                                                num_classes, timage_batch, tlabel_batch, True)       
        trainLog.test_accu =     accuracy_percent    # log
        trainLog.test_confuMat=  accu_confusionMat   # log
        logName = "Final_Log{:%m%d%H%M}".format(datetime.datetime.today())
        with open(os.path.join(DEFAULT_SAVE_NAME, logName+'.pickle'), 'wb') as handle:
            pickle.dump(trainLog, handle, protocol=pickle.HIGHEST_PROTOCOL) # save trainLog
        print('The result (and log) of training is saved!')  
        print('===================================')
        
        # Stop queue threads and close session.
        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords-dir', type=str, dest='tfrecords_dir',
                        default=DEFAULT_TFRECORDS_DIR,
                        help='Directory of TFRecords files.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store saved model files.')
    parser.add_argument('--num-train-steps', type=int, dest='num_train_steps',
                        default=DEFAULT_NUM_TRAIN_STEPS,
                        help='Number of training steps to perform. This value '
                             'should be increased with more data. The number '
                             'of steps should cover several iterations over '
                             'all of the training data (epochs). Example: If '
                             'you have 15000 images in the training set, one '
                             'epoch would be 15000/100 = 150 steps where 100 '
                             'is the batch size. So, for 10 epochs, you would '
                             'put 150*10 = 1500 steps.')
    parser.add_argument('--batch-size', type=str, dest='bath_size',
                        default=BATCH_SIZE,
                        help='batch size.')
    parser.add_argument('--print-steps', type=str, dest='print_steps',
                        default=DEFAULT_PRINT_STEPS,
                        help='Print the accuracy every other this number of steps.')                    
    parser.add_argument('--eval-steps', type=str, dest='evaluate_steps',
                        default=DEFAULT_EVAL_STEPS,
                        help='Evaluate test set every other this number of steps.')
    parser.add_argument('--image-size', type=str, dest='image_size',
                        default=DEFAULT_IMAGE_SIZE,
                        help='Image size assuming images are square.')                    
                        
    args = parser.parse_args()
    
    main(args.tfrecords_dir, args.output_dir, int(args.num_train_steps), int(args.bath_size),
                int(args.print_steps), int(args.evaluate_steps), int(args.image_size))
