import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from numpy import random

import sys
import cv2

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

export_dir = '/Users/liponan/Work/ml/ealc/saved-model/'
model_name = 'chkp_ealc_tensorflow.pb'

vectorize = True

import tensorflow as tf

if len(sys.argv) < 2:
    raise('please specify an input image')


layer = 'weight'

print_z = True
print_a = True
print_filters = False

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

graph = load_graph(export_dir + model_name)

print(graph)
for op in graph.get_operations():
    print(op.name)

img = cv2.imread( sys.argv[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
if vectorize:
    img = np.reshape(img, (-1, 128*128))

print('shape of img')
print(img.shape)

w = graph.get_tensor_by_name('prefix/' + layer + ':0')
x = graph.get_tensor_by_name('prefix/input:0')
z = graph.get_tensor_by_name('prefix/Conv2D:0')
a = graph.get_tensor_by_name('prefix/MaxPool:0')
# z = graph.get_tensor_by_name('prefix/Conv2D_2:0')
# a = graph.get_tensor_by_name('prefix/MaxPool_2:0')
y = graph.get_tensor_by_name('prefix/output:0')
keep_prob = graph.get_tensor_by_name('prefix/keep_prob:0')

with tf.Session(graph=graph) as sess:
    # weights
    w_out = w.eval()
    print(w_out.shape)

    np.save('learned/' + layer + '.npy', w_out)
    # input
    z_out, a_out, y_out = sess.run([z,a,y], feed_dict = {x:img, keep_prob: 1})
    # a_out = sess.run(a, feed_dict = {x:img})
    # y_out = sess.run(y, feed_dict = {x:img})
    print('shape of z_out:')
    print(z_out.shape)
    print('shape of a_out:')
    print(a_out.shape)


if print_z:
    z_out = np.reshape( z_out, (z_out.shape[1], z_out.shape[2], z_out.shape[3], 1) )
    print('shape of z_out:')
    print(z_out.shape)
    var = np.std( z_out )

    for u in range(z_out.shape[2]):
        im0 = plt.imshow(z_out[:,:,u,0], vmin=-var, vmax=var, cmap=cm.coolwarm)
        ax0 = plt.gca()
        plt.axis('off')
        plt.colorbar(im0)
        #cbar0 = plt.colorbar(im0, ticks=[9.95, 10.00, 10.05])
        #cbar0.ax.set_yticklabels(['<9.95', '10', '10.05'])

        #plt.show()
        plt.savefig('learned/zout_' + str(u) + '.png', bbox_inches='tight', dpi=120)
        # plt.savefig('conc_profile_neutral.pdf', bbox_inches='tight', dpi=600)
        plt.clf()
        plt.cla()

if print_a:
    a_out = np.reshape( a_out, (a_out.shape[1], a_out.shape[2], a_out.shape[3], 1) )
    print('shape of a_out:')
    print(a_out.shape)
    var = np.std( a_out )
    vmax = np.max( a_out )

    for u in range(a_out.shape[2]):
        im0 = plt.imshow(a_out[:,:,u,0], vmin=0, vmax= vmax, cmap=cm.gray)
        ax0 = plt.gca()
        plt.axis('off')
        plt.colorbar(im0)
        #cbar0 = plt.colorbar(im0, ticks=[9.95, 10.00, 10.05])
        #cbar0.ax.set_yticklabels(['<9.95', '10', '10.05'])

        #plt.show()
        plt.savefig('learned/aout_' + str(u) + '.png', bbox_inches='tight', dpi=120)
        # plt.savefig('conc_profile_neutral.pdf', bbox_inches='tight', dpi=600)
        plt.clf()
        plt.cla()

if print_filters:
    var = np.std( w_out )

    for u in range(w_out.shape[2]):
        for v in range(w_out.shape[3]):
            im0 = plt.imshow(w_out[:,:,u,v], vmin=-var, vmax=var, cmap=cm.coolwarm)
            ax0 = plt.gca()
            plt.axis('off')
            plt.colorbar(im0)
            #cbar0 = plt.colorbar(im0, ticks=[9.95, 10.00, 10.05])
            #cbar0.ax.set_yticklabels(['<9.95', '10', '10.05'])

            #plt.show()
            plt.savefig('learned/filters/' + layer + '_' + str(u) + '_' + str(v) + '.png', bbox_inches='tight', dpi=120)
            # plt.savefig('conc_profile_neutral.pdf', bbox_inches='tight', dpi=600)
            plt.clf()
            plt.cla()


print('prediction:')
print(y_out)
