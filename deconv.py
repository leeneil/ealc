import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from numpy import random
from numpy import fft
from scipy import fftpack
from scipy import signal
from scipy import misc

import sys
import cv2

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

depth = 3
print_layers = False
export_dir = 'saved-model/'
model_name = 'optimized_ealc_tensorflow_Final.pb'

vectorize = True

import tensorflow as tf

prefix = 'test'

if len(sys.argv) < 2:
    raise('please specify an input image')
if len(sys.argv) > 2:
    prefix = sys.argv[2]


layer = 'weight'


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

def maxpool(a, k=2, s=2):
    new_w = int( (a.shape[1] - k) / s + 1 )
    new_h = int( (a.shape[0] - k) / s + 1 )
    l = a.shape[2]

    output = np.zeros( (new_h, new_w, l, k*k) )
    for u in range(new_h):
        for v in range(new_w):
            output[u,v,:,:] = np.reshape( np.reshape(a[s*u:(s*u+k), s*v:(s*v+k), :], (k*k,l)).T, (1,1,l,k*k) )
    amax = np.argmax( output, axis=3)
    output = np.max( output, axis=3)

    return output, amax

def max_activation( maxp ):
    macact = maxp.copy()
    layer_max = np.max( np.max(maxp, axis=1), axis=0 )
    for u in range(maxp.shape[2]):
        macact[ maxp[:,:,u] != layer_max[u], u] = 0
    return macact

def unmaxpool( maxp, amax, k=2, s=2 ):
    new_w = s*(maxp.shape[1]-1) + k
    new_h = s*(maxp.shape[0]-1) + k
    a = np.zeros( (new_h, new_w, maxp.shape[2]) )
    for u in range(maxp.shape[0]):
        for v in range(maxp.shape[1]):
            for w in range(maxp.shape[2]):
                if maxp[u,v,w] == 0:
                    pass
                else:
                    vv = amax[u,v,w] % k
                    uu = int( (amax[u,v,w]-k) / k )
                    a[u*s+uu,v*s+vv,w] = maxp[u,v,w]
    return a

def deconv2( g, h, b ):
    l2 = h.shape[2]
    l3 = h.shape[3]
    f = np.zeros( (g.shape[0], g.shape[1], l2) )
    for u in range(l2):
        for v in range(l3):
            f[:,:,u] += signal.convolve2d( (g[:,:,v]-b[v]), np.rot90(h[:,:,u,v],0), \
                mode='same', boundary='wrap' )
            # g_fft = np.fft.fft2( g[:,:,u] )
            # h_fft = np.fft.fft2( np.rot90(h[:,:,u], 2) )
            # f[:,:,u] = np.fft.fftshift( np.fft.ifft2( g_fft * h_fft ) )
    f[ np.isnan(f) ] = 0
    return f




def print_volume( v, layer_name = 'test', colormap = cm.gray):
    if len(v.shape) == 2:
        vmax = np.max( v )
        im0 = plt.imshow(v, vmin=0, vmax= vmax, cmap=colormap)
        ax0 = plt.gca()
        plt.axis('off')
        plt.colorbar(im0)
        plt.savefig('tmp/' + layer_name + '.png', bbox_inches='tight', dpi=120)
        # plt.savefig('conc_profile_neutral.pdf', bbox_inches='tight', dpi=600)
        plt.clf()
        plt.cla()
    elif len(v.shape) == 4 and v.shape[3] > 1:
        for u in range(v.shape[2]):
            for v in range(v.shape[3]):
                vmax = np.max( v )
                im0 = plt.imshow(v[:,:,u,v], vmin=0, vmax= vmax, cmap=colormap)
                ax0 = plt.gca()
                plt.axis('off')
                plt.colorbar(im0)
                plt.savefig('tmp/' + layer_name + '_' + str(v) + '_' + str(u) + '.png', bbox_inches='tight', dpi=120)
                # plt.savefig('conc_profile_neutral.pdf', bbox_inches='tight', dpi=600)
                plt.clf()
                plt.cla()
    else:
        for u in range(v.shape[2]):
            vmax = np.max( v )
            im0 = plt.imshow(v[:,:,u], vmin=0, vmax= vmax, cmap=colormap)
            ax0 = plt.gca()
            plt.axis('off')
            plt.colorbar(im0)
            plt.savefig('tmp/' + layer_name + '_' + str(u) + '.png', bbox_inches='tight', dpi=120)
            # plt.savefig('conc_profile_neutral.pdf', bbox_inches='tight', dpi=600)
            plt.clf()
            plt.cla()


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
# layer 1
w = graph.get_tensor_by_name('prefix/weight:0')
b = graph.get_tensor_by_name('prefix/bias:0')
x = graph.get_tensor_by_name('prefix/input:0')
z = graph.get_tensor_by_name('prefix/add:0')
a = graph.get_tensor_by_name('prefix/Relu:0')
q = graph.get_tensor_by_name('prefix/Conv2D:0')
p = graph.get_tensor_by_name('prefix/MaxPool:0')
# layer 2
w2 = graph.get_tensor_by_name('prefix/weight_1:0')
b2 = graph.get_tensor_by_name('prefix/bias_1:0')
z2 = graph.get_tensor_by_name('prefix/add_1:0')
a2 = graph.get_tensor_by_name('prefix/Relu_1:0')
# layer 3
w3 = graph.get_tensor_by_name('prefix/weight_2:0')
b3 = graph.get_tensor_by_name('prefix/bias_2:0')
z3 = graph.get_tensor_by_name('prefix/add_2:0')
a3 = graph.get_tensor_by_name('prefix/Relu_2:0')
# # a = graph.get_tensor_by_name('prefix/MaxPool_2:0')
# y = graph.get_tensor_by_name('prefix/output:0')
# keep_prob = graph.get_tensor_by_name('prefix/keep_prob:0')
#
with tf.Session(graph=graph) as sess:
#     # weights
    w_out = w.eval()
    b_out = b.eval()
    b2_out = b2.eval()
    b3_out = b3.eval()
    w2_out = w2.eval()
    w3_out = w3.eval()
    print('shape of w_out')
    print(w_out.shape)
    print('shape of b_out')
    print(b_out.shape)
    print('shape of w2_out')
    print(w2_out.shape)
    print('shape of w3_out')
    print(w3_out.shape)
#
#     np.save('learned/' + layer + '.npy', w_out)
#     # input
#     z_out, a_out, y_out = sess.run([z,a,y], feed_dict = {x:img, keep_prob: 1})
    if depth == 3:
        [q_out, z_out, a_out, z2_out, a2_out, z3_out, a3_out] \
            = sess.run([q,z,a, z2,a2, z3,a3], feed_dict = {x:img})
        print('shape of z3_out:')
        print(z3_out.shape)
        print('shape of a3_out:')
        print(a3_out.shape)
    elif depth == 2:
        [q_out, z_out, a_out, z2_out, a2_out] = sess.run([q,z,a, z2,a2], feed_dict = {x:img})
        print('shape of z2_out:')
        print(z2_out.shape)
        print('shape of a2_out:')
        print(a2_out.shape)
    else:
        [q_out, z_out, a_out, p_out] = sess.run([q,z,a,p], feed_dict = {x:img})
        print('shape of p_out:')
        print(p_out.shape)
#     # y_out = sess.run(y, feed_dict = {x:img})
        print('shape of z_out:')
        print(z_out.shape)
        print('shape of a_out:')
        print(a_out.shape)



if depth == 3:

    # layer 3

    a3_out = np.reshape( a3_out, (a3_out.shape[1],a3_out.shape[2],a3_out.shape[3]) )
    print('shape of a3_out:')
    print(a3_out.shape)
    output, amax = maxpool(a3_out, 2, 2)

    print('shape of output')
    print(output.shape)
    print('shape of argmax')
    print(amax.shape)
    if print_layers:
        print_volume( output, 'maxpool3')

    maxact3 = max_activation( output )
    if print_layers:
        print_volume( maxact3, 'max_activation3')

    # unmaxp3 = unmaxpool( maxact3, amax, 2, 2 )
    unmaxp3 = unmaxpool( output, amax, 2, 2 )
    if print_layers:
        print_volume(unmaxp3, 'unmaxpool3')

    # w2_out  = np.reshape(w2_out, (w2_out.shape[0], w2_out.shape[1], w2_out.shape[3]))
    print('shape of unmaxp3:')
    print(unmaxp3.shape)
    print('shape of w3_out')
    print(w3_out.shape)

    dconv3 = deconv2( unmaxp3, w3_out, b3_out )
    if print_layers:
        print_volume(dconv3, 'deconv3')
    print('shape of dconv3')
    print(dconv3.shape)

    # misc.imsave('tmp/' + prefix + '_deconv3_sum.png', np.sum(dconv3, axis=2))
    print_volume(np.sum(dconv3, axis=2), 'deconv3_sum', cm.magma)

    # layer 2

    a2_out = np.reshape( a2_out, (a2_out.shape[1],a2_out.shape[2],a2_out.shape[3]) )
    print('shape of a2_out:')
    print(a2_out.shape)
    output, amax = maxpool(a2_out, 2, 2)

    print('shape of output')
    print(output.shape)
    print('shape of argmax')
    print(amax.shape)
    if print_layers:
        print_volume( output, 'maxpool2')

    unmaxp2 = unmaxpool( dconv3, amax, 2, 2 )
    if print_layers:
        print_volume(unmaxp2, 'unmaxpool2')

    # w2_out  = np.reshape(w2_out, (w2_out.shape[0], w2_out.shape[1], w2_out.shape[3]))
    print('shape of unmaxp2:')
    print(unmaxp2.shape)
    print('shape of w2_out')
    print(w2_out.shape)

    dconv2 = deconv2( unmaxp2, w2_out, b2_out )
    if print_layers:
        print_volume(dconv2, 'deconv2')
    print('shape of dconv2')
    print(dconv2.shape)

    # misc.imsave('tmp/' + prefix + '_deconv2_sum.png', np.sum(dconv2, axis=2))
    print_volume(np.sum(dconv2, axis=2), 'deconv2_sum', cm.magma)

    # layer 1

    a_out = np.reshape( a_out, (a_out.shape[1],a_out.shape[2],a_out.shape[3]) )
    output, amax = maxpool(a_out, 2, 2)

    unmaxp1 = unmaxpool( dconv2, amax, 2, 2 )

    dconv1 = deconv2( unmaxp1, w_out, b_out )
    # misc.imsave('tmp/' + prefix + '_deconv1_sum.png', np.sum(dconv1, axis=2))
    print_volume(np.sum(dconv, axis=2), 'deconv1_sum', cm.magma)

elif depth == 2:

    # layer 2

    a2_out = np.reshape( a2_out, (a2_out.shape[1],a2_out.shape[2],a2_out.shape[3]) )
    print('shape of a2_out:')
    print(a2_out.shape)
    output, amax = maxpool(a2_out, 2, 2)

    print('shape of output')
    print(output.shape)
    print('shape of argmax')
    print(amax.shape)
    if print_layers:
        print_volume( output, 'maxpool2')

    maxact2 = max_activation( output )
    if print_layers:
        print_volume( maxact2, 'max_activation1')

    unmaxp2 = unmaxpool( maxact2, amax, 2, 2 )
    if print_layers:
        print_volume(unmaxp2, 'unmaxpool2')

    # w2_out  = np.reshape(w2_out, (w2_out.shape[0], w2_out.shape[1], w2_out.shape[3]))
    print('shape of unmaxp2:')
    print(unmaxp2.shape)
    print('shape of w2_out')
    print(w2_out.shape)

    dconv2 = deconv2( unmaxp2, w2_out, b2_out)
    if print_layers:
        print_volume(dconv2, 'deconv2')
    print('shape of dconv2')
    print(dconv2.shape)

    # misc.imsave('tmp/deconv2_sum.png', np.sum(dconv2, axis=2))
    print_volume(np.sum(dconv, axis=2), 'deconv2_sum', cm.magma)

    # layer 1

    a_out = np.reshape( a_out, (a_out.shape[1],a_out.shape[2],a_out.shape[3]) )
    output, amax = maxpool(a_out, 2, 2)

    unmaxp1 = unmaxpool( dconv2, amax, 2, 2 )

    dconv1 = deconv2( unmaxp1, w_out, b_out )
    # misc.imsave('tmp/deconv1_sum.png', np.sum(dconv1, axis=2))
    print_volume(np.sum(dconv, axis=2), 'deconv1_sum', cm.magma)



else: # 1 layer

    a_out = np.reshape( a_out, (a_out.shape[1],a_out.shape[2],a_out.shape[3]) )
    z_out = np.reshape( z_out, (z_out.shape[1],z_out.shape[2],z_out.shape[3]) )
    p_out = np.reshape( p_out, (p_out.shape[1],p_out.shape[2],p_out.shape[3]) )
    output, amax = maxpool(a_out, 2, 2)

    print('shape of output')
    print(output.shape)
    if print_layers:
        print_volume( a_out, 'a1')
        print_volume( z_out, 'z1')
        print_volume( output, 'maxpool1')
        print_volume( p_out, 'maxpooltf1')
        print_volume( amax, 'argmax1', cm.jet)

    maxact = max_activation( output )
    if print_layers:
        print_volume( maxact, 'max_activation1')

    # print(amax)


    unmaxp = unmaxpool( maxact, amax, 2, 2 )
    # unmaxp = unmaxpool( output, amax, 2, 2 )
    if print_layers:
        print_volume(unmaxp, 'unmaxpool1')


    # w_out  = np.reshape(w_out, (w_out.shape[0], w_out.shape[1], w_out.shape[3]))
    # q_out  = np.reshape(q_out, (q_out.shape[1], q_out.shape[2], q_out.shape[3]))
    pad_h = int( (unmaxp.shape[0] - w_out.shape[0]) / 2 )
    pad_w = int( (unmaxp.shape[1] - w_out.shape[1]) / 2 )
    # w_pad = np.pad(w_out, ((pad_h,pad_h+1), (pad_w,pad_w+1), (0,0)), 'constant')
    print('shape of unmaxp:')
    print(unmaxp.shape)

    # ReLU
    unmaxp[ unmaxp < 0 ] = 0

    dconv = deconv2( unmaxp, w_out, b_out )
    # dconv = deconv2( q_out, w_pad )

    if print_layers:
        print_volume(dconv, 'deconv1')

    print_volume(np.sum(dconv, axis=2), 'deconv1_sum', cm.magma)
    # misc.imsave('tmp/deconv1_sum.png', np.sum(dconv, axis=2))


#
#
# if print_z:
#     z_out = np.reshape( z_out, (z_out.shape[1], z_out.shape[2], z_out.shape[3], 1) )
#     print('shape of z_out:')
#     print(z_out.shape)
#     var = np.std( z_out )
#
#     for u in range(z_out.shape[2]):
#         im0 = plt.imshow(z_out[:,:,u,0], vmin=-var, vmax=var, cmap=cm.coolwarm)
#         ax0 = plt.gca()
#         plt.axis('off')
#         plt.colorbar(im0)
#         #cbar0 = plt.colorbar(im0, ticks=[9.95, 10.00, 10.05])
#         #cbar0.ax.set_yticklabels(['<9.95', '10', '10.05'])
#
#         #plt.show()
#         plt.savefig('learned/zout_' + str(u) + '.png', bbox_inches='tight', dpi=120)
#         # plt.savefig('conc_profile_neutral.pdf', bbox_inches='tight', dpi=600)
#         plt.clf()
#         plt.cla()
#
# if print_a:
#     a_out = np.reshape( a_out, (a_out.shape[1], a_out.shape[2], a_out.shape[3], 1) )
#     print('shape of a_out:')
#     print(a_out.shape)
#     var = np.std( a_out )
#     vmax = np.max( a_out )
#
#     for u in range(a_out.shape[2]):
#         im0 = plt.imshow(a_out[:,:,u,0], vmin=0, vmax= vmax, cmap=cm.gray)
#         ax0 = plt.gca()
#         plt.axis('off')
#         plt.colorbar(im0)
#         #cbar0 = plt.colorbar(im0, ticks=[9.95, 10.00, 10.05])
#         #cbar0.ax.set_yticklabels(['<9.95', '10', '10.05'])
#
#         #plt.show()
#         plt.savefig('learned/aout_' + str(u) + '.png', bbox_inches='tight', dpi=120)
#         # plt.savefig('conc_profile_neutral.pdf', bbox_inches='tight', dpi=600)
#         plt.clf()
#         plt.cla()
#
# if print_filters:
#     var = np.std( w_out )
#
#     for u in range(w_out.shape[2]):
#         for v in range(w_out.shape[3]):
#             im0 = plt.imshow(w_out[:,:,u,v], vmin=-var, vmax=var, cmap=cm.coolwarm)
#             ax0 = plt.gca()
#             plt.axis('off')
#             plt.colorbar(im0)
#             #cbar0 = plt.colorbar(im0, ticks=[9.95, 10.00, 10.05])
#             #cbar0.ax.set_yticklabels(['<9.95', '10', '10.05'])
#
#             #plt.show()
#             plt.savefig('learned/filters/' + layer + '_' + str(u) + '_' + str(v) + '.png', bbox_inches='tight', dpi=120)
#             # plt.savefig('conc_profile_neutral.pdf', bbox_inches='tight', dpi=600)
#             plt.clf()
#             plt.cla()
#
#
# print('prediction:')
# print(y_out)
