import fileinput
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from scipy import misc
import sys
import cv2
import heapq

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

export_dir = '/Users/liponan/Work/ml/ealc-tmp/saved-model/'
model_name = 'chkp_ealc_tensorflow.pb'
vectorize = True
k = 5
s = 1
b = 2

heap_lim = 9

def print_grid( v, name = 'grid', n = 8, colormap = cm.gray):
    if len(v.shape) == 3:
        vmax = np.max( v )
        m = np.ceil( v.shape[2] / n )
        for u in range(v.shape[2]):
            plt.subplot( m, n, u+1)
            im0 = plt.imshow(v[:,:,u], vmin=0, vmax= vmax, cmap=colormap)
            ax0 = plt.gca()
            plt.axis('off')
            # plt.colorbar(im0)
        plt.savefig('tmp/' + name + '.png', bbox_inches='tight', dpi=120)
        # plt.savefig('conc_profile_neutral.pdf', bbox_inches='tight', dpi=600)
        plt.clf()
        plt.cla()
    else:
        pass

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

###############################################################################

graph = load_graph(export_dir + model_name)

for op in graph.get_operations():
    print(op.name)

x = graph.get_tensor_by_name('prefix/input:0')
w = graph.get_tensor_by_name('prefix/weight:0')
p = graph.get_tensor_by_name('prefix/MaxPool:0')

l = int(p.shape[3])
pad_size = int( (k-1)/2 )
# p_max = np.zeros( (1, p.shape[1], p.shape[2], p.shape[3]) )
# p_max_source = np.zeros( (1, p.shape[1], p.shape[2], p.shape[3]) )

p_heap = []
p_list = []
for u in range(l):
    p_list.append( [] )
    heapq.heappush( p_list[u], (0, -1, -1, '') )

with tf.Session(graph=graph) as sess:
    for u, line in enumerate(fileinput.input()):
        filename = line.rstrip()
        img = cv2.imread( filename )
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if vectorize:
            vimg = np.reshape(img, (-1, img.shape[0]*img.shape[1]))
            p_out = sess.run(p, feed_dict = {x:vimg})
        else:
            p_out = sess.run(p, feed_dict = {x:img})

        for v in range(l):
            act_max = np.max( p_out[0,:,:,v] )
            if act_max > p_list[v][0][0]:
                idx = np.argmax( p_out[0,:,:,v] )
                uu = int( idx / p_out.shape[1] )
                vv = idx % p_out.shape[1]
                if len(p_list[v]) < heap_lim:
                    heapq.heappush( p_list[v], (act_max, uu, vv, filename) )
                else:
                    heapq.heappushpop( p_list[v], (act_max, uu, vv, filename) )

            '''
            if act_max > p_list[v]['max']:
                p_list[v]['max'] = act_max
                idx = np.argmax( p_out[0,:,:,v] )
                uu = int( idx / p_out.shape[1] )
                vv = idx % p_out.shape[1]
                img_pad = np.pad( img, ( (pad_size,pad_size), (pad_size,pad_size)), 'constant', constant_values=255 )
                p_list[v]['img'] = img_pad[ (b*s*uu):(b*s*uu+k+b-1), (b*s*vv):(b*s*vv+k+b-1) ]
                p_list[v]['source'] = line
            '''

        # update_list = p_out > p_max
        # p_max[ update_list ] = p_out[ update_list ]
        # p_max_source[ update_list ] = i

total_score = 0

u_stack = np.zeros( (k+b-1,k+b-1,heap_lim*l) )
for u in range(l):
    v_stack = np.zeros( (k+b-1,k+b-1,heap_lim) )
    for v in range(heap_lim):
        total_score += p_list[u][v][0]
        filename = p_list[u][v][3]
        img = cv2.imread( filename )
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_pad = np.pad( img, ( (pad_size,pad_size), (pad_size,pad_size)), 'constant', constant_values=255 )
        (uu,vv) = p_list[u][v][1:3]
        v_stack[:,:,v] = img_pad[ (b*s*uu):(b*s*uu+k+b-1), (b*s*vv):(b*s*vv+k+b-1) ]
        u_stack[:,:,v+heap_lim*u] = v_stack[:,:,v]
    print_grid( v_stack, 'stack_' + str(u), 3)
print_grid( u_stack, 'stack', 18)

print('total score: ' + str(total_score))





'''
u_stack = np.zeros( (k+b-1,k+b-1,l) )
act_score = 0
for u in range(l):
    score = p_list[u]['max']
    act_score += score
    u_stack[:,:,u] = p_list[u]['img']
    print( str(u) + ': ' + str(score) + '\n >> ' +p_list[u]['source'] )
print('total score: ' + str(act_score))
# print_volume( u_stack, 'stack' )
print_grid( u_stack, 'stack', 8)
    # misc.imsave('tmp/patch_' + str(u) + '_.png', p_list[u]['img'])
'''
