import fileinput
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from scipy import misc
import sys
import cv2
import heapq
import json
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

make_figures = True

pb_file = 'saved-model/optimized_ealc_tensorflow_Final.pb'
layer = 1
if len(sys.argv) > 1:
    layer = int(sys.argv[1])
    if len(sys.argv) > 2:
        pb_file = sys.argv[2]

print('input: ' + pb_file)
print('analyzing layer ' + str(layer))


export_dir = '/Users/liponan/Work/ml/ealc-tmp/saved-model/'
model_name = 'chkp_ealc_tensorflow.pb'

vectorize = True
k = 5
s = 1
b = 2

heap_lim = 9

def img_patch( img, uu, vv, k=5, s=1, b=2, layer=1 ):
    uu0 = pow(b,layer) * uu
    uu1 = uu0 + k * ( pow(b,layer) -1 ) + 1
    vv0 = pow(b,layer) * vv
    vv1 = vv0 + k * ( pow(b,layer) -1 ) + 1
    return img[ uu0:uu1, vv0:vv1 ]


def make_mosaic( vol, space = 1):
    (h,w,l) = vol.shape
    m = int( np.ceil( np.sqrt(l) ) )
    n = int( np.ceil( l / m ) )
    canvas = 255 * np.ones( ( (h+space)*m-space, (w+space)*n-space, ) )
    for u in range(l):
        uu = u % m
        vv = int( u / m )
        canvas[ uu*(h+space):(uu*(h+space)+h), vv*(w+space):(vv*(w+space)+w) ] = vol[:,:,u]
    return canvas

def print_grid( v, name = 'grid', n = 8, colormap = cm.gray, dpi = 600):
    if len(v.shape) == 3:
        vmax = np.max( v )
        m = np.ceil( v.shape[2] / n )
        for u in range(v.shape[2]):
            plt.subplot( m, n, u+1)
            im0 = plt.imshow(v[:,:,u], vmin=0, vmax= vmax, cmap=colormap)
            ax0 = plt.gca()
            plt.axis('off')
            # plt.colorbar(im0)
        plt.savefig('tmp/' + name + '.png', bbox_inches='tight', dpi=dpi)
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

graph = load_graph(pb_file)

for op in graph.get_operations():
    print(op.name)

x = graph.get_tensor_by_name('prefix/input:0')
if layer == 1:
    w = graph.get_tensor_by_name('prefix/weight:0')
    p = graph.get_tensor_by_name('prefix/MaxPool:0')
else:
    w = graph.get_tensor_by_name('prefix/weight_' + str(layer-1) + ':0')
    p = graph.get_tensor_by_name('prefix/MaxPool_' + str(layer-1) + ':0')

l = int(p.shape[3])
pad_size = layer * int( (k-1)/2 * (pow(2,layer)-1) )
# p_max = np.zeros( (1, p.shape[1], p.shape[2], p.shape[3]) )
# p_max_source = np.zeros( (1, p.shape[1], p.shape[2], p.shape[3]) )

p_heap = []
p_list = []
label_list = []

for u in range(l):
    p_list.append( [] )
    heapq.heappush( p_list[u], (0, -1, -1, '') )

count = 0
with tf.Session(graph=graph) as sess:
    for u, line in enumerate(sys.stdin):
        # print(line)
        filename = line.rstrip()
        img = cv2.imread( filename )
        if img is None:
            print('weird: ' + filename)
            continue
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
        count += 1
        if count % 1000 == 0:
            print('done ' + str(count))

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


m = int( np.ceil( np.sqrt(heap_lim) ) )
n = int( np.ceil( heap_lim / m ) )
space = 3
h = k * ( pow(b,layer) -1 ) + 1
w = k * ( pow(b,layer) -1 ) + 1

if make_figures:
    print('making figures...')
    u_stack = np.zeros( ( m*(h+space)-space, n*(w+space)-space, l) )
    for u in range(l):
        label_list.append([])
        v_stack = np.zeros( (h, w, heap_lim) )
        for v in range(heap_lim):
            score = p_list[u][v][0]
            total_score += score
            filename = p_list[u][v][3]
            label = re.findall(r'\/([tw]{2}|[kr]{2}|[cn]{2}|[jp]{2})\/', filename)[0]
            label_list[u].append({'score':int(score), 'label': label, 'filename':filename})
            if filename == '':
                break
            img = cv2.imread( filename )
            if img is None:
                print('weird: ' + filename)
                continue
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_pad = np.pad( img, ( (pad_size,pad_size), (pad_size,pad_size)), 'constant', constant_values=255 )
            (uu,vv) = p_list[u][v][1:3]
            v_stack[:,:,v] = img_patch( img_pad, uu, vv, k, s, b, layer )
        u_stack[:,:,u] = make_mosaic( v_stack, space )
        print_grid( v_stack, 'layer_' + str(layer) + '_stack_' + str(u), 3, cm.gray, 300)
    print_grid( u_stack, 'layer_' + str(layer) + '_stack', 8, cm.gray, 600)

    print('total score: ' + str(total_score/l/heap_lim))

print(label_list)
with open('tmp/layer_' + str(layer) + '.json', 'w') as outfile:
    json.dump(label_list, outfile, ensure_ascii=False, indent=2)


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
