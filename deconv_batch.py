import sys
import deconv

for u, line in enumerate(sys.stdin):
    print('[' + str(u) + ']', end='\t')
    filename = line.rstrip()
    print(filename)
    label = filename.replace('/','_').replace('.png','')
    deconv.make_deconv( filename, 3, label, print_layers = False, debug = False )
    print('\tprocessed ' + label)
