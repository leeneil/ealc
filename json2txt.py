import json
import sys

narg = len(sys.argv)

if narg < 2:
    raise('please specify an input json file')
else:
    input_filename = sys.argv[1]
    if narg > 2:
        output_filename = sys.argv[2]
    else:
        output_filename = input_filename[0:-4] + 'txt'

data = json.loads( open(input_filename).read() )

txt = open(output_filename, 'w')

for f in data:
    for m in f:
        filename = m['filename']
        print(filename)
        txt.write(filename + '\n')

txt.close()
