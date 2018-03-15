

from os import walk
import os
import sys
import random
import numpy as np
import re
import json
import math

random.seed(230)




lim = 10
if len(sys.argv) < 2:
    raise('Please specify label')
else:
    label = sys.argv[1]
    dpath = 'data/' + label + '/ted/'
    if len(sys.argv) > 2:
        lim = int(sys.argv[2])

for (dirpath, dirnames, filenames) in walk(dpath):
    files = filenames
    break



# label = 'kr'

counts = {}

file_count = 0

for filename in files:
    if not re.match(r'\w+\.json', filename):
        continue
    full_filename = dpath + filename
    print('analyzing ' + full_filename)


    data = json.load( open(full_filename) )
    paragraphs = data['paragraphs']

    for paragraph in paragraphs:
        cues = paragraph['cues']

        for cue in cues:
            text = cue['text'].replace('\n','')
            if len(text) < 5:
                continue
            print('* ' + text)

            for c in text:
                if c in counts:
                    counts[c] += 1
                else:
                    counts[c] = 1
    file_count += 1
    if file_count >= lim:
        break

with open('tmp/' + label + '_frequency_counts.json', 'w') as outfile:
    json.dump(counts, outfile, ensure_ascii=False, indent=2)
print(counts)
