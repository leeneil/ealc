from os import walk
import os
import sys
import random
import numpy as np
import re
import json
import math
import heapq

random.seed(230)


heap_lim = 100
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
c_count = 0

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
                c_count += 1
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

top_counts = [(0,'')]

for key in counts:
    this_count = counts[key]
    this_freq = this_count*1.0 / c_count
    if this_count > top_counts[0][0]:
        if len(top_counts) >= heap_lim:
            heapq.heappushpop(top_counts, (this_count, this_freq, key))
        else:
            heapq.heappush(top_counts, (this_count, this_freq, key))

top_counts.sort()

with open('tmp/' + label + '_frequency_counts_top' + str(heap_lim) + '.json', 'w') as outfile:
    json.dump(top_counts, outfile, ensure_ascii=False, indent=2)
print(top_counts)
