from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from os import walk
import os
import sys
import random
import numpy as np
from numpy import random
import re
import json
import math

random.seed(230)
noisy = True

fonts = ['../fonts/NotoSerifCJK-ExtraLight.ttc', '../fonts/NotoSerifCJK-Light.ttc',
    '../fonts/NotoSerifCJK-Regular.ttc', '../fonts/NotoSerifCJK-Medium.ttc',
    '../fonts/NotoSerifCJK-SemiBold.ttc', '../fonts/NotoSerifCJK-Bold.ttc',
    '../fonts/NotoSerifCJK-Black.ttc', '../fonts/NotoSansCJK-Thin.ttc',
    '../fonts/NotoSansCJK-Light.ttc', '../fonts/NotoSansCJK-DemiLight.ttc',
    '../fonts/NotoSansCJK-Regular.ttc', '../fonts/NotoSansCJK-Medium.ttc',
    '../fonts/NotoSansCJK-Bold.ttc', '../fonts/NotoSansCJK-Black.ttc']
font = 'fonts/NotoSansCJK-Regular.ttc'
new_size = 128
sample_path = 'samples_128/'


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

dest_path = dpath + sample_path
if os.path.exists(dest_path):
    pass
else:
    os.mkdir(dest_path)

pic_count = 0

# label = 'kr'

for filename in files:
    if pic_count > lim:
        break
    if not re.match(r'\w+\.json', filename):
        continue
    full_filename = dpath + filename
    print('processing ' + full_filename)


    data = json.load( open(full_filename) )
    paragraphs = data['paragraphs']

    for paragraph in paragraphs:
        if pic_count > lim:
            break
        cues = paragraph['cues']
        for cue in cues:
            if pic_count > lim:
                break
            # text = re.sub(r'\\n', '', cue['text'])
            text = cue['text'].replace('\n','')
            if len(text) < 5:
                continue
            print('* ' + text)

            img = Image.new('L', (new_size, new_size), color=255)
            d = ImageDraw.Draw(img)
            font = fonts[random.randint(0, len(fonts)-1)]
            font_size = random.randint(10, 30)
            f = ImageFont.truetype(font, font_size)

            x_offset = ( new_size/8.0 * (random.random()-0.0) )
            y_offset = ( new_size/4.0 * (random.random()-0.0) )
            n_perline = max( 1, math.ceil( (new_size-1*x_offset) / font_size )-1)
            n_lines = math.ceil( 1.0 * len(text) / n_perline )

            y_delta = int( font_size * 1.5 )
            for t in range(n_lines):
                if t == n_lines-1:
                    d.text( (x_offset,y_offset+t*y_delta), \
                        text[ (0+t*n_perline): ], fill=0, font=f)
                else:
                    d.text( (x_offset,y_offset+t*y_delta), \
                        text[(0+t*n_perline):((t+1)*n_perline)], fill=0, font=f)

            if noisy:
                nosie = random.randint(1,25) * random.randn( img.shape )
                img = img + noise


            img.save( dest_path + str(pic_count).zfill(6) + '.png')
            img.close()
            pic_count += 1


        print('')
