#!/bin/bash  
# 0222 Tim
# In 0222B, the program looks strange. 
# This run is just to make sure the program runs fine.
echo "------------------------ timer.py ------------------------"
# --time: sleep time (unit: hour)
/usr/bin/python3 timer.py --time 0
# --path1-data: path of data
# --path2-data: subpath of data 
# --num-data: number of images included in each images
echo "------------------------ buildLabelCSV.py ------------------------"
/usr/bin/python3 buildLabelCSV.py --path1-data data --path2-data wikipedia/samples_256 --num-data 30000
# --test-percentage: the percentage for test set 
echo "------------------------ buildTFRecord.py ------------------------"
/usr/bin/python3 buildTFRecord.py --test-percentage 0.1
# --num-train-steps: number of min-batch traning steps
# --batch-size: the size of each mini-batch
# --print-steps: Print the log to screen every other this number of steps.
echo "------------------------ trainModel.py ------------------------"
/usr/bin/python3 trainModel.py --num-train-steps 1500 --batch-size 100 --print-steps 100 -- eval-steps 500 --image-size 128