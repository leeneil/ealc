#!/bin/bash  
# --path1-data: path of data
# --path2-data: subpath of data 
# --num-data: number of images included in each images
echo "------------------------ buildLabelCSV.py ------------------------"
/usr/bin/python3 buildLabelCSV.py --path1-data data --path2-data ted/samples_128 --num-data 510000
# --test-percentage: the percentage for test set 
echo "------------------------ buildTFRecord.py ------------------------"
/usr/bin/python3 buildTFRecord.py --test-percentage 0.05
# --num-train-steps: number of min-batch traning steps
# --batch-size: the size of each mini-batch
# --print-steps: Print the log to screen every other this number of steps.
echo "------------------------ trainModel.py ------------------------"
/usr/bin/python3 trainModel.py --num-train-steps 50000 --batch-size 100 --print-steps 100
