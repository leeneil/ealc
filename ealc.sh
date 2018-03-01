#!/bin/bash  
# --path1-data: path of data
# --path2-data: subpath of data 
# --num-data: number of images included in each images
echo "------------------------ buildLabelCSV.py ------------------------"
python3 buildLabelCSV.py --path1-data data --path2-data wikipedia/samples_128 --num-data 100
# --test-percentage: the percentage for test set 
echo "------------------------ buildTFRecord.py ------------------------"
python3 buildTFRecord.py --test-percentage 0.2
# --num-train-steps: number of min-batch traning steps
# --batch-size: the size of each mini-batch
# --print-steps: Print the log to screen every other this number of steps.
echo "------------------------ trainModel.py ------------------------"
python3 trainModel.py --num-train-steps 100 --batch-size 10 --print-steps 10
