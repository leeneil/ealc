#!/bin/bash  
echo "------------------------ timer.py ------------------------"
# --time: sleep time (unit: hour)
python3 timer.py --time 0
echo "------------------------ buildLabelCSV.py ------------------------"
# --path1-data: path of data before "tw"
# --path2-data: path of data after "tw"
python3 buildLabelCSV.py --path1-data data --path2-data ted/samples_128 --num-data 624000
echo "------------------------ buildTFRecord.py ------------------------"
python3 buildTFRecord.py --test-percentage 0.05
echo "------------------------ trainModel.py ------------------------"
python3 trainModel.py --num-train-steps 200000 --batch-size 100 --print-steps 1000 --eval-steps 10000 --image-size 128
