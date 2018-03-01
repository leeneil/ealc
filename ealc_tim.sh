#!/bin/bash  

echo "------------------------ timer.py ------------------------"
# --time: sleep time (unit: hour)
python3 timer.py --time 0
echo "------------------------ buildLabelCSV.py ------------------------"
# --path1-data: path of data before "tw"
# --path2-data: path of data after "tw"
python3 buildLabelCSV.py --path1-data data --path2-data ted/samples_128 --num-data 600
echo "------------------------ buildTFRecord.py ------------------------"
python3 buildTFRecord.py --test-percentage 0.1
echo "------------------------ trainModel.py ------------------------"
python3 trainModel.py --num-train-steps 100 --batch-size 10 --print-steps 10 -- eval-steps 30 --image-size 128
