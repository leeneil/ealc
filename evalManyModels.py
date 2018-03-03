import os
import numpy as np
from evaluateModel import evalModel
from trainModel import print_confuMat
import tensorflow as tf

def print_confuMat_to_file(file, confuMat):
    for row in confuMat:
        toPrint = ''
        for col in row:
            toPrint += "{:0.3f}".format(col) + "\t"
        file.write(toPrint)
        file.write("\n")

# Use GPU
with tf.device("/gpu:0"):
    pass       
        
##
model_dir = os.path.join('saved-model')
model_root = "optimized_ealc_tensorflow_"
tfrecords_dir = "tfrecords-output"

## 
name_List = [str(i) for i in range(30000,200000,10000)] 
name_List.append("Final")

## params
bath_size = 25
num_classes = 4
image_size = 128

for name in name_List:
    model_name = model_root + name + ".pb"
    print("========== Eval Model: " + os.path.join(model_dir, model_name)+"==========")
    accuracy_percent, accu_confusionMat = evalModel(model_dir, model_name, tfrecords_dir, bath_size, num_classes, image_size)
    print('accuracy_percent = ', accuracy_percent)
    print(accu_confusionMat)
    print_confuMat(accu_confusionMat)
    
    # write to file 
    file = open(os.path.join(model_dir, "evalManyModel"+ name +".txt \n"),"w")
    file.write("Model at the step:  " + name + "\n") 
    file.write("Test Set Accuracy: = {:0.5} \n".format(accuracy_percent))
    file.write("Confusion Matrix:  \n")
    print_confuMat_to_file(file, accu_confusionMat)
    file.write("\n")
    file.close()
    print("====================================")