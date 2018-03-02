import os
import numpy as np
from evaluateModel import evalModel
from trainModel import print_confuMat

def print_confuMat_to_file(file, confuMat):
    for row in confuMat:
        toPrint = ''
        for col in row:
            toPrint += "{:0.3f}".format(col) + "\t"
        file.write(toPrint)
        
        
##
model_dir = os.path.join('saved-model')
model_root = "optimized_ealc_tensorflow_"
tfrecords_dir = "tfrecords-output"

## 
name_List = [str(i) for i in range(10000,200000,10000)] 
name_List.append("Final")

##
for name in name_List:
    model_name = model_root + name + ".pb"
    print("========== Eval the model ==========")
    print("model file: " + os.path.join(model_dir, model_name))
    accuracy_percent, accu_confusionMat = evalModel(model_dir, model_name, tfrecords_dir)
    print('accuracy_percent = ', accuracy_percent)
    print(accu_confusionMat)
    print_confuMat(accu_confusionMat)
    
    # write to file 
    file = open("evalManyModel"+ name +".txt","w")
    file.write("Model at the step:  " + name) 
    file.write("Test Set Accuracy: = {:0.5}".format(accuracy_percent))
    file.write("Confusion Matrix:  " + name)
    print_confuMat_to_file(file, accu_confusionMats)
    file.write("\n")
    file.close()
    print("====================================")