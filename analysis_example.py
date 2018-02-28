import os
from trainModel import ModelLog
from analysisTool import get_training_log
from analysisTool import get_tfModel_params

# Get log from a .pickle file
log = get_training_log()
print('-------- The content of Log --------')
print(log.keys())
print('------------------------------------')
print('overall training accuracy: ',log['train_overall_accu'] )
print('overall training confusion matrix: ')
print(log['train_overall_confuMat'])


# Get the weigthts/bias/filters of a model from a .pb file
params = get_tfModel_params()
print('-------- The content of Params --------')
print(params.keys())
print('---------------------------------------')


