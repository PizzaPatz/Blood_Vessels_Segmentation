import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
# Local libraries
import sys
sys.path.append('../lib/')
from helper_functions import *
from unet import *
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction import image
import numpy as np
from sklearn import metrics

# Keras
import keras
from keras.models import load_model

#=========================================#
# Check for correct arguments
# 1 = architecture
# 2 = weight number (multiple of 10)
try:
	print('[1] Repeating argument(s)\n')
	print('Architecture: '+sys.argv[1]+'\n') # baseline
	print('Weight: '+sys.argv[2]+' | Epoch: '+str(int(sys.argv[2])*10)+'\n')
	
	architecture = sys.argv[1]
	weight_epoch = sys.argv[2]

	if architecture in ['baseline', 'residual', 'all_conv'] and int(weight_epoch) < 16:
		print('[2] Starting calculation AUC ROC')
	else:
		print('[2] Invalid inputs')
		sys.exit(0)

except IndexError:
	print('[1] Invalid number of argument(s)')
	sys.exit(0)
#=========================================#

#=========================================#
# Declaraction
AUC_ROC_Ave = 0 # Average of AUC ROC
TEST_DATASET_NUM = 20 # Constant: number of test data
batch_size = '32'
#=========================================#

base_path = '../roc_results/' + architecture + '/weight-'+ weight_epoch
graph_path = base_path+'/graphs/'
manuscript_path = base_path+'/manuscript/'
# init directories
if not os.path.exists(graph_path):
	os.makedirs(graph_path)
if not os.path.exists(manuscript_path):
	os.makedirs(manuscript_path)

# Select the architecture
if architecture == 'baseline':
	model = baseline_unet(48,48,3)
	print('Baseline Net')

# load weight
model.load_weights('../batch_size_'+batch_size+'/saved_weights/'+architecture+'_weights/weight-'+str(weight_epoch)+'.h5')

for i in range(1,TEST_DATASET_NUM+1):
	## Process test dataset
	X_test,Y_test = processTestSamples(i)
	_, Y_test = processYLabel(Y_test,Y_test)
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Image: '+ str(i))
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	## compute probability of each pixel being a blood vessel
	X_test_probs = model.predict(X_test)
	X_test_ROC = X_test_probs[:,:,0]

	## process probability vector and label vector to be one dimensional## proc 
	Y_test_ROC = np.reshape(Y_test,(Y_test.shape[0],48,48,2))
	Y_test_ROC = processOutputVector(Y_test_ROC,48,48)
	Y_test_ROC = np.reshape(Y_test_ROC,(Y_test_ROC.shape[0],48,48))

	X_test_ROC = np.reshape(X_test_ROC,(X_test_ROC.shape[0] * X_test_ROC.shape[1]))
	Y_test_ROC = np.reshape(Y_test_ROC,(Y_test_ROC.shape[0] * Y_test_ROC.shape[1] * Y_test_ROC.shape[2]))

	## compute values for the ROC using SK_Learn. 1 = yes blood vessel
	fpr, tpr, thresholds = metrics.roc_curve(Y_test_ROC, X_test_ROC, pos_label=1)
	roc_auc = metrics.auc(fpr, tpr)

	# Dataset output convention
	# This will be named according to DRIVE test dataset
	result_out = ('%02d'%i)+'_test'

	## plot ROC curve
	fig = plt.figure()
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.10f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	fig.savefig(graph_path + result_out + '_roc.png')

	# sum auc roc average
	AUC_ROC_Ave = AUC_ROC_Ave + roc_auc

	# output manuscript
	file = open(manuscript_path+'manuscript.txt', 'a')
	auc_result = ('%0.10f' % roc_auc)
	file.write('ROC \''+result_out+'\': ' + auc_result +'\r\n')
	file.write('Test loss: '+str(score[0])+'\r\n')
	file.write('Test accuracy: '+str(score[1])+'\r\n')
	file.write('\r\n')
	file.close()

# Average the AUC
file = open(manuscript_path+'manuscript.txt', 'a')
file.write('===================================\r\n')
file.write('Average AUC ROC on 20 test dataset: ' + str(AUC_ROC_Ave / float(20))+'\r\n')
file.write('===================================\r\n')
file.close()
