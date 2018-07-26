

############ Importing libraries ###############
# Environment settings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
os.environ['QT_QPA_PLATFORM']='offscreen'

# Append local path
import sys
sys.path.append('../lib/')

# Local libraries
from unet import *
from pre_process_image import *
from helper_functions import *

# Local Libraries

import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
from scipy import misc
import imageio
import h5py

# Keras Libraries
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from keras.utils.vis_utils import plot_model as plot
from keras.models import Model
from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, Merge, merge,\
Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core 

tf.reset_default_graph()

# ==============================================#

previous_epoch = 1
end_epoch = 20
epoch_num = 10

weight_path = '../batch_size_32/saved_weights/baseline_weights/'
weight_file = '../batch_size_32/saved_weights/baseline_weights/weight-'
checkpoint_path = '../batch_size_32/checkpoint_training/baseline/'

#Load model to train
model = baseline_unet(48,48,3)

#Create results folder
if not os.path.exists(checkpoint_path):
	os.makedirs(checkpoint_path)
if not os.path.exists(weight_path):
	os.makedirs(weight_path)

# ==============================================#


X_train, Y_train, X_validation, Y_validation = get_train_validation_dataset()

for i in range(previous_epoch, end_epoch + 1):
	if(os.path.exists(weight_file+str(i-1)+'.h5')): # Load previous weight
		model.load_weights(weight_file+str(i-1)+'.h5')
	else:
		# No weight initial
		pass

	history = model.fit(X_train,
						Y_train,
						batch_size=32,
						epochs=epoch_num,
						verbose=1,
						validation_data=(X_validation, Y_validation),
						shuffle = True)
	
	model.save_weights(weight_file+str(i)+'.h5')

	# Get the history list
	loss_history = history.history["loss"]
	accuracy_history = history.history["acc"]
	val_loss_history = history.history["val_loss"]
	val_accuracy_history = history.history["val_acc"]
		
	# Get the latest history list (10th Epoch)
	final_loss = loss_history[-1]
	final_acc = accuracy_history[-1]
	final_val_loss = val_loss_history[-1]
	final_val_acc = val_accuracy_history[-1]

	# Initial current checkpoint path
	epoch_path = checkpoint_path+'Epoch_'+str(i*10)+'/'
	
	if not os.path.exists(epoch_path):
		os.makedirs(epoch_path)

	history_summary = "Final Loss: " + str(final_loss) + "\n"+"Final Accuracy: "+ str(final_acc)+ "\n"+"Final Validation Loss: "+ str(final_val_loss)+ "\n"+"Final Validation Accuracy: "+ str(final_val_acc)+ "\n"

	# =========================================================== #
	
	file = open(epoch_path+'summary.txt','w')
	file.write(history_summary)
	file.close()

	file = open(epoch_path+'loss.txt','w')
	for elem in loss_history:
		file.write(str(elem))
		file.write('\n')
	file.close()
	
	file = open(epoch_path+'acc.txt','w')
	for elem in accuracy_history:
		file.write(str(elem))
		file.write('\n')
	file.close()

	file = open(epoch_path+'loss_val.txt','w')
	for elem in val_loss_history:
		file.write(str(elem))
		file.write('\n')
	file.close()

	file = open(epoch_path+'acc_val.txt','w')
	for elem in val_accuracy_history:
		file.write(str(elem))
		file.write('\n')
	file.close()
	
	# =========================================================== #

	# Append epoch
	file = open(checkpoint_path+'all_loss.txt', 'a')
	for elem in loss_history:
		file.write(str(elem))
		file.write('\n')
	file.close()

	file = open(checkpoint_path+'all_accuracy.txt', 'a')
	for elem in accuracy_history:
		file.write(str(elem))
		file.write('\n')
	file.close()

	file = open(checkpoint_path+'all_val_loss.txt', 'a')
	for elem in val_loss_history:
		file.write(str(elem))
		file.write('\n')
	file.close()

	file = open(checkpoint_path+'all_val_accuracy.txt', 'a')
	for elem in val_accuracy_history:
		file.write(str(elem))
		file.write('\n')
	file.close()

	# =========================================================== #

	# plot the training loss and accuracy
	plt.style.use("ggplot")
	x = range(1,epoch_num+1)
	plt.figure()
	plt.plot(x, loss_history, label="train_loss")
	plt.plot(x, val_loss_history, label="val_loss")
	plt.title("Baseline Training Loss")
	plt.xlabel("Epochs Number")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig(epoch_path+'Loss_result')
	plt.close()

	plt.figure()
	plt.plot(x, accuracy_history, label="train_acc")
	plt.plot(x, val_accuracy_history, label="val_acc")
	plt.title("Baseline Training Accuracy")
	plt.xlabel("Epoch Number")
	plt.ylabel("Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(epoch_path+'Acc_result')	
	plt.close()

