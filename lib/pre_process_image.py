import sys
sys.path.append('../')

# Importing libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from scipy import misc
import imageio
import matplotlib.image as mpimg

#==== Configuration ====#

im_height = 48
im_width = 48
num_patches = 5000
seed =1

# ===================== #

def process_training_samples():
	# initialize our first training sample vector
	X_train = plt.imread('../DRIVE/training/images/21_training.tif')
	X_patches = image.extract_patches_2d(X_train, (im_height, im_width), max_patches=num_patches, random_state=seed)
	X_validation = X_patches[4900:5000]
	X_patches = X_patches[0:4900]
	
	Y_train = plt.imread('../DRIVE/training/1st_manual/21_manual1.gif')
	Y_patches = image.extract_patches_2d(Y_train, (im_height, im_width), max_patches=num_patches, random_state=seed)
	Y_validation = Y_patches[4900:5000]
	Y_patches = Y_patches[0:4900]
	
	## loop through the rest of the files and append to our initial
	for x in range(22,41):
		currentX = plt.imread('../DRIVE/training/images/' + str(x) + '_training.tif')
		currentXPatches = image.extract_patches_2d(currentX, (im_height, im_width), max_patches=num_patches, random_state=seed)
		X_patches = np.append(X_patches, currentXPatches[0:4900], axis=0)
		X_validation = np.append(X_validation, currentXPatches[4900:5000], axis=0)
	
		currentY = plt.imread('../DRIVE/training/1st_manual/' + str(x) + '_manual1.gif')
		currentYpatches = image.extract_patches_2d(currentY, (im_height, im_width), max_patches=num_patches, random_state=seed)
		Y_patches = np.append(Y_patches, currentYpatches[0:4900], axis=0)
		Y_validation = np.append(Y_validation, currentYpatches[4900:5000], axis=0)
    	
	return X_patches,Y_patches,X_validation,Y_validation

def process_y_label(Y_train,Y_validation):
	Y_train = np.stack((Y_train,)*2, -1)
	Y_train = Y_train / 255
	Y_validation = np.stack((Y_validation,)*2, -1)
	Y_validation = Y_validation / 255
	for x in range(Y_train.shape[0]):
		process_label_vector(Y_train[x,:,:],im_height,im_width)
	for y in range(Y_validation.shape[0]):
		process_label_vector(Y_validation[y,:,:],im_height,im_width)
	Y_train = np.reshape(Y_train, (Y_train.shape[0], im_height * im_width, 2))
	Y_validation = np.reshape(Y_validation, (Y_validation.shape[0], im_height * im_width, 2))
	return Y_train, Y_validation

def process_label_vector(patches,height, width):
	# iterate through reshaped patches and update new patches accordingly
	for i in range(height):
		for j in range(width):
		# invert second channel
			if  patches[i,j,1] == 0:
				patches[i,j,1] = 1
			else:
				patches[i,j,1] = 0
	return patches


def get_train_validation_dataset():
	X_train,Y_train,X_validation,Y_validation = process_training_samples()
	Y_train,Y_validation = process_y_label(Y_train,Y_validation)
	return X_train, Y_train, X_validation, Y_validation
