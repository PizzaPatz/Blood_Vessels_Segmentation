import sys
sys.path.append('../lib/')
from imports import *
from helper_functions import *
from unet import *
from full_patch import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.models import load_model

# Generate prediction #
#================================================
# Check for correct argument(s)
# 1 = architecture
# 2 = weight number (multiple of 10)
try:
	print('[1] Repeating argument(s)\n')
	print('Architecture: '+sys.argv[1]+'\n') #baseline
	print('Weight: '+sys.argv[2]+' | Epoch: '+str(int(sys.argv[2])*10)+'\n')

	architecture = sys.argv[1] 
	weight_epoch = sys.argv[2]

	if architecture in ['baseline', 'residual', 'all_conv'] and int(weight_epoch) < 16:
		print('[2] Starting prediction')
	else:
		print('[2] Invalid inputs')
		sys.exit(0)
except IndexError:
	print('[1] Invalid number of argument(s)')
	sys.exit(0)

#================================================

# Select the architecture

if architecture == 'baseline':
	model = baseline_unet(48,48,3)
	print('Baseline Net')

model.load_weights('../batch_size_32/saved_weights/'+architecture+'_weights/weight-'+weight_epoch+'.h5')

## Read an image to predict
#X_train = plt.imread('../../DRIVE/testing/images/01_testing.tif')
X_train = []
for i in range(1,21):
	j = str(i).zfill(2)
	im, im_horizontal, im_vertical = extract_full_patches(plt.imread('../DRIVE/test/images/'+j+'_test.tif'))
	result = model.predict(im)
	result_horizontal = model.predict(im_horizontal)
	result_vertical = model.predict(im_vertical)
	## reshape prediction back into squares and 1 dimension
	result = np.reshape(result, (result.shape[0],48,48,2))
	result = np.round(result)
	result = processOutputVector(result, 48, 48)
	result_horizontal = np.reshape(result_horizontal, (result_horizontal.shape[0],48,48,2))
	result_horizontal = np.round(result_horizontal)
	result_horizontal = processOutputVector(result_horizontal, 48, 48)
	result_vertical = np.reshape(result_vertical, (result_vertical.shape[0],48,48,2))
	result_vertical = np.round(result_vertical)
	result_vertical = processOutputVector(result_vertical, 48, 48)
	
	## reconstruct patches back into an image
	im_out = reconstruct_patches(result, result_horizontal, result_vertical)
	imread_visual_out2(im_out, architecture+'_weight_'+weight_epoch+'_prediction', j+'_test')
