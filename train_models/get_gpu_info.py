

############ Importing libraries ###############
# Environment settings
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#os.environ['QT_QPA_PLATFORM']='offscreen'

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


