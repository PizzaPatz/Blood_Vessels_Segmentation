from imports import *
from keras.optimizers import SGD


# NOTE: this is from https://github.com/orobix/retina-unet/blob/master/src/retinaNN_training.py
def baseline_unet(height, width, ch):
	#model = Sequential()
	inputs = Input(shape=(height,width,ch))
	conv1 = (Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, ch), padding='same',data_format='channels_last'))(inputs)
	conv1 = (Dropout(0.2))(conv1)
	conv1 = (Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last'))(conv1)
	maxp1 = (MaxPooling2D(pool_size=(2,2)))(conv1)
	
	conv2 = (Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last'))(maxp1)
	conv2 = (Dropout(0.2))(conv2)
	conv2 = (Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last'))(conv2)
	maxp2 = (MaxPooling2D(pool_size=(2,2)))(conv2)
	
	conv3 = (Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last'))(maxp2)
	conv3 = (Dropout(0.2))(conv3)
	conv3 = (Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last'))(conv3)
	
	up1 = UpSampling2D(size=(2, 2))(conv3)
	up1 = concatenate([conv2,up1],axis=3)
	
	conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up1)
	conv4 = Dropout(0.2)(conv4)
	conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4)
	
	up2 = UpSampling2D(size=(2, 2))(conv4)
	up2 = concatenate([conv1,up2], axis=3)
	conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up2)
	conv5 = Dropout(0.2)(conv5)
	conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5)
	
	conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv5)
	conv6 = core.Reshape((height*width,2))(conv6)
	conv6 = core.Permute((1,2))(conv6)
	conv7 = core.Activation('softmax')(conv6)
	
	model = Model(inputs=inputs, outputs=conv7)
	
	#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.3, nesterov=False)
	#model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
	model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
	return model

# ====================================================== #