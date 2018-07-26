
from imports import *
import os
import numpy as np

# Output image from 'imread' assignment. This is to visualize
#	the data
def imread_visual_out(imread_data, subdir, file_name):
	print('<-- saving image file to images/'+subdir+'/'+file_name+' -->')
	fig = plt.figure()
	plt.axis('off')
	fig.set_size_inches(1, 1, forward=False)
	ax = plt.Axes(fig, [0.,0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	plt.imshow(imread_data)
	if not os.path.exists('images/'+subdir):
		os.makedirs('images/'+subdir)
	fig.savefig('images/'+subdir+'/'+file_name+'.png')
	pass

def imread_visual_out2(imread_data, subdir, file_name):
	print('<-- saving image file to images/'+subdir+'/'+file_name+' -->')
	w = imread_data.shape[0]
	h = imread_data.shape[1]
	print(str(w)+"---"+str(h))
	fig = plt.figure()
	DPI = fig.get_dpi()
	fig.set_size_inches(h/float(DPI),w/float(DPI), forward=False)
	ax = plt.Axes(fig, [0.,0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	plt.imshow(imread_data)
	if not os.path.exists('images/'+subdir):
		os.makedirs('images/'+subdir)
	fig.savefig('images/'+subdir+'/'+file_name+'.png')
	pass

def get_first_15_patches_2d(y_patch):
	print('<-- saving patches files to images/y_patches/y_[indxes] -->' )
	if not os.path.exists('images/y_patches'):
		os.makedirs('images/y_patches')
	for i in range(0,15):
		fig = plt.figure()
		plt.imshow(y_patch[i,:,:])
		fig.savefig('images/y_patches/y_'+str(i))
		plt.clf()
		plt.close()
	pass

def get_first_15_patches_3d(x_patch):
	print('<-- saving patches files to images/x_patches/x_[indxes] -->' )
	if not os.path.exists('images/x_patches'):
		os.makedirs('images/x_patches')
	for i in range(0,15):
		fig = plt.figure()
		plt.imshow(x_patch[i,:,:,:])
		fig.savefig('images/x_patches/x_'+str(i))
		plt.clf()
		plt.close()
	pass

# Plotting function
def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)

# Patching dataset accross 20 pictures
def seperation_patch(total_patches, patch_batch):
	if((total_patches % patch_batch) == 0):
		pass
	else:
		print('Invalid Size')


# Plotting function
def plot_loss_accuracy(H, epoch_num):
	# plot the training loss and accuracy
	plt.style.use("ggplot")
	N = epoch_num

	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.title("Training Loss")
	plt.xlabel("Epochs #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig('Loss_result')
	plt.close()

	plt.figure()
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Accuracy")
	plt.legend(loc="lower left")
	plt.savefig('Acc_result')


# Inverting function for 2nd channel
def invert_second_channel(x):
	height = x.shape[0]
	width = x.shape[1]
	channel = x.shape[2]

	for i in range(0, height):
		for j in range(0, width):
			if(x[i,j,1] == [0]):
				x[i,j,1] = 1
			elif(x[i,j,1] == [1]):
				x[i,j,1] = 0
		return x

# Reverse 48x48x2 channels predict into a picture 48x48x1
def construct_predict_image(x):
	#print("START CONSTRUCT")
	height = x.shape[0]
	width = x.shape[1]
	channel = x.shape[2]
	#print('->>', height, '-', width, '-', channel)
	johncena = 0
	# Create empty output
	final_output = np.zeros([48, 48], dtype=np.uint8)

	for i in range(0, height):
		for j in range(0, width):
			if(x[i,j,0] == 1):
				final_output[i,j] = 1
				#print("JOHN CENA",johncena)
				#johncena = johncena + 1
			elif(x[i,j,0] == 0):
				final_output[i,j] = 0
	return final_output




def processOutputVector(resultVector, height, width):
    final_output = np.zeros((resultVector.shape[0],height,width))
    for k in range(resultVector.shape[0]):
        for i in range(0, height):
            for j in range(0, width):
                if(resultVector[k,i,j,0] == 1):
                    final_output[k,i,j] = 1
                elif(resultVector[k,i,j,0] == 0):
                    final_output[k,i,j] = 0
    return final_output

def processTestPatches(X_test):
    """
    Prepares the test data image into evenly divided 12 x 11 (48 x 48) patches
    
    """
    patchHeight = 48
    patchWidth = 48
    channels = 3
    
    imageHeight = 584
    imageWidth = 565
    
    ## calculate the number of iterations needed
    horizontalIteration = int(imageWidth/patchWidth)
    verticalIteration = int(imageHeight/patchHeight)
    
    ## compute the initial first patch on top left corner
    patches = X_test[0:48,0:48,:]
    patches = np.reshape(patches, (1,patchHeight,patchWidth,channels))
    ## initialize starting points for patches
    xl = 0
    xr = 48
    yl = 0
    yr = 48
    ## process test image into patches
    for x in range(verticalIteration):
        for y in range(horizontalIteration):
            
            ## append new patch to array of patches
            newPatchToAdd = np.reshape(X_test[xl:xr,yl:yr,:], (1,patchHeight,patchWidth,channels))
            patches = np.append(patches,newPatchToAdd, axis=0)
            
            ## update horizontal position by patch block size
            yl+= patchWidth
            yr+= patchWidth
            
        ## reset horizontal position, update vertical position
        yl = 0
        yr = patchWidth
        xl += patchHeight
        xr += patchHeight
        
    return patches


def reconstructPatchesToImage(patches):
    """
    Takes the blocks of patches and concatenates them together for the full image.
    
    return the full image
    """
    
    ## prepare the first row
    row0 = patches[1,:,:]
    for x in range(1,11):
        row0 = np.concatenate((row0,patches[1+x,:,:]),axis=1)
        
    rows = np.reshape(row0, ((1,48,528)))
    
    ## prepare the rest of the rows and stack them together
    for x in [12,23,34,45,56,67,78,89,100,111,122]: ## Blocks of 11 patches
        newRow = patches[x,:,:]
        for y in range(1,11):
            newRow = np.concatenate((newRow,patches[x+y,:,:]), axis=1)
        
        ## concatenate newly created row to existing rows
        newRow = np.reshape(newRow,(1,48,528))
        rows = np.concatenate((rows,newRow),axis=0)
    
    ## concatenate all the rows together to form the final image
    final = rows[0]
    for x in range(1,rows.shape[0]):
        final = np.concatenate((final,rows[x]), axis=0)
    
    return final

def processTestSamples(image_num):
	image_num = str(image_num).zfill(2)
	numPatches = 150

	## initialize our first test sample vector
	X_test = plt.imread('../DRIVE/test/images/'+image_num+'_test.tif')
	X_patches = image.extract_patches_2d(X_test, (48, 48), max_patches=numPatches, random_state=1)

	Y_test = plt.imread('../DRIVE/test/1st_manual/'+image_num+'_manual1.gif')
	Y_patches = image.extract_patches_2d(Y_test, (48, 48), max_patches=numPatches, random_state=1)

	## loop through the rest of the files and append to our initial
	for x in range(1,21):	
		x = str(x).zfill(2)
		currentX = plt.imread('../DRIVE/test/images/'+x+'_test.tif')
		currentY = plt.imread('../DRIVE/test/1st_manual/'+x+'_manual1.gif')
		currentXPatches = image.extract_patches_2d(currentX, (48, 48), max_patches=numPatches, random_state=1)
		X_patches = np.append(X_patches, currentXPatches, axis=0)

		currentYpatches = image.extract_patches_2d(currentY, (48, 48), max_patches=numPatches, random_state=1)
		Y_patches = np.append(Y_patches, currentYpatches, axis=0)

	return X_patches,Y_patches

def processYLabel(Y_train,Y_validation):
    Y_train = np.stack((Y_train,)*2, -1)
    Y_train = Y_train / 255
    Y_validation = np.stack((Y_validation,)*2, -1)
    Y_validation = Y_validation / 255

    for x in range(Y_train.shape[0]):
        processLabelVector(Y_train[x,:,:],48,48)

    for y in range(Y_validation.shape[0]):
        processLabelVector(Y_validation[y,:,:],48,48)
    Y_train = np.reshape(Y_train, (Y_train.shape[0], 48 * 48, 2))
    Y_validation = np.reshape(Y_validation, (Y_validation.shape[0], 48 * 48, 2))
    
    return Y_train, Y_validation



def processLabelVector (patches,height, width):
    # iterate through reshaped patches and update new patches accordingly
    for i in range(height):
        for j in range(width):
            # invert second channel
            if  patches[i,j,1] == 0:
                patches[i,j,1] = 1

            else:
                patches[i,j,1] = 0
                
    return patches
