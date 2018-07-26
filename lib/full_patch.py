from imports import *
import os
import numpy as np

def extract_full_patches(X_test):
	patch_height = 48
	patch_width = 48
	channels = 3

	image_height = 584
	image_width = 565
	
	# Maximum patch width and height
	horizontal_iteration = int(image_width / patch_width)
	vertical_iteration = int(image_height / patch_height)
	
	## compute the initial first patch on top left corner
	patches = X_test[0:48, 0:48, :]
	patches = np.reshape(patches, (1, patch_height, patch_width, channels))
	## initialize starting points for patches
	xl = 0
	xr = 48
	yl = 0
	yr = 48
	## process test image into patches
	for x in range(vertical_iteration):
		for y in range(horizontal_iteration):

			## append new patch to array of patches
			new_patch_to_add = np.reshape(X_test[xl:xr,yl:yr,:], (1,patch_height,patch_width,channels))
			patches = np.append(patches,new_patch_to_add, axis=0)

			## update horizontal position by patch block size
			yl+= patch_width
			yr+= patch_width

		## reset horizontal position, update vertical position
		yl = 0
		yr = patch_width
		xl += patch_height
		xr += patch_height

	print('main patches:' +str(patches.shape))	
	
	patches_horizontal = X_test[0:48, 0:48, :]
	patches_horizontal = np.reshape(patches_horizontal, (1, patch_height, patch_width, channels))
	# Remainder patches on horizontal
	xl = 0
	xr = 48
	yl = image_height - 48 
	yr = image_height
	for k in range(horizontal_iteration): # Bottom remainders
		new_patch_to_add = np.reshape(X_test[yl:yr, xl:xr, :], (1, patch_height, patch_width, channels))
		patches_horizontal = np.append(patches_horizontal, new_patch_to_add, axis=0)
		xl += patch_width
		xr += patch_width	
	#print('horizontal remainder patches: ' +str(patches_horizontal.shape))

	patches_vertical = X_test[0:48, 0:48, :]
	patches_vertical = np.reshape(patches_vertical, (1, patch_height, patch_width, channels))
	# Remainder patches on vertical
	xl = image_width - 48
	xr = image_width
	yl = 0
	yr = 48
	for j in range(vertical_iteration): # Right remainders
		new_patch_to_add = np.reshape(X_test[yl:yr, xl:xr, :], (1, patch_height, patch_width, channels))
		patches_vertical = np.append(patches_vertical, new_patch_to_add, axis=0)
		yl += patch_height
		yr += patch_height
	#print('vertical remainder patches: ' + str(patches_vertical.shape))

	# Last patch in the bottom right corner
	xl = image_width - 48
	xr = image_width
	yl = image_height - 48
	yr = image_height
	new_patch_to_add = np.reshape(X_test[yl:yr, xl:xr, :], (1, patch_height, patch_width, channels))
	patches = np.append(patches, new_patch_to_add, axis = 0)

	#print('----->'+str(patches.shape))
	return patches, patches_horizontal, patches_vertical

def reconstruct_patches(patches, patches_horizontal, patches_vertical):
	"""
	Takes the blocks of patches and concatenates them together for the full image.

	return the full image
	"""
	print('>>>>'+str(patches.shape))
	print('hor'+str(patches_horizontal.shape))
	print('ver'+str(patches_vertical.shape))
	
	patch_height = 48
	patch_width = 48
	
	image_height = 584
	image_width = 565

	xl = 0
	xr = 48
	yl = 0
	yr = 48

	p = 1
	p_h = 1
	p_v = 1

	final = np.zeros((image_height,image_width))
	for i in range(0,12): # Vertical
		for j in range(0,11): # Horizontal
			final[yl:yr, xl:xr] = patches[p,:,:]
			p += 1
			xl += patch_width
			xr += patch_width
		#Fill out patch on the right
		final[yl:yr, image_width - patch_width : image_width] = patches_vertical[p_v,:,:]
		p_v += 1
		xl = 0
		xr = 48
		yl += patch_height
		yr += patch_height
	
	# Last row concat
	xl = 0
	xr = 48
	yl = image_height - 48
	yr = image_height
	for i in range(0,11):
		final[yl:yr, xl:xr] = patches_horizontal[p_h,:,:]
		p_h += 1
		xl += patch_width
		xr += patch_width
	#print('--->'+str(final.shape))
	return final
	

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
	"""
