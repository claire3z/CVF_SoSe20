import numpy as np


def conv_2D(img, filter):
	"""
	Hints:
	- First, use numpy.pad to pad the image with zeros
	- You probably want to flip the filter with numpy.flip
	- You should not need more than two nested python loops. Make use of numpy matrix multiplications and numpy.sum()
	"""
	assert img.ndim == 2 and filter.ndim == 2, "Function implemented only for 2D images"
	assert filter.shape[0] == filter.shape[1], "Filter should be a square matrix"
	assert (filter.shape[0] % 2) != 0, "Filter should have an odd size"

	# Implement and return your solution here: NOTE - REMEMBER TO FLIP THE FILTER TWICE!
	k = int((filter.shape[0] - 1 )/2)
	img_pad = np.pad(img, pad_width=k, mode='constant', constant_values=0)
	filter_ = np.flip(filter, 1) #Flip an array horizontally (axis=1).
	filter_flip = np.flip(filter_,0) #Flip an array vertically (axis=0).
	output = np.zeros_like(img)

	height, width = img.shape[0], img.shape[1]
	for i in range(height):
		for j in range (width):
			output[i,j] = (img_pad[i:(i+2*k+1),j:(j+2*k+1)]*filter_flip).sum()
	return output


def conv_1D (image, filter):
	filter_flip = np.flip(filter)
	k = len(filter) // 2
	height, width = [image.shape[0]-2*k, image.shape[1]-2*k]
	output = np.zeros([height, width])
	for i in range(height):
		for j in range (width):
			output[i,j] = (image[i:(i+2*k+1),j:(j+2*k+1)]*filter_flip).sum()
	return output


def rescale(im):
	max, min = im.max(), im.min()
	out = (im - min) / (max - min)
	return out
