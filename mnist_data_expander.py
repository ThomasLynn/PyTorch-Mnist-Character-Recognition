from mlxtend.data import loadlocal_mnist
import numpy as np
from image_distorter import *
import os

starting_number = 1

x_data, y_data = loadlocal_mnist(
    images_path='train-images.idx3-ubyte', 
    labels_path='train-labels.idx1-ubyte')
x_data = x_data.reshape(x_data.shape[0],1,28,28)/255.0
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.int64)

print("shapes:",x_data.shape,y_data.shape)
try:
	os.mkdir("dataset")
except:
	pass
try:
	np.save("dataset/train_images",x_data)
	np.save("dataset/train_labels",y_data)
except:
	pass

x = x_data
for i in range(20):
	try:
		to_add = image_distorter(x_data,20,5,15)
		np.save("dataset/train_images_dist"+str(i+starting_number),to_add)
		print("adding",to_add.shape,i+starting_number)
	except:
		pass
	