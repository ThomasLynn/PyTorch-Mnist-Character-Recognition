from mlxtend.data import loadlocal_mnist
import numpy as np
from image_distorter import *

x_data, y_data = loadlocal_mnist(
    images_path='train-images.idx3-ubyte', 
    labels_path='train-labels.idx1-ubyte')
x_data = x_data.reshape(x_data.shape[0],1,28,28)/255.0
y_data = y_data.astype(np.int64)

print("shapes:",x_data.shape,y_data.shape)

x = x_data
y = y_data
for i in range(99):
	x = np.append(x,image_distorter(x_data,30,5,10))
	y = np.append(y,y_data)
	
	
print("final shapes:",x.shape,y)
	
np.save("train_images",x)
np.save("train_labels",y)