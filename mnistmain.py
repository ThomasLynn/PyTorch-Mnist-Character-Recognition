# -*- coding: utf-8 -*-
import torch
import math
from mlxtend.data import loadlocal_mnist
from mnist_common import *
from image_distorter import image_distorter
from matplotlib import pyplot as plt
import os

batch_size = 4000
learning_rate = 1e-3

save_model = "mnist-6-classifier.model"
load_model = save_model
#load_model = None

if torch.cuda.is_available():  
  device_id = "cuda:0" 
else:  
  device_id = "cpu" 
#device_id = "cpu"  
print("device id:",device_id)

device = torch.device(device_id)

model = ConvNet_6()
print("network created")
if load_model!=None:
    try:
        model.load_state_dict(torch.load(load_model))
        print("loaded model from file")
    except:
        print("failed to load model. using new model")
        
model.to(device)

x_data, y_data = loadlocal_mnist(
    images_path='train-images.idx3-ubyte', 
    labels_path='train-labels.idx1-ubyte')
x = torch.tensor(x_data, dtype=torch.float32)/255.0
x = x.reshape(x.shape[0],1,28,28).to(device)
y = torch.tensor(y_data,dtype = torch.int64).to(device)

test_x_data, test_y_data = loadlocal_mnist(
    images_path='t10k-images.idx3-ubyte', 
    labels_path='t10k-labels.idx1-ubyte')
test_x = torch.tensor(test_x_data, dtype=torch.float32)/255.0
test_x = test_x.reshape(test_x.shape[0],1,28,28)[:batch_size].to(device)
test_y = torch.tensor(test_y_data, dtype=torch.int64)[:batch_size].to(device)




loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

t=0
loss = None
while True:
    #timer = time.time()
    if t % 10 == 0:
        with torch.set_grad_enabled(False):
            model.eval()
            if loss != None:
                print_loss = loss.item()
                print_training_acc = (correct_amount*100.0)/len(y)
                
            test_y_pred = model(test_x)
            test_y_pred_argmax = test_y_pred.argmax(1)
            test_correct_amount = (test_y_pred_argmax.eq(test_y)).sum()
            print_testing_acc = (test_correct_amount*100.0)/len(test_y_pred_argmax)
            if loss == None:
                print("epoch:",t,"testing acc: {:.2f}%".format(print_testing_acc))
            else:
                print("epoch:",t,"loss: {:.5f}".format(print_loss),
                    "train acc: {:.2f}%".format(print_training_acc),
                    "testing acc: {:.2f}%".format(print_testing_acc))
        if save_model!=None:
            
            try:
                os.remove(save_model)
            except:
                print("no model to delete")
            torch.save(model.state_dict(), save_model)
    
    correct_amount = 0
    model.train()
    for i in range(int(x.shape[0]/batch_size)):
        #print(i)
        
        #images = image_distorter(x[batch_size*i:batch_size*(i+1)],30,5,10)
        
        #for j in range(3):
        #    pixels = images[j][0].reshape((28, 28))
        #    plt.imshow(pixels*255, cmap='gray')
        #    plt.show()
        #y_pred = model(images)
        #y_pred = model(x)
        y_pred = model(x[batch_size*i:batch_size*(i+1)])
        loss = loss_fn(y_pred, y[batch_size*i:batch_size*(i+1)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_argmax = y_pred.argmax(1)
        correct_amount += (y_pred_argmax.eq(y[i*batch_size:(i+1)*batch_size])).sum()
    
        
    t+=1