# -*- coding: utf-8 -*-
import torch
import math
from mlxtend.data import loadlocal_mnist
from mnist_common import *
from image_distorter import image_distorter
from matplotlib import pyplot as plt
import os
from mnist_dataset import Mnist_Dataset

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

training_set = Mnist_Dataset("dataset/train_images", "dataset/train_labels",10_000)

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
    if t % 1 == 0:
        with torch.set_grad_enabled(False):
            model.eval()
            if loss != None:
                print_loss = loss.item()
                print_training_acc = (correct_amount*100.0)/len(training_set)
                
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
    for local_batch, local_labels in training_set:
        local_batch, local_labels = torch.tensor(local_batch).to(device), torch.tensor(local_labels).to(device)
        y_pred = model(local_batch)
        loss = loss_fn(y_pred, local_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_argmax = y_pred.argmax(1)
        correct_amount += (y_pred_argmax.eq(local_labels)).sum()
    
        
    t+=1