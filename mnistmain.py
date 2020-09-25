# -*- coding: utf-8 -*-
import torch
import torchvision
import math
from mlxtend.data import loadlocal_mnist
from mnist_common import *
from matplotlib import pyplot as plt
import os

batch_size = 1000
learning_rate = 4e-4

save_model = "models/mnist-11-classifier.model"
load_model = save_model
#load_model = None

if torch.cuda.is_available():  
  device_id = "cuda:"+str(torch.cuda.device_count()-1)
else:  
  device_id = "cpu" 
#device_id = "cpu"  
print("device id:",device_id)

device = torch.device(device_id)

model = ConvNet_11()
print("network created")
if load_model!=None:
    try:
        model.load_state_dict(torch.load(load_model))
        print("loaded model from file")
    except:
        print("failed to load model. using new model")
        
model.to(device)


transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(20,expand = True),
    torchvision.transforms.RandomPerspective(),
    torchvision.transforms.RandomResizedCrop(32,scale = (0.2,1.3)),
    torchvision.transforms.ColorJitter(brightness = 0.05),
    torchvision.transforms.ToTensor()
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Pad(2),
    torchvision.transforms.ToTensor()
])
"""
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomAffine(25,(0.2,0.2),(0.5,1.4)),
    torchvision.transforms.ToTensor()
])"""

training_dataset = torchvision.datasets.MNIST("dataset/mnist_dataset", train=True, transform=transform, download=True)
training_generator = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle=True)

testing_set = torchvision.datasets.MNIST("dataset/mnist_dataset",
    train=False, transform=transform_test, download=True)
testing_generator = torch.utils.data.DataLoader(testing_set, batch_size = batch_size, shuffle=False)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()


correct_amount = 0
t=0
loss = None
total_loss = 0

while True:
    model.eval()
    with torch.set_grad_enabled(False):
        if loss != None:
            print_loss = total_loss / len(training_generator)
            total_loss = 0
            print_training_acc = (correct_amount*100.0)/len(training_dataset)
            correct_amount = 0
            
        test_correct_amount = 0
        for local_batch, local_labels in testing_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            test_y_pred = model(local_batch)
            test_y_pred_argmax = test_y_pred.argmax(1)
            test_correct_amount += (test_y_pred_argmax.eq(local_labels)).sum()
        print_testing_acc = (test_correct_amount*100.0)/len(testing_set)
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
        
    model.train()
    for local_batch, local_labels in training_generator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        y_pred = model(local_batch.to(device))
        loss = loss_fn(y_pred, local_labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_argmax = y_pred.argmax(1)
        correct_amount += (y_pred_argmax.eq(local_labels)).sum()
    
        
    t+=1