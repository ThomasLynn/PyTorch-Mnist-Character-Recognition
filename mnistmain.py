# -*- coding: utf-8 -*-
import torch
import pygame
import math
from mlxtend.data import loadlocal_mnist

#D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in = 28*28
H = 16
H2 = 16
D_out = 10

# Create random Tensors to hold inputs and outputs
##x = torch.randn(N, D_in)
x_data, y_data = loadlocal_mnist(
    images_path='train-images.idx3-ubyte', 
    labels_path='train-labels.idx1-ubyte')
x = torch.tensor(x_data, dtype=torch.float32)
y_data_onehot = []
for i in range(len(y_data)):
    to_add = [0]*10
    to_add[y_data[i]] = 1
    y_data_onehot.append(to_add)
y = torch.tensor(y_data_onehot,dtype = torch.float32)
#print("train",x,y)

test_x_data, test_y_data = loadlocal_mnist(
    images_path='t10k-images.idx3-ubyte', 
    labels_path='t10k-labels.idx1-ubyte')
test_x = torch.tensor(test_x_data, dtype=torch.float32)
test_y_data_onehot = []
for i in range(len(test_y_data)):
    to_add = [0.0]*10
    to_add[test_y_data[i]] = 1.0
    test_y_data_onehot.append(to_add)
test_y = torch.tensor(test_y_data_onehot, dtype=torch.float32)
#print("test",test_x,test_y)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, H2),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H2, D_out)
)
loss_fn = torch.nn.MSELoss(reduction='mean')
#loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
learning_rate = 1e-3
#learning_rate = 0.5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

t=0

while True:
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 20 == 19:
        print(t, loss.item())
        test_y_pred = model(test_x)
        test_loss = loss_fn(test_y_pred,test_y)
        y_pred_argmax = y_pred.argmax(1)
        correct_amount = 0
        
        print(y_pred,"yeet",y_pred_argmax,"yote",y_data)
        for i in range(len(y_pred_argmax)):
            if y_pred_argmax[i]==y_data[i]:
                correct_amount += 1
        print("train accuracy:",(correct_amount*100.0)/(len(y_pred_argmax)),"%")
        test_y_pred_argmax = test_y_pred.argmax(1)
        correct_amount = 0
        for i in range(len(test_y_pred_argmax)):
            if test_y_pred_argmax[i]==test_y_data[i]:
                correct_amount += 1
        print("test accuracy:",(correct_amount*100.0)/(len(test_y_pred_argmax)),"%")
        
    t+=1