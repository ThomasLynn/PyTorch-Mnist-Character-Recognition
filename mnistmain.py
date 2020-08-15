# -*- coding: utf-8 -*-
import torch
import math
from mlxtend.data import loadlocal_mnist

D_in = 28*28
H = 300
H2 = 40
D_out = 10

load_model = "mnist-300-40-classifier.model"
save_model = "mnist-300-40-classifier.model"

# Create random Tensors to hold inputs and outputs
x_data, y_data = loadlocal_mnist(
    images_path='train-images.idx3-ubyte', 
    labels_path='train-labels.idx1-ubyte')
x = torch.tensor(x_data, dtype=torch.float32)
y = torch.tensor(y_data,dtype = torch.int64)

test_x_data, test_y_data = loadlocal_mnist(
    images_path='t10k-images.idx3-ubyte', 
    labels_path='t10k-labels.idx1-ubyte')
test_x = torch.tensor(test_x_data, dtype=torch.float32)
test_y = torch.tensor(test_y_data, dtype=torch.int64)


model = torch.nn.Sequential(
    #torch.nn.Conv2d(1, H, kernel_size=5, stride=1, padding=2),
    #torch.nn.ReLU(),
    #torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Linear(D_in, H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, H2),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H2, D_out)
)
print("loaded")
if load_model!=None:
    try:
        model.load_state_dict(torch.load(load_model))
        model.eval()
    except:
        print("failed to load model. using new model")
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

t=0

while True:
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        
        print_loss = loss.item()
        test_y_pred = model(test_x)
        test_loss = loss_fn(test_y_pred,test_y)
        y_pred_argmax = y_pred.argmax(1)
        correct_amount = 0
        for i in range(len(y_pred_argmax)):
            if y_pred_argmax[i]==y_data[i]:
                correct_amount += 1
        print_training_acc = (correct_amount*100.0)/len(y_pred_argmax)
        test_y_pred_argmax = test_y_pred.argmax(1)
        correct_amount = 0
        for i in range(len(test_y_pred_argmax)):
            if test_y_pred_argmax[i]==test_y_data[i]:
                correct_amount += 1
        print_testing_acc = (correct_amount*100.0)/len(test_y_pred_argmax)
        print("epoch:",t,"loss: {:.5f}".format(print_loss),
            "train acc: {:.2f}%".format(print_training_acc),
            "testing acc: {:.2f}%".format(print_testing_acc))
        if save_model!=None:
            torch.save(model.state_dict(), save_model)
        
    t+=1