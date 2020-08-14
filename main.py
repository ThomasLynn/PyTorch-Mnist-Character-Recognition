# -*- coding: utf-8 -*-
import torch
import pygame
import math
from nnfs.datasets import spiral_data

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = 64
D_in = 2
H = 300
H2 = 300
H3 = 300
D_out = 3

# Create random Tensors to hold inputs and outputs
##x = torch.randn(N, D_in)
x_data, y_data = spiral_data(300, 3)  
x = torch.tensor(x_data, dtype=torch.float32)
y_list = []
for w in y_data:
    out = [0.0]*3
    out[w] = 1.0
    y_list.append(out)
y = torch.tensor(y_list,dtype = torch.float32)
x2_data = []
for i in range(41):
    for j in range(41):
        x2_data.append([i/20.0 - 1, j/20.0 - 1])
x2 = torch.tensor(x2_data)
#x = x.reshape((len(x),1))
##y = torch.randn(N, D_out)
#y = torch.sin(x*10)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2, H3),
    torch.nn.ReLU(),
    torch.nn.Linear(H3, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

screen = pygame.display.set_mode((600, 300))
pygame.init()
clock = pygame.time.Clock()

#for t in range(50000):
t=0
running = True

def get_color(vals):
    index = torch.argmax(vals)
    if index == 0:
        return (255,0,0)
    elif index == 1:
        return (0,255,0)
    elif index == 2:
        return (0,0,255)
    return (100,100,100)
        

while running:
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        print(t, loss.item()/len(x))
        
        screen.fill([255, 255, 255])
        for i in range(1,x.shape[0]):
            pygame.draw.circle(screen,get_color(y[i]),
                (int(x[i][0]*125)+150,int(x[i][1]*125)+150),1)
            pygame.draw.circle(screen,get_color(y_pred[i]),
                (int(x[i][0]*125)+450,int(x[i][1]*125)+150),1)
            
        y_pred2 = model(x2)
        for i in range(len(y_pred2)):
            pygame.draw.circle(screen,get_color(y_pred2[i]),
                (int(x2[i][0]*125)+450,int(x2[i][1]*125)+150),1)
        pygame.display.update()
        clock.tick(30)
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    t+=1