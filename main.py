# -*- coding: utf-8 -*-
import torch
import pygame
import math

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1, 100, 1

# Create random Tensors to hold inputs and outputs
#x = torch.randn(N, D_in)
x = torch.arange(-1,1.05,0.1)
x = x.reshape((len(x),1))
#y = torch.randn(N, D_out)
y = torch.sin(x*10)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

screen = pygame.display.set_mode((300, 300))
pygame.init()
clock = pygame.time.Clock()

#for t in range(50000):
t=0
running = True
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
        print(t, loss.item())
        
        screen.fill([255, 255, 255])
        for i in range(1,x.shape[0]):
            pygame.draw.line(screen,(0,255,0),
                (int(x[i-1][0]*125)+150,int(y[i-1][0]*125)+150),
                (int(x[i][0]*125)+150,int(y[i][0]*125)+150),1)
            pygame.draw.line(screen,(0,0,255),
                (int(x[i-1][0]*125)+150,int(y_pred[i-1][0]*125)+150),
                (int(x[i][0]*125)+150,int(y_pred[i][0]*125)+150),1)
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