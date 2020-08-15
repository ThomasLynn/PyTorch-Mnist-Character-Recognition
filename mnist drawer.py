import pygame
import torch
from mnist_common import *

pygame.font.init()
FONT = pygame.font.SysFont('Comic Sans MS', 14)

model = ConvNet_3()
model.load_state_dict(torch.load("mnist-3-classifier.model"))
model.eval()

screen = pygame.display.set_mode((600, 300))
pygame.init()
clock = pygame.time.Clock()

guesses = torch.zeros(10)
image = torch.zeros((28,28))
scale = 10

def draw_to_image(set_to,prev_pos,pos):
    #y = pos[0]
    #x = pos[1]
    dist = int(((prev_pos[0]-pos[0])**2 + (prev_pos[1]-pos[1])**2) ** 0.5)
    for i in range(dist + 1):
        lerp = i/(dist + 1.0)
        y = (pos[0]-prev_pos[0])* lerp + prev_pos[0]
        x = (pos[1]-prev_pos[1])* lerp + prev_pos[1]
        if x/scale < 28 and x>=0 and y/scale <28 and y>=0:
            if x/scale+1<28:
                if image[int(y/scale)][int(x/scale)+1]<set_to/2:
                    image[int(y/scale)][int(x/scale)+1] = set_to/2
            if x-1 >=0:
                if image[int(y/scale)][int(x/scale)-1]<set_to/2:
                    image[int(y/scale)][int(x/scale)-1] = set_to/2
            if y/scale+1<28:
                if image[int(y/scale)+1][int(x/scale)]<set_to/2:
                    image[int(y/scale)+1][int(x/scale)] = set_to/2
            if y-1 >=0:
                if image[int(y/scale)-1][int(x/scale)]<set_to/2:
                    image[int(y/scale)-1][int(x/scale)] = set_to/2
            image[int(y/scale)][int(x/scale)] = set_to
    guesses[:] = model(image.reshape(1,1,28,28))
        

running = True
mouse_pos = pygame.mouse.get_pos()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
            
    mouse_pressed = pygame.mouse.get_pressed()
    
    mouse_prev_pos = mouse_pos
    mouse_pos = pygame.mouse.get_pos()
    #print(mouse_pressed)
    if mouse_pressed[0]==1:
        draw_to_image(1,mouse_prev_pos,mouse_pos)
    elif mouse_pressed[2] == 1:
        draw_to_image(0,mouse_prev_pos,mouse_pos)
    
    key = pygame.key.get_pressed()
    if key[pygame.K_SPACE]:
        for i in range(28):
            image = torch.zeros((28,28))
            guesses = torch.zeros(10)
    
    screen.fill([255, 255, 255])
    #print(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            val = min(255,max(0,image[i][j]*255.0))
            pygame.draw.rect(screen,(val,val,val),
                (i*scale,j*scale,scale,scale))
    for i in range(guesses.shape[0]):
        text_surface = FONT.render(str(i)+': {:.5f}'.format(guesses[i]), False, (0, 0, 0))
        screen.blit(text_surface,((300,10+i*16)))
            
    text_surface = FONT.render("guess: "+str(int(torch.argmax(guesses))), False, (0, 0, 0))
    screen.blit(text_surface,((400,10)))
    
    pygame.display.update()
    clock.tick(30)