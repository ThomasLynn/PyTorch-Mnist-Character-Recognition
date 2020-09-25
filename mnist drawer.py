import pygame
import torch
from mnist_common import *
import time

pygame.font.init()
FONT = pygame.font.SysFont('Comic Sans MS', 16)

if torch.cuda.is_available():
  device_id = "cuda:"+str(torch.cuda.device_count()-1)
else:  
  device_id = "cpu" 
#device_id = "cpu"  
print("device id:",device_id)

device = torch.device(device_id)

image_size = 32
scale = 10

model = ConvNet_11()
model.load_state_dict(torch.load("models/mnist-11-classifier.model"))
model.to(device)
model.eval()

screen = pygame.display.set_mode((image_size*scale + 200, image_size*scale+20))
pygame.init()
clock = pygame.time.Clock()

guesses = torch.zeros(10)
image = torch.zeros((image_size,image_size))

softmax = torch.nn.Softmax(1)

def draw_pixel(image,x,y,set_to):
    if x < image_size and x>=0 and y <image_size and y>=0:
        if set_to == 0 or image[int(x)][int(y)]<set_to:
            image[int(x)][int(y)] = set_to

def draw_to_image(set_to,prev_pos,pos):
    #y = pos[0]
    #x = pos[1]
    dist = int(((prev_pos[0]-pos[0])**2 + (prev_pos[1]-pos[1])**2) ** 0.5)
    for i in range(dist + 1):
        lerp = i/(dist + 1.0)
        y = (pos[0]-prev_pos[0])* lerp + prev_pos[0]
        x = (pos[1]-prev_pos[1])* lerp + prev_pos[1]
        draw_pixel(image,x/scale,y/scale,set_to)
        
        draw_pixel(image,x/scale,y/scale-1,set_to/2)
        draw_pixel(image,x/scale,y/scale+1,set_to/2)
        draw_pixel(image,x/scale+1,y/scale,set_to/2)
        draw_pixel(image,x/scale-1,y/scale,set_to/2)
        
        draw_pixel(image,x/scale-1,y/scale-1,set_to/3)
        draw_pixel(image,x/scale-1,y/scale+1,set_to/3)
        draw_pixel(image,x/scale+1,y/scale-1,set_to/3)
        draw_pixel(image,x/scale+1,y/scale+1,set_to/3)
        
        draw_pixel(image,x/scale,y/scale-2,set_to/3)
        draw_pixel(image,x/scale,y/scale+2,set_to/3)
        draw_pixel(image,x/scale+2,y/scale,set_to/3)
        draw_pixel(image,x/scale-2,y/scale,set_to/3)
    guesses[:] = softmax(model(image.reshape(1,1,image_size,image_size).to(device)))
    #guesses[:] = model(image.reshape(1,1,image_size,image_size))
        

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
        for i in range(image_size):
            image = torch.zeros((image_size,image_size))
            guesses = torch.zeros(10)
    
    screen.fill([255, 255, 255])
    #print(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            val = min(255,max(0,image[j][i]*255.0))
            pygame.draw.rect(screen,(val,val,val),
                (i*scale,j*scale,scale,scale))
    for i in range(guesses.shape[0]):
        text_surface = FONT.render(str(i)+': {:.1f}%'.format(guesses[i]*100), False, (0, 0, 0))
        screen.blit(text_surface,((image_size*scale + 30,30+i*18)))
            
    text_surface = FONT.render("guess: "+str(int(torch.argmax(guesses))), False, (0, 0, 0))
    screen.blit(text_surface,((image_size*scale + 30,10)))
    
    pygame.display.update()
    clock.tick(30)