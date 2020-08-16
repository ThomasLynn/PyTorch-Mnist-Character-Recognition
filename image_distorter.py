import torch
import random
import numpy as np
from scipy import ndimage

def image_distorter(images, r_rotation, r_translation, r_l_noise):
    new_images = (images.clone()).numpy()
    
    new_images = ndimage.rotate(new_images, -r_rotation+random.random()*2*r_rotation, axes = (3,2), reshape=False, order = 1)
    
    x_delta = random.randint(-r_translation,r_translation)
    y_delta = random.randint(-r_translation,r_translation)
    
    new_images = ndimage.shift(new_images,[0,0,y_delta,x_delta],order = 0)
    
    l_noise = np.zeros((new_images.shape[0],new_images.shape[1],new_images.shape[2],new_images.shape[3]))
    for j in range(l_noise.shape[0]):
        for i in range(r_l_noise):
            l_noise[j][0][random.randint(0,l_noise.shape[2]-1)]\
                [random.randint(0,l_noise.shape[3]-1)] = random.random()

    new_images+=l_noise
    
    new_images = np.clip(new_images,0,1)
    
    return torch.from_numpy(new_images)