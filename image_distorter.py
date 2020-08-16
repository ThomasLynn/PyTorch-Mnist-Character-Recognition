import torch
import random
import numpy as np

def roll_zeropad(a_in, shift, axis):
    if shift == 0: return a_in
    a = a_in.numpy()
    n = a.shape[axis]
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    return torch.from_numpy(res)


def image_distorter(images, r_scale, r_rotation, r_translation, r_s_noise, r_l_noise):
    new_images = images.clone()
    
    x_delta = random.randint(-r_scale,r_scale)
    y_delta = random.randint(-r_scale,r_scale)
    
    new_images = roll_zeropad(new_images,x_delta,3)
    new_images = roll_zeropad(new_images,y_delta,2)
    
    return new_images