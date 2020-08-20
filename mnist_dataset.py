import numpy as np
import torch
import os

class Mnist_Dataset():

    def __init__(self, images_base, labels_name, batch_size):
        self.index = 0
        self.batch_size = batch_size
        self.labels = torch.from_numpy(np.load(labels_name+".npy"))
        self.image_files = []
        try:
            self.image_files.append(images_base+".npy")
            
        except:
            print("couldn't load base images file")
        counter = 0
        while True:
            counter+=1
            file_name = images_base+"_dist"+str(counter)+".npy"
            if os.path.isfile(file_name):
                self.image_files.append(file_name)
            else:
                break
        self.loaded_image_set_id = 0
        self.loaded_image_set = np.load(self.image_files[self.loaded_image_set_id])

    def __len__(self):
        return len(self.image_files)*60_000
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.__len__():
            self.index -= self.__len__()
            raise StopIteration
        
        set_id = int(self.index / 60_000)
        if set_id != self.loaded_image_set_id:
            self.loaded_image_set_id = set_id
            self.loaded_image_set = np.load(self.image_files[self.loaded_image_set_id])
            
        start_id = self.index % 60_000
        stop_id = start_id + self.batch_size
        
        Xs = self.loaded_image_set[start_id:min(stop_id,len(self.loaded_image_set))]
        if stop_id>60_000:
            start_id2=0
            stop_id2=stop_id - 60_000
            self.loaded_image_set_id += 1
            try:
                self.loaded_image_set = np.load(self.image_files[self.loaded_image_set_id%len(self.image_files)])
                Xs = np.concatenate([Xs, self.loaded_image_set[start_id2:stop_id2]])
            except:
                print("can't load next data")
                
            ys = self.labels[start_id:min(stop_id,len(self.loaded_image_set))]
            ys = np.concatenate([ys, self.labels[:stop_id2]])
        else:
            ys = self.labels[start_id:stop_id]
        
        
        self.index += self.batch_size
        return Xs, ys