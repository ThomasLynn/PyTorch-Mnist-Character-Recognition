import numpy as np
import torch
import os

class Mnist_Dataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, images_base, labels_name):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = torch.from_numpy(np.load(labels_name+".npy"))
        self.image_files = []
        try:
            self.image_files.append(images_base+".npy")
            
        except:
            print("couldn't load base images file")
        counter = 0
        while True:
            print("adding path")
            counter+=1
            file_name = images_base+"_dist"+str(counter)+".npy"
            if os.path.isfile(file_name):
                self.image_files.append(file_name)
            else:
                break
        self.loaded_image_set_id = 0
        print("loading:",self.image_files[self.loaded_image_set_id])
        self.loaded_image_set = np.load(self.image_files[self.loaded_image_set_id])

    def __len__(self):
        return len(self.image_files)*60_000

    def __getitem__(self, index):
        #if index%10000==0:
        #    print("index = ",index)
        'Generates one sample of data'
        # Select sample
        set_id = int(index / 60_000)
        if set_id != self.loaded_image_set_id:
            print("changing file")
            self.loaded_image_set_id = set_id
            self.loaded_image_set = np.load(self.image_files[self.loaded_image_set_id])
            
        individual_id = index % 60_000
        # Load data and get label
        X = self.loaded_image_set[individual_id]
        y = self.labels[individual_id]

        return X, y