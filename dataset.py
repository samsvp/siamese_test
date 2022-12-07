import PIL
import torch
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

from typing import *



class Config():
    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    train_batch_size = 16
    train_number_epochs = 20


class SiameseDataset(Dataset):
    def __init__(self, image_folder_dataset: datasets.ImageFolder, transform: transforms=None) -> None:
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        
    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        img_tuple = random.choice(self.image_folder_dataset.imgs)
        
        get_same_class = random.random() < 0.5
        if get_same_class:
            while True:
                #keep looping till the same class image is found
                comp_img_tuple = random.choice(self.image_folder_dataset.imgs) 
                if img_tuple[1]==comp_img_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                comp_img_tuple = random.choice(self.image_folder_dataset.imgs) 
                if img_tuple[1] !=comp_img_tuple[1]:
                    break
                
        img0 = Image.open(img_tuple[0])
        img1 = Image.open(comp_img_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            
        return img0, img1, torch.tensor([float(get_same_class)])
    
    
    def __len__(self):
        return len(self.image_folder_dataset.imgs)
