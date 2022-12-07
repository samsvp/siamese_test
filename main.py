#%%
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from siamese_net import SiameseNetwork
from dataset import Config, SiameseDataset 
from contrastive_loss import ContrastiveLoss


def imshow(img,text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


if __name__ == "__main__":
    # load data
    folder_dataset = datasets.ImageFolder(Config.training_dir)
    transform = transforms.Compose([
        transforms.Resize((100,100)), transforms.ToTensor()
    ])
    
    siamese_dataset = SiameseDataset(folder_dataset, transform)
    
    # train
    train_dataloader = DataLoader(
        siamese_dataset,
        shuffle=True,
        num_workers=2,
        batch_size=Config.train_batch_size)
    
    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0005)
    
    counter = []
    loss_history = []
    iteration_number= 0
    
    for epoch in range(0, Config.train_number_epochs):
        for i, data in enumerate(train_dataloader):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
                
    show_plot(counter,loss_history)

     

# %%
