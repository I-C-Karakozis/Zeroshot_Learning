import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

from tools.awa import AWA

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_THREADS = 4
IMAGE_SIZE = 224

# ---------------------------- Data Preprocessing ---------------------------- #

def transform_train():
    return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, 
                                 std=IMAGENET_STD),
        ])

def transform_test():
    return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, 
                                 std=IMAGENET_STD),
            ])

# ---------------------------- Data Loading ------------------------ #

def view_dataset(dataset):
    for idx, (img_path, lbl) in enumerate(zip(dataset.img_paths, dataset.lbls)):
        # get labels
        label = ""
        for i, l in enumerate(lbl[:-1]):
            if l == 1: label += ("_" + dataset.id2attribute[i])

        # show image
        if idx % 500 == 0: 
            print(label)
            b = Image.open(img_path).convert("RGB")
            b.save('test/{}{}.jpg'.format(idx, dataset.classes[lbl[-1]]))

def get_dataloaders(batch_size, dataset_train, dataset_test):
    # prepare dataloaders
    trainloader  = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                                               shuffle=True,  num_workers=NUM_THREADS)  
    testloader   = torch.utils.data.DataLoader(dataset_test,  batch_size=batch_size, 
                                               shuffle=False, num_workers=NUM_THREADS)

    return trainloader, testloader

def get_data(datadir):
    dataset_train = AWA(split=0, transform=transform_train(), datadir=datadir)
    dataset_test  = AWA(split=1, transform=transform_test(), datadir=datadir)

    return dataset_train, dataset_test
