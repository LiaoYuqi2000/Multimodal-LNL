from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
import os.path as osp
import json
import open_clip
from randaugment import RandAugmentMC
from auglib import Augment







class animal_dataset(Dataset): 

    def __init__(self, root_dir, transform, mode, th=0.7, consistency_score_file_path=""): 
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode 

        self.imgs = []
        self.labels = {}

        if self.mode == "test":
            for img_name in os.listdir(osp.join(root_dir, "testing")):
                img_path = osp.join("testing", img_name)
                self.imgs.append(img_path)
                self.labels[img_path] = int(img_name.split('_')[0])
        elif self.mode == "finetune_total":
                for img_name in os.listdir(osp.join(root_dir, "training")):
                    img_path = osp.join("training", img_name)        
                    self.imgs.append(img_path)
                    self.labels[img_path] = int(img_name.split('_')[0])
        else:
            score = json.load(open(consistency_score_file_path, "r"))     # {img_path: score}
            if self.mode == "finetune_clean" or self.mode == "labeled":
                for img_name in os.listdir(osp.join(root_dir, "training")):
                    img_path = osp.join("training", img_name)
                    if score[img_path] >= th:         
                        self.imgs.append(img_path)
                        self.labels[img_path] = int(img_name.split('_')[0])
            elif self.mode == "unlabeled":
                for img_name in os.listdir(osp.join(root_dir, "training")):
                    img_path = osp.join("training", img_name)
                    if score[img_path] < th:         
                        self.imgs.append(img_path)
                        self.labels[img_path] = int(img_name.split('_')[0])                

        print(mode, " : ", len(self.imgs))       


    def __getitem__(self, index):

        img_path = self.imgs[index]
        label = self.labels[img_path]
        
        if self.mode == "unlabeled":
            img = Image.open(osp.join(self.root_dir, img_path)).convert('RGB')
            img_w = self.transform[0](img)
            img_s = self.transform[1](img)
            return img_w, img_s

        else:
            img = Image.open(osp.join(self.root_dir, img_path)).convert('RGB')
            img = self.transform(img)
            return img, label

       
    def __len__(self):
        return len(self.imgs)
   


class animal_dataloader():  
    def __init__(self, root_dir, num_workers, th = 0.7, consistency_score_file_path = ""):

        
        self.num_workers = num_workers
        self.root_dir = root_dir

        image_resize = 256 
        crop_size = 224 

     
        mean=(0.48145466, 0.4578275, 0.40821073)
        std=(0.26862954, 0.26130258, 0.27577711)


        self.transform_labeled = transforms.Compose([
                transforms.Resize(image_resize),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize(mean, std),                     
            ]) 
        
        self.transform_unlabeled_w = self.transform_labeled

        
        self.transform_unlabeled_s = transforms.Compose([
                transforms.Resize(image_resize),
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n = 3, m = 10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std), 
            ])

        self.transform_finetune = self.transform_labeled

        self.transform_test = transforms.Compose([
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])  
    
        self.th = th
        self.consistency_score_file_path = consistency_score_file_path

    def run(self, mode, batch_size):
    
        self.batch_size = batch_size

        if mode == 'finetune_total':
            finetune_dataset = animal_dataset(root_dir=self.root_dir, transform=self.transform_labeled, mode=mode, th=self.th, consistency_score_file_path=self.consistency_score_file_path)                
            trainloader = DataLoader(
                dataset = finetune_dataset, 
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True)                 
            return trainloader
        

        if mode == 'finetune_clean' or mode == 'labeled':
            labeled_dataset = animal_dataset(root_dir=self.root_dir, transform=self.transform_labeled, mode=mode, th=self.th, consistency_score_file_path=self.consistency_score_file_path)                
            trainloader = DataLoader(
                dataset = labeled_dataset, 
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True)                 
            return trainloader


        if mode == 'unlabeled':
            unlabeled_dataset = animal_dataset(root_dir=self.root_dir, transform = [self.transform_unlabeled_w, self.transform_unlabeled_s], mode=mode, th=self.th, consistency_score_file_path=self.consistency_score_file_path)                
            trainloader = DataLoader(
                dataset = unlabeled_dataset, 
                batch_size = self.batch_size,   
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True)                 
            return trainloader


        elif mode =='test' or mode == 'val':
            test_dataset = animal_dataset(root_dir=self.root_dir, transform=self.transform_test, mode = mode, th=self.th, consistency_score_file_path=self.consistency_score_file_path)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return test_loader
               








