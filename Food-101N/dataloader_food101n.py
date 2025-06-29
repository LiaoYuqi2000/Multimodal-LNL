import os
import json
import random
import numpy as np
import os.path as osp
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from randaugment import RandAugmentMC


class food101n_dataset(Dataset): 
    def __init__(self, root_train_dir, root_test_dir, transform, mode, th=0.7, consistency_score_file_path="", ood_file_path=""): 

        self.root_train_dir = root_train_dir
        self.root_test_dir = root_test_dir
        self.transform = transform
        self.mode = mode    

        
        self.imgs = []
        self.labels = []

        class2idx = {}
        with open(osp.join(root_test_dir, "meta/classes.txt")) as f:
            lines = f.readlines()
            for idx, class_name in enumerate(lines):
                class2idx[class_name.strip()] = idx

        
        if self.mode=='test':
            with open(osp.join(root_test_dir, "meta/test.txt"), 'r') as f:
                for img_name in f.readlines():
                    img = img_name.strip() + ".jpg"
                    label = class2idx[img_name.split('/')[0]]
                    self.imgs.append(img)
                    self.labels.append(label)

        else:
            print(mode)
            img_list = []
            label_list = []
            with open(osp.join(root_train_dir, "meta/imagelist.tsv"), 'r') as f:
                for img_name in f.readlines()[1:]:
                    img = img_name.strip()
                    label = class2idx[img_name.split('/')[0]]
                    img_list.append(img)
                    label_list.append(label)
            
            with open(consistency_score_file_path, 'r') as f:
                img_score_dict = json.load(f)
            
            ood_list = json.load(open(ood_file_path, 'r'))
        
            if mode == "finetune_total":
                self.imgs = img_list
                self.labels = label_list
            if mode == "finetune_wo_ood":
                print("ood:", len(ood_list))
                for img, label in zip(img_list, label_list):
                    if img not in ood_list:
                        self.imgs.append(img)
                        self.labels.append(label)
            elif mode == "finetune_clean" or mode == "labeled":
                for img, label in zip(img_list, label_list):
                    if img_score_dict[img] >= th and img not in ood_list:
                        self.imgs.append(img)
                        self.labels.append(label)
            elif mode == "unlabeled":
                for img, label in zip(img_list, label_list):
                    if img_score_dict[img] < th and img not in ood_list:
                        self.imgs.append(img)
                        self.labels.append(label)
                        
        print(mode,' : ', len(self.imgs))


            
              
    def __getitem__(self, index):

        img_path, label = self.imgs[index], self.labels[index]
        if self.mode == "test":
            img = Image.open(osp.join(self.root_test_dir, "images", img_path)).convert('RGB')
        else:
            img = Image.open(osp.join(self.root_train_dir, "images", img_path)).convert('RGB')

        if self.mode == "unlabeled":
            img_w = self.transform[0](img)
            img_s = self.transform[1](img) 
            return img_w, img_s
        else:
            img = self.transform(img)
            return img, label

       
    def __len__(self):
        return len(self.imgs)
   


class food101n_dataloader():  
    
    def __init__(self, root_train_dir, root_test_dir, num_workers, th = 0.7, consistency_score_file_path = "", ood_file_path=""):
       
        self.num_workers = num_workers
        self.root_train_dir = root_train_dir
        self.root_test_dir = root_test_dir

        
        image_resize = 256 
        crop_size = 224 

        mean = (0.6959, 0.6537, 0.6371)
        std = (0.3113, 0.3192, 0.3214)

    
        self.transform_labeled = transforms.Compose([
                transforms.Resize(image_resize),
                transforms.RandomCrop(crop_size),
                # transforms.RandomCrop(size = crop_size, padding = int(image_resize*0.125), padding_mode = 'reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize(mean, std),                     
            ]) 
        
        self.transform_unlabeled_w = self.transform_labeled
        self.transform_finetune = self.transform_labeled
        
        self.transform_unlabeled_s = transforms.Compose([
                transforms.Resize(image_resize),
                transforms.RandomCrop(crop_size),
                # transforms.RandomResizedCrop(crop_size),
                # transforms.RandomCrop(size = crop_size, padding = int(image_resize*0.125), padding_mode = 'reflect'),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n = 2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std), 
            ])

        self.transform_test = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])  
        
        self.th = th
        self.consistency_score_file_path = consistency_score_file_path
        self.ood_file_path = ood_file_path


    def run(self, mode, batch_size):
        if mode=='finetune_clean' or mode == "finetune_total" or mode == "finetune_wo_ood":
            finetune_dataset = food101n_dataset(self.root_train_dir, self.root_test_dir, self.transform_finetune, mode, self.th, self.consistency_score_file_path, self.ood_file_path)
            trainloader = DataLoader(
                dataset = finetune_dataset, 
                batch_size = batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True)                 
            return trainloader
        

        if mode == 'labeled':
            labeled_dataset = food101n_dataset(self.root_train_dir, self.root_test_dir, self.transform_labeled, mode, self.th, self.consistency_score_file_path, self.ood_file_path)
            trainloader = DataLoader(
                dataset = labeled_dataset, 
                batch_size = batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True)                 
            return trainloader


        if mode == 'unlabeled':
            unlabeled_dataset = food101n_dataset(self.root_train_dir, self.root_test_dir, [self.transform_unlabeled_w, self.transform_unlabeled_s], mode, self.th, self.consistency_score_file_path, self.ood_file_path)
            trainloader = DataLoader(
                dataset = unlabeled_dataset, 
                batch_size = batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True)                 
            return trainloader


        elif mode =='test':
            test_dataset = food101n_dataset(self.root_train_dir, self.root_test_dir, self.transform_test, mode, self.th, self.consistency_score_file_path, self.ood_file_path)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return test_loader
               








