import os
import json
import random
import numpy as np
import os.path as osp
from PIL import Image
import _pickle as cPickle


import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from randaugment import RandAugmentMC




def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def build_noise_label_file(dataset_name, org_label, noise_ratio, noise_type, save_file):
    transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} 

    noise_label = []
    idx = list(range(50000))
    random.shuffle(idx)
    num_noise = int(noise_ratio * 50000)            
    noise_idx = idx[:num_noise]
    for i in range(50000):
        if i in noise_idx:
            if noise_type=='sym':
                if dataset_name =='cifar10': 
                    noiselabel = random.randint(0,9)
                elif dataset_name=='cifar100':    
                    noiselabel = random.randint(0,99)
                noise_label.append(noiselabel)
            elif noise_type=='asym':   
                noiselabel = transition[org_label[i]]
                noise_label.append(noiselabel)                    
        else:    
            noise_label.append(org_label[i])   
    print("save noisy labels to %s ..."%save_file)        
    json.dump(noise_label,open(save_file,"w"))


class cifar_dataset(Dataset): 
    def __init__(self, root_dir, dataset_name, transform, mode, noise_type = "sym", noise_ratio = 0.4, th = 0.7, consistency_score_file_path = ""): 
        """
        Args:
            root_dir : dataset path
            dataset_name : cifar10 / cifar100
            transform: data augmentation
            mode : finetune、labeled、unlabeled、test
            noise_type : sym / asym
            noise_ratio : 0.2, 0.5, 0.8, 0.9, 0.4
        
        """
        

        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode     

        # load test images
        if self.mode=='test':
            if dataset_name == 'cifar10':      
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.imgs = test_dic['data']
                self.imgs = self.imgs.reshape((10000, 3, 32, 32))
                self.imgs = self.imgs.transpose((0, 2, 3, 1))  
                self.labels = test_dic['labels']
            elif dataset_name == 'cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.imgs = test_dic['data']
                self.imgs = self.imgs.reshape((10000, 3, 32, 32))
                self.imgs = self.imgs.transpose((0, 2, 3, 1))  
                self.labels = test_dic['fine_labels']                            
        
        # load train images
        else: 
            train_data = []
            train_label = []
            if dataset_name == 'cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label += data_dic["labels"]
                train_data = np.concatenate(train_data)
            elif dataset_name == 'cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))     
            
            self.imgs, self.labels = [], []
           
            # load noisy label
            noise_file = osp.join(root_dir, str(noise_ratio) + "_" + noise_type + ".json")
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
            else:
                build_noise_label_file(dataset_name, train_label, noise_ratio, noise_type, noise_file)
                noise_label = json.load(open(noise_file,"r"))


            if self.mode == "finetune_total":
                self.imgs, self.labels = train_data, noise_label 
            else:
                pred = json.load(open(consistency_score_file_path,"r"))
                pred0 = torch.tensor(pred)
                noise_label0 = torch.tensor(noise_label).view(-1, 1).expand_as(pred0)
                score = torch.sum(pred0.eq(noise_label0), axis = 1) / pred0.size(1)                      # consistency score
                if self.mode == "labeled" or self.mode == "finetune_clean":                              # select clean and noisy label
                    for idx, value  in enumerate(score):
                        if value >= th:
                            self.imgs.append(train_data[idx])
                            self.labels.append(noise_label[idx])
                if self.mode == "unlabeled":
                    for idx, value in enumerate(score):
                        if value < th:
                            self.imgs.append(train_data[idx])
                            self.labels.append(noise_label[idx])
        
              
        print(mode,' : ', len(self.imgs))


            
              
    def __getitem__(self, index):

        img, label = self.imgs[index], self.labels[index]
        img = Image.fromarray(img)
        if self.mode == "unlabeled":
            img_w = self.transform[0](img)
            img_s = self.transform[1](img) 
            return img_w, img_s

        else:
            img = self.transform(img)
            return img, label

       
    def __len__(self):
        return len(self.imgs)
   


class cifar_dataloader():  
    
    def __init__(self, root_dir, dataset_name, num_workers, noise_type = "sym", noise_ratio = 0.5, th = 0.7, consistency_score_file_path = ""):
       
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.dataset_name = dataset_name              # cifar10 / cifar100
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        
        crop_size = 224 

        if self.dataset_name == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif self.dataset_name == "cifar100":
            mean = (0.507, 0.487, 0.441)
            std = (0.267, 0.256, 0.276)
            

    
        self.transform_labeled = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize(mean, std),                     
            ]) 
        
        self.transform_unlabeled_w = self.transform_labeled
        
        # cifar10
        if self.dataset_name == "cifar10":
            self.transform_unlabeled_s = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(crop_size),
                    # transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    RandAugmentMC(n = 2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std), 
                ])

        else:
            self.transform_unlabeled_s = transforms.Compose([
                    transforms.Resize(crop_size),
                    RandAugmentMC(n = 2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std), 
                ])

        self.transform_finetune = self.transform_labeled
        
        self.transform_test = transforms.Compose([
                transforms.Resize(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])  
        
        self.th = th
        self.consistency_score_file_path = consistency_score_file_path


    def run(self, mode, batch_size):
        if mode == "finetune_total":
            finetune_dataset = cifar_dataset(self.root_dir, self.dataset_name, self.transform_finetune, mode, self.noise_type, self.noise_ratio, self.th, self.consistency_score_file_path)
            trainloader = DataLoader(
                dataset = finetune_dataset, 
                batch_size = batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True)                 
            return trainloader
        

        if mode=='finetune_clean' or mode == 'labeled':
            labeled_dataset = cifar_dataset(self.root_dir, self.dataset_name, self.transform_labeled, mode, self.noise_type, self.noise_ratio, self.th, self.consistency_score_file_path)
            trainloader = DataLoader(
                dataset = labeled_dataset, 
                batch_size = batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True)                 
            return trainloader


        if mode == 'unlabeled':
            unlabeled_dataset = cifar_dataset(self.root_dir, self.dataset_name, [self.transform_unlabeled_w, self.transform_unlabeled_s], mode, self.noise_type, self.noise_ratio, self.th, self.consistency_score_file_path)
            trainloader = DataLoader(
                dataset = unlabeled_dataset, 
                batch_size = batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True)                 
            return trainloader


        elif mode =='test':
            test_dataset = cifar_dataset(self.root_dir, self.dataset_name, self.transform_test, mode, self.th, self.noise_type, self.noise_ratio)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return test_loader
               








