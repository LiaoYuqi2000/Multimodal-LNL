#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import numpy as np 
from torchvision.transforms import transforms



class Augment(object):
    def __init__(self, target_size):
        self.target_size = target_size
    
    def __call__(self, img):
        if np.random.random() > 0.75:
            color_aug = transforms.ColorJitter(brightness=[1.5, 2.1], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=0.5)
            #random_factor = np.random.randint(0, 31) / 10.
            #sharpe_aug = transforms.RandomAdjustSharpness(random_factor, p=0.5)
            img = color_aug(img)
        
        if np.random.random() > 0.25:
            gau_blur = transforms.GaussianBlur(kernel_size=11, sigma=5)
            img = gau_blur(img)
        
        if np.random.random() > 0.75:
            p = np.random.choice([0, 1])
            transform_hflip = transforms.RandomHorizontalFlip(p)
            img = transform_hflip(img)
        
        
        # RandomResizedCrop
        if np.random.random() > 0.5:
            W,H = img.size
            transform_resizecrop = transforms.RandomResizedCrop(size=(H,W), scale=(0.3, 0.7),ratio=(0.8,1.2))
            img = transform_resizecrop(img)
        
        
        # RandomAffine
        if np.random.random() > 0.6:
            translate = (0.3,0.3)
            transform_translate = transforms.RandomAffine(degrees=0, translate=translate)
            img = transform_translate(img)
        

        # RandomPerspective
        if np.random.random() > 0.3:
            p = np.random.choice([0, 1]) 
            transform_pers = transforms.RandomPerspective(distortion_scale=0.25, p=p)
            img = transform_pers(img)
        
        # RandomRotation
        if np.random.random() > 0.3:
            p = np.random.uniform(-25,25,1)[0]
            transform_rotate = transforms.RandomRotation((p, p), expand=True)
            img = transform_rotate(img)

        
        # RandomVerticalFlip
        if np.random.random() > 0.5:
            p = np.random.choice([0, 1])
            transform_vflip = transforms.RandomVerticalFlip(p)
            img = transform_vflip(img)


        # resize
        transform_resize = transforms.Resize([self.target_size,self.target_size])
        img = transform_resize(img)     

        return img
