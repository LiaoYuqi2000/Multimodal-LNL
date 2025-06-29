import torch
import open_clip
import shutil
import os
import os.path as osp
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from sklearn.mixture import GaussianMixture
from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm

import sys
sys.path.append('./')
from model import *
from utils.get_config import load_config, get_args




# Augment (target_size = 224)
class Augment(object):
    '''
    '''
    def __init__(self, target_size):
        self.target_size = target_size
    
    def __call__(self, img):
        
        # RandomHorizontalFlip
        if np.random.random() > 0.5:
            transform_hflip = transforms.RandomHorizontalFlip(1)
            img = transform_hflip(img)
        
        
        # RandomResizedCrop
        if np.random.random() > 0.5:
            W = self.target_size
            H = self.target_size
            transform_resizecrop = transforms.RandomResizedCrop(size=(H,W), scale=(0.7, 0.9),ratio=(0.8,1.2))
            img = transform_resizecrop(img)
        
        
        # RandomAffine
        if np.random.random() > 0.5:
            translate = (0.3,0.3)
            transform_translate = transforms.RandomAffine(degrees=0, translate=translate)
            img = transform_translate(img)
        
        
        # RandomRotation
        if np.random.random() > 0.5:
            p = np.random.uniform(-90,90,1)[0]
            transform_rotate = transforms.RandomRotation((p, p), expand=True)
            img = transform_rotate(img)


         # RandomVerticalFlip
        if np.random.random() > 0.5:
            transform_vflip = transforms.RandomVerticalFlip(1)
            img = transform_vflip(img)

        # resize
        transform_resize = transforms.Resize([self.target_size,self.target_size])
        img = transform_resize(img)     

        return img




#  dataloader
class dataset(Dataset):
    def __init__(self, root_dir, num_classes, K):
        self.img_list = []
        self.label_list = []
        self.K = K

        for img_name in os.listdir(osp.join(root_dir, "training")):
            img_path = osp.join("training", img_name)        
            self.img_list.append(img_path)
            self.labels_list.append(int(img_name.split('_')[0]))
            
        # transform     
        target_size = 224
        mean = (0.6959, 0.6537, 0.6371)
        std  = (0.3113, 0.3192, 0.3214)
        self.aug = Augment(target_size = target_size)
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),    
            ]) 

        print(len(self.img_list))

    def __getitem__(self, index):
        aug_img_list = []
        img_path = self.img_list[index]
        pil = Image.open(img_path).convert("RGB")
        for i in range(self.K):
            img = self.aug(pil)
            img = self.transform(img)   #[C, 224, 224]
            aug_img_list.append(img)
        label = self.label_list[index]
        aug_img_list = torch.stack(aug_img_list).squeeze()  #[K, C, 224, 224]
        parts = img_path.split(os.sep)
        img_path = os.path.join(parts[-2], parts[-1])  
        return aug_img_list, label, img_path
    
    def __len__(self):
        return len(self.img_list)



# calculate confidence score
def calculate_confidence_score(model, dataloader):

    total_pred = []      # the prediction results for all samples
    label_list = []      # the labels for all samples
    img_path_list = []   # the file paths of all samples

    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for imgK, labels, img_path in tqdm(dataloader):
            batch_pred = []    
            img_path_list.append(img_path)
            imgK = imgK.transpose(1, 0)     # [k, B, C, 224, 224]
            for k in range(imgK.size()[0]):
                image = imgK[k]   # [B, C, 244, 244]
                image, labels = image.cuda(), labels.cuda()    # 【N, 3， 244, 244】
                outputs = model(image)                         #  [B, 50]
                pred = torch.max(outputs, axis = 1)[1]              #  tensor:[B]
                pred = torch.tensor([i.item() for i in pred])    # tensor: [B]
                batch_pred.append(pred) 
            batch_pred = torch.stack(batch_pred, dim=0).transpose(1,0)       # tensor：【B, K】
            total_pred.append(batch_pred)
            label_list.append(labels)

        total_pred = torch.cat(total_pred, dim = 0)  # tensor: [N, K]
        label_list = torch.cat(label_list, dim = 0).view(-1, 1).expand_as(total_pred).cpu()     # tensor:[N, K]
        score = torch.sum((total_pred.eq(label_list)), axis = 1) / total_pred.size(1)           # tensor:[N]
        img_path_list = [img_path for batch_path in img_path_list  for img_path in batch_path]
        

    result = {}
    for idx, s in enumerate(score):
        result[img_path_list[idx]] = s.item()
    
    return result
        


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args = get_args()
    config = load_config(args.config)
    
     
    # Stage2 info
    K = config["stage2"]["K"]            # Number of augmentations per image.
    th = config["stage2"]["th"]
    save_file = config["stage2"]["consistency_score_save_path"]
    finetuned_model_path = config["stage2"]["finetuned_model_path"]


    # Load model
    cp_classifier = config["classifier"]["classifier_save_path"]
    clip_args = config["clip"]
    model = Net(clip_args, cp_classifier)  
    model = model.cuda()
    if finetuned_model_path is not None:
        checkpoint = torch.load(finetuned_model_path)
        model.load_state_dict(checkpoint['model'])
    
    

    # Load data
    root_dir = config["data"]["data_path"]
    num_classes = config["data"]["num_classes"]
    animal_dataset = dataset(root_dir, num_classes, K)
    animal_loader = DataLoader(dataset = animal_dataset, 
                            batch_size = 256,
                            shuffle = False,
                            num_workers = 2,
                            pin_memory = True)
    print("data num:", len(animal_dataset))


    # Calculate score
    img_score_result = calculate_confidence_score(model, animal_loader)   # {img_path: score}

    # Save result
    with open(save_file, "w") as f:
        json.dump(img_score_result, f)
    print("The result is saved at ", save_file)
