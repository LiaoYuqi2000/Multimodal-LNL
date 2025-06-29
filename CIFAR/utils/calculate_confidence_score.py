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
import sys
sys.path.append('../')
from model import *
from utils.get_config import load_config, get_args


from tqdm import tqdm
import _pickle as cPickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

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
    def __init__(self, dataset, root_dir, K):
        self.img_list = []
        self.label_list = []
        self.K = K
     
        if dataset=='cifar10': 
            for n in range(1,6):
                dpath = '%s/data_batch_%d'%(root_dir,n)
                data_dic = unpickle(dpath)
                self.img_list.append(data_dic['data'])
            self.img_list = np.concatenate(self.img_list)
        elif dataset=='cifar100':    
            train_dic = unpickle('%s/train'%root_dir)
            self.img_list = train_dic['data']
        self.img_list = self.img_list.reshape((50000, 3, 32, 32))
        self.img_list = self.img_list.transpose((0, 2, 3, 1))     
       

        # transform     
        target_size = 224
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        self.aug = Augment(target_size = target_size)
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),    
            ]) 

        print(len(self.img_list))

    def __getitem__(self, index):
        aug_img_list = []
        pil = Image.fromarray(self.img_list[index])
        
        for i in range(self.K):
            img = self.aug(pil)
            img = self.transform(img)   #[C, 224, 224]
            aug_img_list.append(img)
        aug_img_list = torch.stack(aug_img_list).squeeze()  #[K, C, 224, 224]
        return aug_img_list
    
    def __len__(self):
        return len(self.img_list)



# calculate confidence score
def calculate_confidence_score(model, dataloader):
    """
    Calculate confidence scores for images in the dataloader using the given model.

    Returns:
        list: [[pred11, pred12, ..., pred1k], [pred21, pred22, ..., pred2k], ..., [predN1, predN2, ..., predNk]] 
    """
    
    total_pred = []      # the prediction results for all samples
    label_list = []      # the labels for all samples
    img_path_list = []   # the file paths of all samples

    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for imgK in tqdm(dataloader):
            batch_pred = []    # the predictions of samples in this batch.
            # img_path_list.append(img_path)
            imgK = imgK.transpose(1, 0)     # [k, B, C, 224, 224]
            for k in range(imgK.size()[0]):
                image = imgK[k]   # [B, C, 244, 244]
                image = image.cuda()    # 【N, 3， 244, 244】
                outputs = model(image)                         #  [B, 50]
                pred = torch.max(outputs, axis = 1)[1]              #  tensor:[B]
                pred = torch.tensor([i.item() for i in pred])    # tensor: [B]
                batch_pred.append(pred) 
            # print(imgK.size()[0])
            batch_pred = torch.stack(batch_pred, dim=0).transpose(1,0)       # tensor：【B, K】
            # print(batch_pred.size())
            total_pred.append(batch_pred)
         

        total_pred = torch.cat(total_pred, dim = 0)  # tensor: [N, K]
        
    return total_pred.tolist()
        





if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args = get_args()
    config = load_config(args.config)
    
    K = config["stage2"]["K"]            # Number of augmentations per image.
    root_dir = config["data"]["data_path"]

    
    # Load model
    cp_classifier = config["classifier"]["classifier_save_path"]
    clip_args = config["clip"]
    model = Net(clip_args, cp_classifier)  
    model = model.cuda()
       

    # Load data
    dataset_name = config["data"]["dataset_name"]
    cifar_dataset = dataset(dataset_name, root_dir, K)
    data_loader = DataLoader(dataset = cifar_dataset, 
                            batch_size = 128,
                            shuffle = False,
                            num_workers = 2,
                            pin_memory = True)

    print("data num:", len(cifar_dataset))


    # Predictions on k augmentations
    total_pred = calculate_confidence_score(model, data_loader)


    # Save the result as a txt file
    save_file = config["stage2"]["consistency_score_save_path"]
    with open(save_file, "w") as f:
        json.dump(total_pred, f)
    print("The result is saved at", save_file)
