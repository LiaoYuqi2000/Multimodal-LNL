import random
import numpy as np
from PIL import Image
import os
import os.path as osp
import json
from tqdm import tqdm
import _pickle as cPickle
import sys
sys.path.append('./')
from get_config import get_args, load_config




# CIFAR10/CIFAR100
# Generate symmetric and asymmetric noise


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def generate_noisy_data(root_dir, dataset, noise_ratio, noise_mode, noise_file):
    
    transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise

    train_label=[]
    if dataset=='cifar10': 
        for n in range(1,6):
            dpath = '%s/data_batch_%d'%(root_dir,n)
            data_dic = unpickle(dpath)
            train_label = train_label+data_dic['labels']
    elif dataset=='cifar100':    
        train_dic = unpickle('%s/train'%root_dir)
        train_label = train_dic['fine_labels']
    
    # Synthetic noise
    noise_label = []
    idx = list(range(50000))
    random.shuffle(idx)
    num_noise = int(noise_ratio*50000)            
    noise_idx = idx[:num_noise]
    for i in tqdm(range(50000)):
        if i in noise_idx:
            if noise_mode=='sym':
                if dataset=='cifar10': 
                    noiselabel = random.randint(0,9)
                elif dataset=='cifar100':    
                    noiselabel = random.randint(0,99)
                noise_label.append(noiselabel)
            elif noise_mode=='asym':   
                noiselabel = transition[train_label[i]]
                noise_label.append(noiselabel)                    
        else:    
            noise_label.append(train_label[i])   
    print(num_noise)
    print("save noisy labels to %s ..."%noise_file)        
    json.dump(noise_label,open(noise_file,"w"))     





if __name__ == "__main__":

    args = get_args()
    config = load_config(args.config)
    
    root_dir = config["data"]["data_path"]
    dataset_name = config["data"]["dataset_name"]
    noise_ratio = config["generate_noisy_data"]["noise_ratio"]
    noise_mode = config["generate_noisy_data"]["noise_mode"]    # "sym", "asym"
    noise_file = "a.json"
    # noise_file = osp.join(root_dir, str(noise_ratio) + '_' + noise_mode + ".json")
    generate_noisy_data(root_dir, dataset_name, noise_ratio, noise_mode, noise_file)


