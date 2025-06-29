import torch
from PIL import Image
import open_clip
import os
import os.path as osp
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Imag
import shutil
from sklearn.mixture import GaussianMixture

import sys
sys.path.append('./')
from model import *
from utils.get_config import load_config, get_args

# img_dict (label: [img_path1. img_path2, ...])
def load_img(root_dir, img_file, num_classes):
    img_dict = {}
    for i in range(num_classes):
        img_dict[i] = []
    cnt = 0

    with open(img_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_path, label = line.strip().split(' ')
            # print(img_path)
            if int(label) < num_classes:
                img_dict[int(label)].append(osp.join(root_dir, img_path))
                cnt += 1
            elif int(label) > num_classes:
                break
    return img_dict


class dataset(Dataset):
    def __init__(self, img_list, transform):
        self.img_list = img_list
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.img_list)

# 加载label到 label_dict  (label:[text1, text2, text3,...])
def load_label(text_file, num_classes):
    label_dict = {}
    for i in range(num_classes):
        label_dict[i] = []

    with open(text_file, "r") as f:
        lines = f.readlines()
        for label, line in enumerate(lines):
            if label < num_classes:
                line = line.strip()[10:].split(",")
                for text in line:
                    label_dict[label].append(text)
            else:
                break
    return label_dict


# 计算样本与lable的相似性 sim_dict  (label：[sim1,sim2, sim3, ....])
def calculate_sim(model, img_dict, label_dict, num_classes):
    sim_dict = {} 
    for i in range(num_classes):
        sim_dict[i] = []
    cnt = 0
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for label in range(num_classes):
            # 定义dataloader
            web_dataset = dataset(img_dict[label], preprocess)
            web_loader = DataLoader(dataset = web_dataset, 
                                    batch_size = 256,
                                    shuffle = False,
                                    num_workers = 8,
                                    pin_memory = True)
            
            # 计算text_feature
            text = tokenizer(label_dict[label]).cuda()
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 提取图片特征，计算相似性
            for image in web_loader:
                image = image.cuda()
                image_features = model.encode_image(image)   # 【N, 3， 244, 244】
                image_features /= image_features.norm(dim=-1, keepdim=True)
                sim_list = torch.max((100 * image_features @ text_features.T), axis = 1)[0]   # 余弦相似性   
                sim_list = sim_list.cpu().numpy()   
                for sim in sim_list:
                    sim_dict[label].append(sim)
                cnt += 1
  
            print("已完成", label, ":", len(img_dict[label]))
    return sim_dict




if __name__ == '__main__':

    args = get_args()
    config = load_config(args.config)

    # Stage 1
    th_o = config["stage1"]["th_o"]
    save_path = config["stage1"]["ood_file_path"]


    # Data info  
    num_classes = config["data"]["num_classes"]
    train_data_path = config["data"]["train_data_path"]


    # Model
    model_type = config["clip"]["model_type"]
    model = ImageEncoder(model_type)
    model.cuda()
    _, _, preprocess = open_clip.create_model_and_transforms(model_type[0])
    tokenizer = open_clip.get_tokenizer(model_type[0])
    

    img_file = osp.join(root_dir, "info/train_filelist_google.txt")
    text_file = osp.join(root_dir, "info/synsets.txt")
    # text_file = osp.join(root_dir, "info/synsets_openai.txt")
    img_dict = load_img(root_dir, img_file, num_classes)                            # img_dict (label: [img_path1. img_path2, ...])
    label_dict = load_label(text_file, num_classes)                                 # label_dict  (label:[text1, text2, text3,...])
    sim_dict = calculate_sim(model, img_dict, label_dict, num_classes)   # 计算样本到text的相似性，并保存


    # GMM筛选ood
    cnt = 0    

    for label in range(num_classes):
        sim_list = sim_dict[label]
        img_list = img_dict[label]
        sim = torch.tensor(np.array(sim_list))
        sim = (sim - sim.min()) / (sim.max() - sim.min())
        input_sim = sim.reshape(-1,1)
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=1e-3)
        gmm.fit(input_sim)
        prob = gmm.predict_proba(input_sim)
        prob = prob[:,gmm.means_.argmax()] 
        for idx, prob0  in enumerate(prob):
            if prob0 > th:   
                cnt += 1
                with open(save_path, "a") as f:
                    f.write(img_list[idx][len(root_dir) + 1:] + " " + str(label) + "\n")

    print("ood number :", cnt)  




    


