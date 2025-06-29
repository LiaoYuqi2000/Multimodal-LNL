import os
import sys
import time
import numpy as np
import os.path as osp
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import open_clip
from model import *
from dataloader_food101n import food101n_dataloader as dataloader

import sys
sys.path.append('./')
from utils.get_config import load_config, get_args

def accu(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim = 1, largest = True, sorted=True) 
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_num = torch.sum(correct[:k].sum(0) > 0)
            res.append(correct_num)
        return res
    



def test(model, test_loader, num_classes):

    correct_nums = 0
    total = 0
    result = np.zeros([num_classes, num_classes])
    model.eval() 
    with torch.no_grad():  
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)  
            correct_num = accu(outputs, targets, topk=(1,))[0]
            correct_nums += correct_num.item()
            total += targets.size(0)
            # result[]
            _, pred = torch.max(outputs, dim=1)
            result[targets.item()][pred.item()] += 1

    acc = 100. * correct_nums/ total
    print(result)

    return acc




if __name__ == "__main__":
   
    args = get_args()
    config = load_config(args.config)
    

    # Load model
    cp_classifier = config["classifier"]["classifier_save_path"]
    clip_args = config["clip"]
    model = Net(clip_args, cp_classifier)  
    model = model.cuda()
    checkpoint_dir = config["test"]["test_model_path"]
    if checkpoint_dir:
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint['model'])
        print(checkpoint_dir)
    else:
        print("zero-shot result")


    # Load data
    train_data_path = config["data"]["train_data_path"]
    test_data_path = config["data"]["test_data_path"]
    num_classes = config["data"]["num_classes"]
    loader = dataloader(train_data_path, test_data_path, num_workers=4)
    test_loader = loader.run(mode = "test", batch_size = 1)

    acc = test(model, test_loader, num_classes)

    print("acc:%.2f%%" %(acc))