#coding:utf8

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
import time
import sys
import math
import os.path as osp
import numpy as np
import torch.nn.functional as F
import open_clip
from model import *
from dataloader_food101n import food101n_dataloader as dataloader
import sys
sys.path.append("./")
from utils.get_config import get_args, load_config, save_config




            


def warm_up (epoch, model, train_loader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    loss_recorder = []
    num_iterations = len(train_loader)
    
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_recorder.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        sys.stdout.write('\n')
        sys.stdout.write('Epoch%d\t Iter[%d|%d]\t Loss:%.3f'%(epoch, batch_idx, num_iterations, loss) )
        sys.stdout.flush()

    loss = np.mean(loss_recorder)

    return loss



def train (epoch, model, labeled_loader, unlabeled_loader, threshold, T, lambda_u, optimizer, scheduler):
    loss_recorder = []
    num_iterations = len(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader) 
    num_unlabeled = 0

    model.train()
    for batch_idx, (inputs_x, labels_x) in enumerate(labeled_loader):
        try:
            inputs_u_w, inputs_u_s = unlabeled_iter.next()    
        except:
            unlabeled_iter = iter(unlabeled_loader)           
            inputs_u_w, inputs_u_s = unlabeled_iter.next() 
        
       
        inputs = torch.cat([inputs_x, inputs_u_s, inputs_u_w], dim = 0)
        outputs = model(inputs.cuda())
        outputs_x = outputs[:inputs_x.size()[0]]
        outputs_u_s = outputs[inputs_x.size()[0] : inputs_x.size()[0] + inputs_u_s.size()[0]]
        outputs_u_w = outputs[inputs_x.size()[0] + inputs_u_s.size()[0]:]

        labels_x = labels_x.cuda()
        
        Lx = F.cross_entropy(outputs_x, labels_x, reduction='mean')
        pseudo_label = torch.softmax(outputs_u_w.detach()/T, dim=-1)   # [B, C]
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)         # [B]
        mask = max_probs.ge(threshold).float()   # ≥
        num_unlabeled += torch.sum(mask)

        Lu = (F.cross_entropy(outputs_u_s, targets_u, reduction='none') * mask).mean()
        num_unlabeled += torch.sum(mask)

        loss = Lx + lambda_u * Lu

        loss_recorder.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        sys.stdout.write('\n')
        sys.stdout.write('Epoch%d\t Iter[%d|%d]\t Lx:%.3f\t Lu:%.3f\t loss:%.3f\t num_unlabeled:%d'%(epoch, batch_idx, num_iterations, Lx, Lu, loss, num_unlabeled) )
        sys.stdout.flush()
    
    loss = np.mean(loss_recorder)

    return loss, num_unlabeled.item()



def accu(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, dim = 1, largest = True, sorted=True)  #largest为True从大到小排序；sorted为True，
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_num = torch.sum(correct[:k].sum(0) > 0)
            res.append(correct_num)
        return res
    

def test(model, test_loader):
  
    correct_nums = 0
    total = 0
    model.eval() 
    with torch.no_grad():  
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)  
            correct_num = accu(outputs, targets, topk=(1,))[0]
            correct_nums += correct_num.item()
            total += targets.size(0)

    acc = 100. * correct_nums/ total

    return acc


def save_model(epoch, acc, model, optimizer, scheduler, save_path):
    checkpoint = {
        "epoch": epoch,
        "acc": acc,
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, save_path)



if __name__ == '__main__':
    args = get_args()
    config = load_config(args.config)

    # Stage 1 info
    ood_file_path = config["stage1"]["ood_file_path"]

    # Stage 2 info
    th = config["stage2"]["th"]                     # Threshold for selecting clean labels
    consistency_score_file_path = config["stage2"]["consistency_score_save_path"]
 
    # Train info (stage 3)
    batch_size = config["stage3"]["batch_size"]    
    num_epochs = config["stage3"]["num_epochs"]
    num_workers = config["stage3"]["num_workers"]
    base_lr = config["stage3"]["base_lr"]
    T = config["stage3"]["T"]                           # Temperature coefficient
    threshold = config["stage3"]["threshold"]           # Threshold for selecting high-confidence pseudo labels
    lambda_u = config["stage3"]["lambda_u"]             # Loss weight for noisy samples   

    # Data info
    root_train_dir = config["data"]["train_data_path"]
    root_test_dir = config["data"]["test_data_path"]
    

    # GPU
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
    gpu_ids = [0, 1]
    
    # Load model
    cp_classifier = config["classifier"]["classifier_save_path"]
    clip_args = config["clip"]
    model = Net(clip_args, cp_classifier)  
    model = model.cuda()

    # Save model path
    checkpoint_dir = config["finetune"]["checkpoint_dir"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = osp.join(checkpoint_dir, "configs.yaml")
    save_config(args.config, save_path)
    print("Saving to ", checkpoint_dir, "......")


    # Load data
    num_workers = config["finetune"]["num_workers"]
    loader = dataloader(root_train_dir, root_test_dir, num_workers, th, consistency_score_file_path, ood_file_path)
    finetune_loader = loader.run("finetune_clean", batch_size * 2)
    labeled_loader = loader.run("labeled", batch_size)
    unlabeled_loader = loader.run("unlabeled",  int(batch_size/2))
    test_loader = loader.run("test", int(batch_size * 4))


    # Load optimizer
    warm_up_epochs = config["stage3"]["warm_up_epochs"]
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay = 5e-4)  
    warm_up_iterations = 500
    total_batches = len(labeled_loader) * (num_epochs - warm_up_epochs) + len(finetune_loader) * warm_up_epochs
    warm_up_with_cosine_lr = lambda k: (k+1) / warm_up_iterations if k < warm_up_iterations else 0.5 * (math.cos((k - warm_up_iterations) /(total_batches - warm_up_iterations) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)


    # Resume
    resume = config["finetune"]["resume"]
    if resume:
        checkpoint = torch.load(osp.join(checkpoint_dir, "lastest.pth"), map_location='cpu')
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        log = open(osp.join(checkpoint_dir, "acc.txt"), 'a')
        best_acc = checkpoint["acc"]
        print("Resuming the train process at %d epochs..."%(start_epoch))
    
    else:
        log = open(osp.join(checkpoint_dir, "acc.txt"), 'w')
        start_epoch = 0
        best_acc = 0
        
    
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids = gpu_ids)


    
    # train
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        if epoch < warm_up_epochs:
            loss = warm_up(epoch, model, finetune_loader, optimizer, scheduler)
            num_unlabeled = 0
        else:
            loss, num_unlabeled = train(epoch, model, labeled_loader, unlabeled_loader, threshold, T, lambda_u, optimizer, scheduler)
        test_acc = test(model, test_loader)
        save_model(epoch, test_acc, model, optimizer, scheduler, osp.join(checkpoint_dir,"lastest.pth"))
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(epoch, best_acc, model, optimizer, scheduler, osp.join(checkpoint_dir,"best_model.pth"))
        end_time = time.time()


        print()
        print("Epoch: %d, lr:%.6f, loss:%.3f, num_unlabeled:%.1f, test_acc:%.2f%%" %(epoch, scheduler.get_last_lr()[0], loss, num_unlabeled, test_acc))
        print('Single epoch cost time:%2.f min'%((end_time - start_time)/60))
        log.write("Epoch: %d, lr:%.6f, loss:%.3f, num_unlabeled:%.1f, test_acc:%.2f%%\n" %(epoch, scheduler.get_last_lr()[0], loss, num_unlabeled, test_acc))
        log.flush()


    print("best test_acc:%.2f%%\n" %(best_acc))
    log.write("best_test_acc:%.2f%%\n" %(best_acc))
    log.flush()



    

