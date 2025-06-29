import argparse

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
from dataloader_cifar import cifar_dataloader as dataloader
import sys
sys.path.append("./")
from utils.get_config import get_args, load_config, save_config





def train (epoch, model, train_loader, optimizer, scheduler):
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


if __name__ == "__main__":
 
    args = get_args()
    config = load_config(args.config)

    # Train info
    batch_size = config["finetune"]["batch_size"]    
    num_epochs = config["finetune"]["num_epochs"]
    num_workers = config["finetune"]["num_workers"]
    base_lr = config["finetune"]["base_lr"]


    # Save model path
    checkpoint_dir = config["finetune"]["checkpoint_dir"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = osp.join(checkpoint_dir, "configs.yaml")
    save_config(args.config, save_path)
    print("Saving to ", checkpoint_dir, "......")

    # Data info
    root_dir = config["data"]["data_path"]
    dataset_name = config["data"]["dataset_name"]
    noise_type = config["finetune"]["noise_type"]   # sym, asym
    noise_ratio = config["finetune"]["noise_ratio"]
    

    # Stage 2 info
    th = config["stage2"]["th"]
    consistency_score_file_path = config["stage2"]["consistency_score_save_path"]

    # GPU
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
    gpu_ids = [0, 1]


    # Load model
    cp_classifier = config["classifier"]["classifier_save_path"]
    clip_args = config["clip"]
    model = Net(clip_args, cp_classifier)  
    model = model.cuda()
 

    # Load data
    num_workers = config["finetune"]["num_workers"]
    loader = dataloader(root_dir, dataset_name, num_workers, noise_type, noise_ratio, th, consistency_score_file_path)
    train_loader = loader.run(config["finetune"]["train_mode"], batch_size)
    test_loader = loader.run("test", int(batch_size * 4))


    # Load optimizer
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay = 5e-4)  
    warm_up_epochs = 500
    total_batches = len(train_loader) * num_epochs
    warm_up_with_cosine_lr = lambda k: (k+1) / warm_up_epochs if k < warm_up_epochs else 0.5 * (math.cos((k - warm_up_epochs) /(total_batches - warm_up_epochs) * math.pi) + 1)
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
        best_acc = torch.load(osp.join(checkpoint_dir, "best_model.pth"))["acc"]
        print("Resuming the train process at %d epochs..."%(start_epoch))
    
    else:
        log = open(osp.join(checkpoint_dir, "acc.txt"), 'w')
        start_epoch = 0
        best_acc = 0


    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids = gpu_ids)

    
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        loss = train(epoch, model, train_loader, optimizer, scheduler)
        test_acc = test(model, test_loader)
        scheduler.step() 

        save_model(epoch, test_acc, model, optimizer, scheduler, osp.join(checkpoint_dir,"lastest.pth"))
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(epoch, best_acc, model, optimizer, scheduler, osp.join(checkpoint_dir,"best_model.pth"))
        end_time = time.time()

        print()
        print("Epoch: %d, lr:%.6f, loss:%.3f, test_acc:%.2f%%" %(epoch, scheduler.get_last_lr()[0], loss, test_acc))
        print('Single epoch cost time:%2.f min'%((end_time - start_time)/60))
        log.write("Epoch: %d, lr:%.6f, loss:%.3f, test_acc:%.2f%%\n" %(epoch, scheduler.get_last_lr()[0], loss, test_acc))
        log.flush()
        
    print("best test_acc:%.2f%%\n" %(best_acc))
    log.write("best_test_acc:%.2f%%\n" %(best_acc))
    log.flush()
        
