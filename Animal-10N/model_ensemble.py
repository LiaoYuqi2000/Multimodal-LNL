# 混合zero-shot model 和 finetune model，并测试

import os
import os.path as osp
import torch
from model import *
from dataloader_animal import animal_dataloader as dataloader
import sys
sys.path.append("./")
from utils.get_config import get_args, load_config, save_config



def accu(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
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
            # result[]
            _, pred = torch.max(outputs, dim=1)

    acc = 100. * correct_nums/ total


    return acc



def save_model(alpha, model, save_path):
    checkpoint = {
        "alpha": alpha,
        "model": model.module.state_dict(),
    }
    torch.save(checkpoint, save_path)



if __name__ == "__main__":

    args = get_args()
    config = load_config(args.config)

    # Load data
    print("loading  data")
    data_path = config["data"]["data_path"]
    num_classes = config["data"]["num_classes"]
    dataset_name = config["data"]["dataset_name"]
    loader = dataloader(data_path, num_workers=4)
    test_loader = loader.run(mode = "test", batch_size = 256)

    # GPU
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    gpu_ids = [0, 1]


   # Load models
    print("loading model")
    checkpoint_finetuned = config["model_ensemble"]["trained_model_path"]
    cp_classifier = config["classifier"]["classifier_save_path"]
    clip_args = config["clip"]
    zeroshot = Net(clip_args, cp_classifier)  
    theta_0 = {"module." + k: v.clone() for k, v in zeroshot.state_dict().items()}
    finetuned = Net(clip_args, cp_classifier)
    finetuned.load_state_dict(torch.load(osp.join(checkpoint_finetuned, "best_model.pth"), map_location='cpu')["model"])
    theta_1 = {"module." + k: v.clone() for k, v in finetuned.state_dict().items()}
    del zeroshot
    finetuned = finetuned.cuda()
    if len(gpu_ids) > 1:
        finetuned = nn.DataParallel(finetuned, device_ids = gpu_ids)

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())    


    # Log
    log = open(osp.join(checkpoint_finetuned, "model_ensemble.txt"), 'w')
    log.write("(1 - alpha) * zero_shot + alpha * finetuned \n")
    log.flush()      
    
    best_acc = 0.0
    alpha_list = config["model_ensemble"]["alpha_list"]      # (1 - alpha) * zero_shot + alpha * finetuned
    for alpha in alpha_list:
        # merge two model
        theta = {key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
                 for key in theta_0.keys()}

        # update the model (in-place) acccording to the new weights
        finetuned.load_state_dict(theta)


        # evaluate
        test_acc = test(finetuned, test_loader)
        print("alpha: %.1f, test_acc:%.2f%%" %(alpha, test_acc))
        log.write("alpha: %.1f, test_acc:%.2f%%\n" %(alpha, test_acc))
        log.flush()
      

        # save model
        save_path = osp.join(checkpoint_finetuned, "model_ensemble_"  +  str(alpha) + ".pth")
        save_model(alpha, finetuned, save_path)



