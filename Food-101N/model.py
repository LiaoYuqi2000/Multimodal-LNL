import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math
import open_clip




class ImageEncoder(torch.nn.Module):
    def __init__(self, clip_args):
        super().__init__()
        model_type = clip_args["model_type"]
        self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_type[0])
        self.model.load_state_dict(torch.load(clip_args["pretrained_model_path"]))

    def forward(self, images):
        return self.model.encode_image(images)



class ClassificationHead(torch.nn.Linear):
    def __init__(self, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        inputs = inputs / inputs.norm(dim=-1, keepdim=True)   # 特征归一化
        return super().forward(inputs)



class Net(torch.nn.Module):
    def __init__(self, clip_args, cp_classifier):
        super().__init__()
        self.image_encoder = ImageEncoder(clip_args)

        checkpoint = torch.load(cp_classifier)
        weights = checkpoint["model"]['weight']
        biases = checkpoint["model"]['bias']
        self.classification_head = ClassificationHead(weights, biases)


    def forward(self, x):
        features = self.image_encoder(x)
        outputs = self.classification_head(features)
        return outputs

