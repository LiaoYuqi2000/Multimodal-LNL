import os.path as osp
import torch
import open_clip
import templates
from tqdm import tqdm
import sys
sys.path.append('../')
from model import ClassificationHead
from utils.get_config import load_config, get_args



def get_classnames(file, num_classes):
    label_classnames_dict = {}
    for i in range(num_classes):
        label_classnames_dict[i] = []

    with open(file, 'r') as f:
        lines = f.readlines()
        for label, line in enumerate(lines):
            label_classnames_dict[label].append(line.strip())
    print(label_classnames_dict)
    return label_classnames_dict


def get_zeroshot_classifier(model_type, pretrained_clip_path, synsets_path, template_name, num_classes):
    # load model
    clip, _, _ = open_clip.create_model_and_transforms(model_type[0])
    clip.load_state_dict(torch.load(pretrained_clip_path))
    tokenizer = open_clip.get_tokenizer(model_type[0])
    clip = clip.cuda()


    # Load the template and class names.
    template = getattr(templates, template_name)
    label_classnames_dict = get_classnames(synsets_path, num_classes)

    # calculate text embeddings
    clip.eval()
    with torch.no_grad():
        zeroshot_weights = []
        for label in range(num_classes):
            classnames = label_classnames_dict[label]
            texts = []
            for classname in classnames:
                for t in template:
                    texts.append(t(classname))
            print(texts)
            texts = tokenizer(texts).cuda() # tokenize
            embeddings = clip.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)    # [80, d]

            embeddings = embeddings.mean(dim=0, keepdim=True)      # [1, d]，将不同的template平均
            embeddings /= embeddings.norm()                        # [1, d]

            zeroshot_weights.append(embeddings)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=0)          # [num_classes, 1, d]
    zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)       # [d, 1, num_classes]
    
    zeroshot_weights *= clip.logit_scale.exp()                       # *100
    
    zeroshot_weights = zeroshot_weights.squeeze().float()            # [d, num_classes]
    zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)       # [num_classes, d]


    classification_head = ClassificationHead(weights=zeroshot_weights)

    return classification_head




def save(model, model_type, synset_name, template_name, save_path):
    checkpoint = {
        "model" : model.state_dict(),
        'model_type' : model_type,
        "synset_name" : synset_name,
        "template_name" : template_name,
    }
    torch.save(checkpoint, save_path)



if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config)
    num_classes = config["data"]["num_classes"]
    model_type = config["clip"]["model_type"]
    synsets_path = config["classifier"]["synsets_path"]
    save_path = config["classifier"]["classifier_save_path"]
    template_name = config["classifier"]["template_name"]      
    pretrained_clip_path = config["clip"]["pretrained_model_path"]
    synsets_name = osp.basename(synsets_path)
    zero_shot_classifier = get_zeroshot_classifier(model_type, pretrained_clip_path, synsets_path, template_name, num_classes)
    save(zero_shot_classifier, model_type, synsets_name, template_name, save_path)

