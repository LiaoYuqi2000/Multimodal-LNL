# Learning from Noisy Data Using Pretrained Vision-Language Representations

## 1. Requirements
The code requires `python>=3.7` and the following packages.
```
torch==1.21.1
torchvision==0.12.0
numpy==1.23.1
tqdm==4.64.1
```
These packages can be installed directly by running the following command:
```
pip install -r requirements.txt
```
Note that all the experiments are conducted under two  <b>GeForce RTX3090 GPU</b>, so the results may be a little different with the original paper when you use a different gpu.



## 2. Data
This code includes seven datasets including: CIFAR-10, CIFAR-100, Tiny-ImageNet, Animals-10N, Food-101, Mini-WebVision (top-50 classes from WebVisionV1.0 (training set and validation set) and ILSVRC-2012 (only validation set)) and Clothing1M.

|Datasets|Download links|
| --------- | ---- |
|CIFAR-10|[link](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)|
|CIFAR-100|[link](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)|
|WebVision V1.0|[link](https://data.vision.ee.ethz.ch/cvl/webvision/download.html)|
|ILSVRC-2012|[link](https://image-net.org/challenges/LSVRC/2012/index.php)|
|Animal-10N|[link](https://forms.gle/8mbmbNgDFQ2rA1fLA)|
|Food-101|[link](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)|

## 3. Reproduce the results
### 3.1 Synthetic dataset (CIFAR-10, CIFAR-100)
For instance, for the synthetic label noise dataset CIFAR10, the example of script is shown as follows:
Update the configuration file `cifar10_config.yaml` with your desired settings.
```bash
# Before running the script, please update the configuration file `cifar10_config.yaml` as needed (e.g., dataset path, noise ratio, batch size).
cd synthetic dataset
# Generate synthetic noise
python utils/generate_noisy_data.py --config "./configs/cifar10_config.yaml"       
# Initialize classifier weights with label text embeddings     
python utils/get_zeroshot_classifier.py --config "./configs/cifar10_config.yaml"        
# Perform stage2
python utils/calculate_confidence_score.py  --config "./configs/cifar10_config.yaml"    
# Perform stage3
python train.py  --config "./configs/cifar10_config.yaml"   
# Perform model ensembling
python model_ensemble.py "./configs/cifar10_config.yaml"    
# Test the model
python test.py  --config "./configs/cifar10_config.yaml"    

```

### 3.2 Real-world noise dataset
For instance, for the real-world label noise dataset Food-101N, the example of script is shown as follows:
Update the configuration file `_config.yaml` with your desired settings.
```bash
# Before running the script, please update the configuration file `cifar10_config.yaml` as needed (e.g., dataset path, noise ratio, batch size).
      
# Initialize classifier weights with label text embeddings     
python utils/get_zeroshot_classifier.py --config "./configs/cifar10_config.yaml"        
# Perform stage2
python utils/calculate_confidence_score.py  --config "./configs/cifar10_config.yaml"    
# Perform stage3
python train.py  --config "./configs/cifar10_config.yaml"   
# Perform model ensembling
python model_ensemble.py "./configs/cifar10_config.yaml"    
# Test the model
python test.py  --config "./configs/cifar10_config.yaml"    
```

