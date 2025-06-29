# Before running the script, please update the configuration file `cifar10_config.yaml` as needed (e.g., dataset path, noise ratio, batch size).

python utils/get_zeroshot_classifier.py --config "./config_animal-10n.yaml"   # Initialize classifiers with text features

python finetune.py --config "./config_animal-10n.yaml"       # Used when zero-shot performance is poor. 

python utils/calculate_confidence_score.py --config "./config_animal-10n.yaml"       # Stage2

python train.py --config "./config_animal-10n.yaml"    # Stage3

python model_ensemble.py  --config "./config_animal-10n.yaml"     # Model ensemble

python test.py --config "./config_animal-10n.yaml"     # Test