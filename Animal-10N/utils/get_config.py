import yaml
import argparse
import shutil

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/cifar10_config.yaml', help='YAML配置文件路径')
    args, _ = parser.parse_known_args()
    return args




def save_config(config_file_path, save_path):
    shutil.copyfile(config_file_path, save_path)
    