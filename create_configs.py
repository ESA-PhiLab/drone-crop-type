from copy import deepcopy
from glob import glob
import sys
import os
from os.path import abspath, basename, join

import yaml


folds_dir = sys.argv[1]
results_dir = sys.argv[2]
prefix = "" if len(sys.argv) < 4 else sys.argv[3]
selection = None if len(sys.argv) < 5 else sys.argv[4]

config_dir = "configs"
if prefix:
    config_dir = join(config_dir, prefix)

class_mapping = False

# Malawi Summer classes:
# classes = ['cassava', 'groundnut', 'maize', 'tobacco']
# classes = ['cassava', 'groundnut', 'maize', 'sweet_potatoes', 'tobacco']
# classes = ['cassava', 'groundnut', 'maize', 'other', 'sweet_potatoes', 'tobacco']
# classes = ['crop', 'non-crop']
# class_mapping = {
#         'cassava': 'crop', 
#         'groundnut': 'crop', 
#         'maize': 'crop', 
#         'other': 'non-crop', 
#         'sweet_potatoes': 'crop', 
#         'tobacco': 'crop'
#     }

# Mozambique classes:
classes = ['cassava', 'maize', 'other', 'rice']

base = {
    'optimizer': 'SGD',
    'optimizer_options': {
        'lr': 0.0003
    },
    'model': None,
    'model_options': {
        'frozen_layers': 0,
        'num_outputs': len(classes),
        'input_shape': (224, 224, 3),
        'layer_sizes': [
            [0.3, 60],
            [0.3, 30],
        ],
        'weights': None,
    },
    'training_folds': None,
    'validation_folds': None,
    'results_dir': results_dir,
    'balance_training_data': False,
    'augmentation': True,
    'classes': classes,
    'class_mapping': class_mapping,
    'batch_size': 32,
    'training_epochs': 50,
    'training_steps': 350,
    'validation_steps': None,
    'verbose': 2,
}

experiments = dict()
experiments['xception'] = {
    **deepcopy(base),
    'model': 'xception',
}

experiments['xception-f95'] = {
    **deepcopy(base),
    'model': 'xception',
    'model_options': {
        **base['model_options'],
        'frozen_layers': 95,
        'weights': 'imagenet',
    },
    'optimizer_options': {
        **base['optimizer_options'],
        'lr': 0.001
    },
}

experiments['vgg16-vanilla'] = {
    **deepcopy(base),
    'model': 'vgg16',
}

# VGG-16 layer names
# 0 input_1
# 1 block1_conv1
# 2 block1_conv2
# 3 block1_pool
# 4 block2_conv1
# 5 block2_conv2
# 6 block2_pool
# 7 block3_conv1
# 8 block3_conv2
# 9 block3_conv3
# 10 block3_pool
# 11 block4_conv1
# 12 block4_conv2
# 13 block4_conv3
# 14 block4_pool
# 15 block5_conv1
# 16 block5_conv2
# 17 block5_conv3
# 18 block5_pool
# 19 flatten
# 20 fc1
# 21 fc2
# 22 predictions

for fl in [0, 2, 4, 7, 11, 15, 18]:
    experiments[f'vgg16-f{fl}'] = {
        **deepcopy(base),
        'model': 'vgg16',
        'model_options': {
            **base['model_options'],
            'frozen_layers': fl,
            'weights': 'imagenet',
        },
    }

os.makedirs(config_dir, exist_ok=True)

fold_files = glob(join(abspath(folds_dir), "*"))
for validation_fold in fold_files:
    training_folds = fold_files.copy()
    training_folds.remove(validation_fold)
    validation_fold_id = basename(validation_fold).replace('.yaml', '')
    
    for name, experiment in experiments.items():
        _name = experiment['model']
        if experiment['model_options']['weights'] is None:
            _name += "_vanilla" 
        name = "_".join([
            _name,
            f"l{len(experiment['model_options']['layer_sizes'])}",
            f"ls{experiment['model_options']['layer_sizes'][0][1]}",
            f"lr{experiment['optimizer_options']['lr']}",
            f"d{experiment['model_options']['layer_sizes'][0][0]}",
            f"fl{experiment['model_options']['frozen_layers']}",
            validation_fold_id
        ])
        if prefix:
            experiment['name'] = join(prefix, name)
        
        experiment['training_folds'] = training_folds
        experiment['validation_folds'] = [validation_fold]
        
        print('Create', join(config_dir, name)+'.yml')
        with open(join(config_dir, name)+'.yml', 'w') as outfile:
            yaml.dump(experiment, outfile, default_flow_style=False)