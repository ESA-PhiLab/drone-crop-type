from copy import deepcopy
from glob import glob
import sys
import os
from os.path import abspath, basename, join

import yaml


folds_dir = sys.argv[1]
results_dir = sys.argv[2]
prefix = None if len(sys.argv) < 4 else sys.argv[3]
selection = None if len(sys.argv) < 5 else sys.argv[4]

config_dir = "configs"
if prefix is not None:
    config_dir = join(config_dir, prefix)

classes = ['cassava', 'groundnut', 'maize', 'sweet_potatoes', 'tobacco']
#classes = ['cassava', 'groundnut', 'maize', 'other', 'sweet_potatoes', 'tobacco']

base = {
    'optimizer': 'SGD',
    'optimizer_options': {},
    'model': None,
    'model_options': {
        'frozen_layers': 0,
        'num_outputs': len(classes),
        'input_shape': (299, 299, 3),
        'layer_sizes': [
            [0.5, 32],
            [0.3, 16],
        ],
        'weights': None,
    },
    'training_folds': None,
    'validation_folds': None,
    'results_dir': results_dir,
    'balance_training_data': False,
    'augmentation': False,
    'classes': classes,
    'batch_size': 32,
    'training_epochs': 40,
    'training_steps': 300,
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

experiments['vgg16'] = {
    **deepcopy(base),
    'model': 'vgg16',
    'optimizer_options': {
        **base['optimizer_options'],
        'lr': 0.015
    },
}

experiments['vgg16-f15'] = {
    **deepcopy(base),
    'model': 'vgg16',
    'model_options': {
        **base['model_options'],
        'frozen_layers': 15,
        'weights': 'imagenet',
    },
    'optimizer_options': {
        **base['optimizer_options'],
        'lr': 0.001
    },
}

os.makedirs(config_dir, exist_ok=True)

fold_files = glob(join(abspath(folds_dir), "*"))
for validation_fold in fold_files:
    training_folds = fold_files.copy()
    training_folds.remove(validation_fold)
    validation_fold_id = basename(validation_fold).replace('.yaml', '')
    
    for name, experiment in experiments.items():
        full_name = "-".join([name, validation_fold_id])
        if prefix is None:
            experiment['name'] = full_name
        else:
            experiment['name'] = prefix + "-" + full_name
        experiment['training_folds'] = training_folds
        experiment['validation_folds'] = [validation_fold]
        with open(join(config_dir, full_name)+'.yml', 'w') as outfile:
            yaml.dump(experiment, outfile, default_flow_style=False)