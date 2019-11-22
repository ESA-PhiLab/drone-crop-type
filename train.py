"""Train a CNN on the patches extracted from the drone mosaics

To set the paths to the patches, please create your own configurations file

Usage:
    > python train.py CONFIG_FILE
"""

import json
import os
from os.path import join
import sys
from time import time
import yaml

# Load the config file:
if len(sys.argv) < 2:
    print('You have to provide a path to the configurations file!')
    exit()

try:
    with open(sys.argv[1]) as config_file:
        config = yaml.safe_load(config_file)
except Exception as e:
    print('Could not load configurations file')
    raise e

# Set the GPU device:
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_device'])


from tensorflow.keras import optimizers, backend

from dataset import Datasets
from models import PretrainedModel
from utils import get_experiment_name

# These classes are going to be used:
classes = sorted(
    ['cassava', 'groundnut', 'maize', 'other', 'tobacco']
)

# Load the datasets for training, validation and testing
data = Datasets(
    config['patches_path'], classes=classes, 
    validation_split=config['validation_split'],
    testing_split=config['testing_split']
)
print(data.summary())

# Let's set our own class weights:
# data.class_weights = {
#     0: 8., # cassava
#     1: 5, # groundnut
#     2: 1.2, # maize
#     3: 1.0, # other
#     4: 15, # sweetpotatoes
#     5: 7.0 # tobacco
# }

base_name = 'balanced'
experiments = [
    {'model': 'xception', 'frozen_layers': 25, 'optimizer': optimizers.RMSprop(lr=0.001)},
    {'model': 'xception', 'frozen_layers': 50, 'optimizer': optimizers.RMSprop(lr=0.001)},
    {'model': 'xception', 'frozen_layers': 50, 'optimizer': optimizers.RMSprop(lr=0.0005)},
    {'model': 'xception', 'frozen_layers': 50, 'optimizer': optimizers.RMSprop(lr=0.0008)},
    {'model': 'xception', 'frozen_layers': 25, 'optimizer': optimizers.RMSprop(lr=0.0005)},
    {'model': 'xception', 'frozen_layers': 25, 'optimizer': optimizers.RMSprop(lr=0.0008)},
#     {'model': 'xception', 'frozen_layers': 80, 'optimizer': optimizers.RMSprop(lr=0.001)},
#     {'model': 'xception', 'frozen_layers': 80, 'optimizer': optimizers.RMSprop(lr=0.0005)},
#     {'model': 'xception', 'frozen_layers': 80, 'optimizer': optimizers.RMSprop(lr=0.0008)},
#     {'model': 'xception', 'frozen_layers': 70, 'optimizer': optimizers.RMSprop(lr=0.001)},
#     {'model': 'xception', 'frozen_layers': 70, 'optimizer': optimizers.RMSprop(lr=0.0005)},
#     {'model': 'xception', 'frozen_layers': 70, 'optimizer': optimizers.RMSprop(lr=0.0008)},
#     {'model': 'baseline', 'optimizer': optimizers.RMSprop(lr=0.003)},
#     {'model': 'baseline', 'optimizer': optimizers.RMSprop(lr=0.001)},
#     {'model': 'baseline', 'optimizer': optimizers.RMSprop(lr=0.0005)},
#     {'model': 'effnet', 'optimizer': optimizers.RMSprop(lr=0.002)},
#     {'model': 'effnet', 'optimizer': optimizers.RMSprop(lr=0.0005)},
#     {'model': 'effnet', 'optimizer': optimizers.RMSprop(lr=0.001)},
]

backend.clear_session()

for experiment in experiments:
    hyperparameters = experiment.copy()
    hyperparameters['classes'] = classes
    hyperparameters['class_weights'] = data.class_weights

    experiment_name = \
        experiment['model'] + '-' + base_name + '-' + get_experiment_name(config['models_path'])
    experiment_dir = join(config['models_path'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Load the model architecture:
    model = PretrainedModel(
        num_outputs=len(classes), output_dir=experiment_dir, 
        input_shape=(299, 299, 3), **experiment
    )

    hyperparameters['optimizer'] = type(model.optimizer).__name__
    hyperparameters['optimizer_config'] = model.optimizer.get_config()

    with open(join(experiment_dir, 'hyperparameters.json'), 'w') as file:
        json.dump(hyperparameters, file)

    print("Train network...", experiment_name)
    model.train(
        data.training, data.validation, epochs=config['training_epochs'],
        class_weights=data.class_weights if config['class_weights'] else None, 
        description=str(hyperparameters)
    )
    print("Save network...", experiment_name)
    model.save(join(experiment_dir, 'model.h5'))

    print('Results on validation dataset:', experiment_name)
    model.evaluate(
        data.validation, classes,
        results_file=join(experiment_dir, 'results-validation.json'),
        plot_file=join(experiment_dir, 'results-validation.png'),
    )

    print('Results on testing dataset:', experiment_name)
    model.evaluate(
        data.testing, classes,
        results_file=join(experiment_dir, 'results-testing.json'),
        plot_file=join(experiment_dir, 'results-testing.png'),
    )
    
    backend.clear_session()
