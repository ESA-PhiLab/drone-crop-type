"""Train a CNN on the patches extracted from the drone mosaics

To set the paths to the patches, please create your own configurations file

Usage:
    > python train.py CONFIG_FILE
"""

from glob import glob
import os
from os.path import basename, dirname, join
from pprint import pprint
import sys
from time import time

import numpy as np
from imgaug import augmenters as iaa
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras import callbacks, optimizers
from tensorflow.python.ops import summary_ops_v2
import tensorflow as tf
import yaml

from ai4eo.preprocessing import ImageLoader
import models

class AdvancedMetrics(callbacks.Callback):
    def __init__(self, logdir, data_generator):
        super().__init__()
        self.data_generator = data_generator
        self.logdir = logdir
        self.logstep = 0
        return
    
    def on_train_begin(self, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        path = os.path.join(self.logdir, 'validation')
        writer = summary_ops_v2.create_file_writer_v2(path)
        
        steps = len(self.data_generator)
        gen = iter(self.data_generator)
        y_true = np.empty(steps*self.data_generator.batch_size)
        y_pred = np.empty(steps*self.data_generator.batch_size)
        
        for i in range(steps):
            self.logstep += self.data_generator.batch_size
            
            x, y_t = next(gen)
            y_p = self.model.predict_on_batch(x)
            y_t = np.argmax(y_t,axis=-1).ravel()
            y_p = np.argmax(y_p,axis=-1).ravel()
            y_true[i*y_t.size:(i+1)*y_t.size] = y_t
            y_pred[i*y_p.size:(i+1)*y_p.size] = y_p
    
        accuracy = accuracy_score(y_true,y_pred)
        precision,recall,f1,_ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        with writer.as_default():
            tf.summary.scalar('accuracy', accuracy, step=self.logstep)
            tf.summary.scalar('precision', precision, step=self.logstep)
            tf.summary.scalar('recall', recall, step=self.logstep)
            tf.summary.scalar('f1', f1, step=self.logstep)
            
        print("\n val_f1: {:.3f} — val_pre: {:.3f} — val_rec {:.3f}".format(
            f1, precision, recall
        ))
        return


def get_patches(filenames, classes):
    folds = []
    for filename in filenames:
        with open(filename) as file:
            folds.extend(yaml.safe_load(file))
    
    labels, images = zip(*[
        (basename(dirname(image)), image)
        for fold in folds
        for prefix in fold['prefixes']
        for image in glob(join(fold['path'], prefix)+'_*.png')
        if basename(dirname(image)) in classes
    ])
    return images, labels

    
def train(config):
    # Just in case if we need a on-the-fly augmentator
    augmentator = iaa.SomeOf((0, None), [
        iaa.Add((-20, 20)),
        iaa.Crop(percent=(0, 0.02)),
        iaa.Affine(
            scale=(0.7, 1.3),
            rotate=(-20, 20), mode='reflect'),
        iaa.Fliplr(0.25), # horizontally flip 25% of the images
        iaa.Flipud(0.25),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0, 0.8)), # blur images with a sigma of 0 to 3.0
    ])
    
    preprocess_input = getattr(models, config['model']+'_preprocess_input')

    train_images, train_labels = get_patches(config['training_folds'], config['classes'])
    train_loader = ImageLoader(
        images=train_images, 
        labels=train_labels,
        augmentator=augmentator if config['augmentation'] else None, 
        balance=config['balance_training_data'],
        preprocess_input=preprocess_input,
        classes=config['classes'],
        batch_size=config['batch_size'],
    )
    val_images, val_labels = get_patches(config['validation_folds'], config['classes'])
    val_loader = ImageLoader(
        images=val_images, 
        labels=val_labels,
        preprocess_input=preprocess_input,
        classes=config['classes'],
        batch_size=config['batch_size'],
    )

    print('Training samples:', len(train_images))
    print('Validation samples:', len(val_images))
    import numpy as np
#     for _, labels in train_loader:
#         print(np.unique(np.argmax(labels, axis=-1), return_counts=True))

    # Let's set our own class weights:
    # data.class_weights = {
    #     0: 8., # cassava
    #     1: 5, # groundnut
    #     2: 1.2, # maize
    #     3: 1.0, # other
    #     4: 15, # sweetpotatoes
    #     5: 7.0 # tobacco
    # }

    os.makedirs(config['results_dir'], exist_ok=True)

    # Load the model architecture:
    print("Load model...")
    model_loader = getattr(models, config['model'])
    model = model_loader(**config['model_options'])

    optimizer = getattr(optimizers, config['optimizer'])(**config['optimizer_options'])

    model.compile(
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'],
        optimizer=optimizer
    )
    
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print(model.summary())

    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
        ),
        callbacks.TensorBoard(
            log_dir=join(config['results_dir'], 'tb_logs', config['name']), histogram_freq=0,
            write_graph=False, write_images=False,
        ),
#         callbacks.ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=3,
#             verbose=1,
#             mode='auto',
#             min_delta=0.00002,
#             cooldown=0,
#             min_lr=0.00002
#         ),
        callbacks.ModelCheckpoint(
            join(config['results_dir'], 'models', config['name']),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=1
        ),
        AdvancedMetrics(
            logdir=join(config['results_dir'], 'tb_logs', config['name']),
            data_generator=val_loader,
        )
    ]

    print("Train model...")
    model.fit_generator(
        train_loader,
        steps_per_epoch=len(train_loader) if config['training_steps'] is None else config['training_steps'],
        validation_data=val_loader,
        validation_steps=len(val_loader) if config['validation_steps'] is None else config['validation_steps'],
        epochs=config['training_epochs'],
        use_multiprocessing=True,
        workers=2,
        callbacks=callback_list,
    #     class_weight=class_weights,
        verbose=config['verbose'],
    )

    print("Save model...")
    model_dir = join(config['results_dir'], 'models', config['name'])
    os.makedirs(model_dir, exist_ok=True)
    model.save_weights(join(model_dir, 'best.h5'))
    with open(join(model_dir, 'config.yml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # print('Results on validation dataset:', experiment_name)
    # model.evaluate(
    #     data.validation, classes,
    #     results_file=join(experiment_dir, 'results-validation.json'),
    #     plot_file=join(experiment_dir, 'results-validation.png'),
    # )

    # print('Results on testing dataset:', experiment_name)
    # model.evaluate(
    #     data.testing, classes,
    #     results_file=join(experiment_dir, 'results-testing.json'),
    #     plot_file=join(experiment_dir, 'results-testing.png'),
    # )


if __name__ == '__main__':
    # Load the config file:
    if len(sys.argv) < 2:
        print('You have to provide a path to the configurations file!')
        exit()

    try:
        with open(sys.argv[1]) as config_file:
            config = yaml.load(config_file)
    except Exception as e:
        print('Could not load configurations file')
        raise e

    print('Start experiment', config['name'])
    pprint(config)
    train(config)