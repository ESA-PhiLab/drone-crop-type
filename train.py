"""Train a CNN on the patches extracted from the drone mosaics

To create the CONFIG_FILE, please use create_config

Usage:
    > python train.py CONFIG_FILE
"""

from glob import glob, iglob
import os
from os.path import basename, dirname, exists, join
from pprint import pprint
import sys
from time import time

from ai4eo.preprocessing import ImageLoader
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras import callbacks, optimizers
from tensorflow.python.ops import summary_ops_v2
import tensorflow as tf
import yaml

import models
from utils import plot_confusion_matrix

class AdvancedMetrics(callbacks.Callback):
    def __init__(self, logdir, plotdir, data_generator, classes):
        super().__init__()
        self.data_generator = data_generator
        self.logdir = logdir
        self.plotdir = plotdir
        self.logstep = 0
        self.classes = classes
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
            
            if len(self.classes) > 2:
                y_t = np.argmax(y_t,axis=-1).ravel()
                y_p = np.argmax(y_p,axis=-1).ravel()
            else:
                y_t = np.squeeze(y_t)
                y_p = (np.squeeze(y_p) > 0.5).astype(int)
            
            y_true[i*y_t.size:(i+1)*y_t.size] = y_t
            y_pred[i*y_p.size:(i+1)*y_p.size] = y_p
    
        filename = join(self.logdir, "validation.csv")
        if exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=['step', 'f1', 'precision', 'recall', 'accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'])
        metric = {k: 0 for k in df.columns}
        metric['step'] = self.logstep
    
        accuracy = accuracy_score(y_true, y_pred)
        precision,recall,f1,_ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        metric['precision'] = precision
        metric['recall'] = recall
        metric['f1'] = f1
        metric['accuracy'] = accuracy
        with writer.as_default():
            tf.summary.scalar('accuracy', accuracy, step=self.logstep)
            tf.summary.scalar('precision', precision, step=self.logstep)
            tf.summary.scalar('recall', recall, step=self.logstep)
            tf.summary.scalar('f1', f1, step=self.logstep)
        print(f"\nValidation: f1: {f1:.3f} — pre: {precision:.3f} — rec {recall:.3f}")
        cm_title_1 = f"[macro] f1: {f1:.2f} — pre: {precision:.2f} — rec {recall:.2f}"
        
        precision,recall,f1,_ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        metric['precision_weighted'] = precision
        metric['recall_weighted'] = recall
        metric['f1_weighted'] = f1
        with writer.as_default():
            tf.summary.scalar('weighted_precision', precision, step=self.logstep)
            tf.summary.scalar('weighted_recall', recall, step=self.logstep)
            tf.summary.scalar('weighted_f1', f1, step=self.logstep)
        print(f"\n weighted: f1: {f1:.3f} — pre: {precision:.3f} — rec {recall:.3f}")
        cm_title_2 = f"[weighted] f1: {f1:.2f} — pre: {precision:.2f} — rec {recall:.2f}"
        
        os.makedirs(self.plotdir, exist_ok=True)
        fig, ax = plt.subplots(ncols=2, figsize=(12, 12))
        plot_confusion_matrix(y_true, y_pred, self.classes, ax=ax[0], title=cm_title_1)
        plot_confusion_matrix(y_true, y_pred, self.classes, ax=ax[1], title=cm_title_2, normalize=True)
        fig.tight_layout()
        fig.savefig(join(self.plotdir, f'cm_{int(self.logstep/(steps*self.data_generator.batch_size)):03d}.png'))
        plt.close(fig)
        
        df = df.append(metric, ignore_index=True)
        df.to_csv(filename, index=False)
        
        return

    
# def parse_fold(fold):
#     if "path" in fold:
#         placeholder = list(set(re.findall(r"{(\w+)}", fold['path'])))
#         if placeholder:
#             for image in glob(fold['path'].format(placeholder), prefix)):
#                 yield basename(dirname(image)), image
#         else:
#             for image in glob(fold['path']):
#                 yield 

def get_patches(fold_files, class_mapping):
    folds = []
    for fold_file in fold_files:
        with open(fold_file) as file:
            this_fold = []
            for line in file.readlines():
                if "*" in line:
                    this_fold.extend(glob(line))
                else:
                    this_fold.append(line)
            folds.extend(this_fold)
    
    labels, images = zip(*[
        (class_mapping[basename(dirname(image))], image)
        for image in folds
        if basename(dirname(image)) in class_mapping
    ])
    
    return images, labels

    
def train(config):
    # Just in case if we need a on-the-fly augmentator
    augmentator = iaa.SomeOf((0, None), [
        iaa.Add((-40, 40)),
        iaa.Affine(
            scale=(0.7, 1.3),
            rotate=iap.Choice([0, 90, 180, -90]), mode='reflect'),
        iaa.Fliplr(0.25), # horizontally flip 25% of the images
        iaa.Flipud(0.25),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0, 0.8)), # blur images with a sigma of 0 to 3.0
    ])
    
    # set a default class mapping:
    if not config['class_mapping']:
        config['class_mapping'] = {k: k for k in config['classes']}
    
    preprocess_input = getattr(models, config['model']+'_preprocess_input')
    
    train_images, train_labels = get_patches(config['training_folds'], config['class_mapping'])
    train_loader = ImageLoader(
        images=train_images, 
        labels=train_labels,
        augmentator=augmentator if config['augmentation'] else None, 
        balance=config['balance_training_data'],
        preprocess_input=preprocess_input,
        classes=config['classes'],
        label_encoding='binary',
        batch_size=config['batch_size'],
    )
    val_images, val_labels = get_patches(config['validation_folds'], config['class_mapping'])
    val_loader = ImageLoader(
        images=val_images, 
        labels=val_labels,
        preprocess_input=preprocess_input,
        classes=config['classes'],
        label_encoding='binary',
        batch_size=config['batch_size'],
    )

    print('Training samples:', len(train_images))
    print('Validation samples:', len(val_images))
    
    os.makedirs(config['results_dir'], exist_ok=True)

    # Load the model architecture:
    print("Load model...")
    model_loader = getattr(models, config['model'])
    model = model_loader(**config['model_options'])

    optimizer = getattr(optimizers, config['optimizer'])(**config['optimizer_options'])

    if len(config['classes']) == 2:
        model.compile(
            loss='binary_crossentropy',
            metrics=['binary_accuracy'],
            optimizer=optimizer
        )
    else:
        model.compile(
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'],
            optimizer=optimizer
        )
    print(model.summary())

    callback_list = [
#         callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=30,
#         ),
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
            plotdir=join(config['results_dir'], 'plots', config['name']),
            data_generator=val_loader, classes=config['classes']
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