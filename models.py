"""Define different model structures for crop type classification

"""

import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow.keras.applications.xception as keras_xception
import tensorflow.keras.applications.vgg16 as keras_vgg16
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score


def plot_confusion_matrix(
        y_true, y_pred, classes, normalize=False, 
        title=None, cmap=plt.cm.Blues, ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    return ax
    
def add_classifier(x, num_outputs, layer_sizes):
    
    for layer_size in layer_sizes:
        if len(layer_size) == 1:
            x = layers.Dense(layer_size[0], activation='relu')(x)
        else:
            x = layers.Dropout(layer_size[0])(x)
            x = layers.Dense(layer_size[1], activation='relu')(x)
    
    return layers.Dense(num_outputs, activation='softmax')(x)

def freeze_layers(model, n_layers):
    if n_layers is not None or n_layers == 0:
        for layer in model.layers[:n_layers]:
            layer.trainable = False
        for layer in model.layers[n_layers:]:
            layer.trainable = True
    
def xception(input_shape, frozen_layers=0, weights=None, pooling='avg', **kwargs):
    # create the base model
    base_model = keras_xception.Xception(
        weights=weights, include_top=False, input_shape=input_shape,
        pooling=pooling
    )
    predictions = add_classifier(base_model.output, **kwargs)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze some layers:
    freeze_layers(model, frozen_layers)
    
    return model

def xception_preprocess_input(*args, **kwargs):
    return keras_xception.preprocess_input(*args, **kwargs)

def vgg16(input_shape, frozen_layers=0, weights=None, pooling='avg', **kwargs):
    # create the base model
    base_model = keras_vgg16.VGG16(
        weights=weights, include_top=False, input_shape=input_shape,
        pooling=pooling
    )
    predictions = add_classifier(base_model.output, **kwargs)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze some layers:
    freeze_layers(model, frozen_layers)
    
    return model

def vgg16_preprocess_input(*args, **kwargs):
    return keras_vgg16.preprocess_input(*args, **kwargs)

def init_baseline(num_outputs, input_shape):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(64, (5, 5))(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (5, 5))(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (1, 1))(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    predictions = layers.Dense(num_outputs, activation='softmax')(x)
    print()
    
    # this is the model we will train
    return models.Model(inputs=inputs, outputs=predictions)


def get_effnet_block(x_in, ch_in, ch_out):
    x = layers.Conv2D(ch_in,
                   kernel_size=(1, 1),
                   padding='same',
                   use_bias=False)(x_in)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.DepthwiseConv2D(kernel_size=(1, 3), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 1),
                      strides=(2, 1))(x) # Separable pooling

    x = layers.DepthwiseConv2D(kernel_size=(3, 1),
                            padding='same',
                            use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(ch_out,
                   kernel_size=(2, 1),
                   strides=(1, 2),
                   padding='same',
                   use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    return x


def init_effnet(num_outputs, input_shape):
    inputs = layers.Input(shape=input_shape)

    x = get_effnet_block(inputs, 32, 64)
    x = get_effnet_block(x, 64, 128)
    x = get_effnet_block(x, 128, 256)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(num_outputs, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=x)