"""Define different model structures for crop type classification

"""

import json
import os

import tensorflow as tf
from tensorflow.keras import callbacks as keras_callbacks
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.xception import Xception
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


class AdvancedMetrics(keras_callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.validation_data = val_data
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        b_predictions, b_targets = None, None
        for b, batch in enumerate(self.validation_data):
            if b > len(self.validation_data):
                break

            inputs, targets = batch
            predictions = self.model.predict_on_batch(inputs)
            targets, predictions = \
                np.argmax(targets, 1), np.argmax(predictions, 1)

            if b_predictions is None:
                b_predictions = predictions
                b_targets = targets
            else:
                b_predictions = np.concatenate([b_predictions, predictions])
                b_targets = np.concatenate([b_targets, targets])

        _val_f1 = f1_score(b_targets, b_predictions, average='macro')
        _val_recall = recall_score(b_targets, b_predictions, average='macro')
        _val_precision = precision_score(b_targets, b_predictions, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("\n val_f1: {:.3f} — val_precision: {:.3f} — val_recall {:.3f}".format(
            _val_f1, _val_precision, _val_recall
        ))
        return


class PretrainedModel:
    def __init__(self, model=None, output_dir=None, optimizer=None, frozen_layers=None, **kwargs):
        self.output_dir = output_dir
        self.optimizer = optimizer or optimizers.RMSprop()

        if output_dir is not None:
            os.makedirs(self.output_dir+'/logs', exist_ok=True)

        # create model:
        model_initializer = {
            'xception': init_pretrained_xception,
            'baseline': init_baseline,
            'effnet': init_effnet,
        }

        if model in model_initializer:
            self.model = model_initializer[model](**kwargs)
        elif os.path.isfile(model):
            self.model = models.load_model(model)
        else:
            print(f'Could not load model {model}')

        # Freeze some layers:
        if frozen_layers is not None:
            for layer in self.model.layers[:frozen_layers]:
                layer.trainable = False
            for layer in self.model.layers[frozen_layers:]:
                layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        self.model.compile(
            optimizer=self.optimizer, loss='categorical_crossentropy',
        )

    def train(
            self, training_data, validation_data, epochs=40, callbacks=None,
            class_weights=None, description='None'
    ):

        if callbacks is None:
            callbacks = [
                keras_callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True,
                ),
                keras_callbacks.TensorBoard(
                    log_dir=self.output_dir+'/logs', histogram_freq=0,
                    write_graph=False, write_images=False,
                ), 
                AdvancedMetrics(validation_data)
            ]

        self.model.fit_generator(
            training_data,
            steps_per_epoch=len(training_data),
            validation_data=validation_data,
            validation_steps=len(validation_data),
            epochs=epochs,
            use_multiprocessing=True,
            workers=4,
            callbacks=callbacks,
            class_weight=class_weights,
        )

    def predict(self, data):
        return self.model.predict_generator(
            data, steps=len(data) // data.batch_size
        )

    def save(self, filename):
        self.model.save(filename)

    def evaluate(self, data, classes, report=True, results_file=None,
                 plot_file=True,):
        b_predictions, b_targets = None, None
        for b, batch in enumerate(data):
            if b > len(data):
                break

            inputs, targets = batch
            predictions = self.model.predict_on_batch(inputs)
            targets, predictions = \
                np.argmax(targets, 1), np.argmax(predictions, 1)

            if b_predictions is None:
                b_predictions = predictions
                b_targets = targets
            else:
                b_predictions = np.concatenate([b_predictions, predictions])
                b_targets = np.concatenate([b_targets, targets])

        results = {
            'predictions': b_predictions.tolist(),
            'targets': b_targets.tolist(),
            'classes': classes
        }

        if report:
            print(classification_report(
                b_targets, b_predictions, labels=list(range(len(classes))),
                target_names=classes
            ))
            print(confusion_matrix(
                b_targets, b_predictions,
                labels=list(range(len(classes)))
            ))

        if results_file is not None:
            with open(results_file, 'w') as file:
                json.dump(results, file)

        if plot_file is not None:
            fig, axes = plt.subplots(ncols=2, figsize=(15, 7))
            plot_confusion_matrix(
                results['targets'], results['predictions'], results['classes'],
                ax=axes[0]
            )
            plot_confusion_matrix(
                results['targets'], results['predictions'], results['classes'],
                ax=axes[1], normalize=True
            )
            fig.tight_layout()
            fig.savefig(plot_file)

        return results


def init_pretrained_xception(num_outputs, input_shape):
    # create the base pre-trained model
    base_model = Xception(
        weights='imagenet', include_top=False, input_shape=input_shape,
    )
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    # let's add a fully-connected layer
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(num_outputs, activation='relu')(x)
    
    # and a softmax layer
    predictions = layers.Dense(num_outputs, activation='softmax')(x)
    
    # this is the model we will train
    return models.Model(inputs=base_model.input, outputs=predictions)


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