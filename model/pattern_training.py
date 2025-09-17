import os

import tensorflow as tf

from .architecture import neurite_classifier
from .data import prepare_pattern_dataset


def train_pattern_vs_unpatterned(train_image_dir='PatternTrainImages', validation_image_dir='PatternValidationImages', as_generic=False):
    """This method trains the model to distinguish between patterned and unpatterned images.
    As auxiliary classifications, this method also trains the model to distinguish between different sex and genotype categories.
    """
    train_data = prepare_pattern_dataset(base_image_dir=train_image_dir, batch_size=4, is_training=True)
    validation_data = prepare_pattern_dataset(base_image_dir=validation_image_dir, batch_size=4, is_training=False)
    model = neurite_classifier(out_classes=3, out_layers=1, dense_size=128, generic=as_generic)
    train_model(model,
                train_data,
                validation_data,
                epochs=300,
                num_classes=3,
                label_smoothing=0.05,
                training_log='PatternvsUnpatterned_train_log',
                validation_log='PatternvsUnpatterned_val_log',
                model_save_path='PatternvsUnpatterned_model.tf')


def train_model(model, train_data, validation_data, learning_rate=1e-3, epochs=300, num_classes=3, label_smoothing=0.05, log_path='Pattern_train_log', checkpoint_dir='chk', model_save_path='model_weights.tf'):
    """Train the model.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The model to train
    train_data : tf.data.Dataset
        The training data
    validation_data : tf.data.Dataset | None
        Optional validation data
    learning_rate : float
        The learning rate
    epochs : int
        The total number of epochs to train for
    num_classes : int
        The number of predicted classes
    label_smoothing : float
        The amount of label smoothing to (0.1 smooths "true" labels to 0.1 and 0.9)
    log_path : str
        The output location for the tensorboard log file
    model_save_path : str
        The output location for the trained model
    """

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=label_smoothing)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_path),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'model_weights_e{epoch}'), save_best_only=True, mointor='val_loss', save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    ]
    metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]
    model.compile(loss=loss_func, optimizer=opt, metrics=metrics)
    model.fit(train_data, validation_data=validation_data, epochs=epochs, initial_epoch=0, callbacks=callbacks)
    model.save_weights(model_save_path)
