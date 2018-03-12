# Deep Learning Quick Reference Chapter 8: Transfer Learning
# Mike Bernico <mike.bernico@gmail.com>


# seed random number generators before importing keras
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard, EarlyStopping, CSVLogger, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import os
import argparse


def build_model_feature_extraction():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_model_fine_tuning(model, learning_rate=0.0001, momentum=0.9):

        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True
        model.compile(optimizer=SGD(lr=learning_rate, momentum=momentum), loss='binary_crossentropy', metrics=['accuracy'])
        return model


def create_callbacks(name):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log", name), write_graph=True, write_grads=False)
    checkpoint_callback = ModelCheckpoint(filepath="./model-weights" + name + ".{epoch:02d}-{val_loss:.6f}.hdf5", monitor='val_loss',
                                          verbose=0, save_best_only=True)
    return [tensorboard_callback, checkpoint_callback]


def setup_data(train_data_dir, val_data_dir, img_width=299, img_height=299, batch_size=16):
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    return train_generator, validation_generator


def main():

    data_dir = "data/train/"
    val_dir = "data/val/"
    epochs = 10
    batch_size = 30
    model = build_model_feature_extraction()
    train_generator, val_generator = setup_data(data_dir, val_dir)
    callbacks_fe = create_callbacks(name='feature_extraction')
    callbacks_ft = create_callbacks(name='fine_tuning')

    # stage 1 fit
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.n // batch_size,
        callbacks=callbacks_fe,
        verbose=1)

    scores = model.evaluate_generator(val_generator, steps=val_generator.n // batch_size)
    print("Step 1 Scores: Loss: " + str(scores[0]) + " Accuracy: " + str(scores[1]))

    # stage 2 fit
    model = build_model_fine_tuning(model)
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.n // batch_size,
        callbacks=callbacks_ft,
        verbose=2)

    scores = model.evaluate_generator(val_generator, steps=val_generator.n // batch_size)
    print("Step 2 Scores: Loss: " + str(scores[0]) + " Accuracy: " + str(scores[1]))


if __name__ == "__main__":
    main()


