# -*- coding: utf-8 -*-
"""

@author: Rem
@contack: remch183@outlook.com
@time: 2017/02/04/ 16:35 
"""
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import keras
import re
import numpy as np
import os
from cat_read_data import get_test, get_train_val_generator


def get_model_v0():
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(256, 256, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    adamax = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy', 'categorical_crossentropy'])
    return model


def get_model_v1():
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(256, 256, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 1, 1, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(256, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # adamax = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    optimizer = keras.optimizers.RMSprop(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', 'categorical_crossentropy'])

    return model


def standerize_images(X):
    X = X.astype(np.float16)
    X = (X - 128.) / 256.
    return X


def train(model, start=0, version=''):
    batch_size = 16
    validation_index = batch_size*10 * (2000 // (batch_size * 10))
    save_step = 5
    length = 25000
    train_gen, val_gen = get_train_val_generator(validation_index=validation_index, batchsize=batch_size)

    def scheduler(epoch):
        if epoch >= 39:
            return 1e-5
        return 1e-4

    callback = keras.callbacks.TensorBoard(log_dir='./logs'+version, histogram_freq=1, write_graph=True, write_images=False)
    change_lr = keras.callbacks.LearningRateScheduler(scheduler)

    for i in range(start, start+save_step*20, save_step):
        model.fit_generator(generator=train_gen(),
                            samples_per_epoch=(length - validation_index),
                            nb_epoch=i+save_step,
                            initial_epoch=i,
                            callbacks=[callback, change_lr],
                            # callbacks=[callback, ],
                            validation_data=val_gen(),
                            nb_val_samples=validation_index )
        model.save('models_' + version + '/trained_model_%05d.model' % (i+save_step, ))
    return model


def main(load=True, version=''):
    models_root = 'models_' + version + '/'
    models_path = os.listdir(models_root)
    if load and models_path:
        models_path = models_root + models_path[-1]
        print('\n\n\n========================')
        print('Loading model from file: ' + models_path)
        start = re.findall(r'\d+', models_path)[-1]
        start = int(start.lstrip('0'))
        print('Start from epoch: ' + str(start))
        print('========================\n\n\n')
        model = load_model(models_path)
    else:
        model = get_model_v1()
        start = 0
    print('Model loaded.')
    print("Parameters num: " + str(model.count_params()))
    print(model.summary())
    model = train(model, start, version=version)
    print('Finish!')
    del model
    return

if __name__ == '__main__':
    main(version='v1')



