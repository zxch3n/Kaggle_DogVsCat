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
from cat_read_data import get_train, get_test

def get_model():
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(128, 128, 3)))
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


def standerize_images(X):
    X = X.astype(np.float16)
    X = (X - 128.) / 256.
    return X


def train(model, start=0):
    batch_size = 32
    validation_index = 1200
    save_step = 20
    print('Getting Training Data.')
    X_train, Y_train = get_train()

    def generator(X_input, Y_input):
        while True:
            for idx in range(0, len(X_input), batch_size):
                X = X_input[idx: min(len(X_input)-1, idx + batch_size)]
                Y = Y_input[idx: min(len(X_input)-1, idx + batch_size)]
                X = standerize_images(X)
                yield X, Y
    print('\rGot Training Data.')

    def scheduler(epoch):
        if epoch >= 50:
            return 0.0002
        return 0.002
    callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
    change_lr = keras.callbacks.LearningRateScheduler(scheduler)

    for i in range(start, start+20, save_step):
        model.fit_generator(generator=generator(X_train[validation_index:], Y_train[validation_index:]),
                            samples_per_epoch=(len(X_train) - validation_index)//batch_size,
                            nb_epoch=i+save_step,
                            initial_epoch=i,
                            max_q_size=2,
                            callbacks=[callback, change_lr],
                            validation_data=generator(X_train[:validation_index], Y_train[:validation_index]),
                            nb_val_samples=validation_index // batch_size,)
        model.save('models/trained_model_%05d.model' % (i+save_step, ))
    return model


def main(load=True):
    print('Model loaded.')
    models_path = os.listdir('models')
    if load and models_path:
        models_path = 'models/' + models_path[-1]
        print('Loading model from file: ' + models_path)
        model = load_model(models_path)
        start = re.findall(r'\d+', models_path)[0]
        start = int(start)
    else:
        model = get_model()
        start = 0
    model = train(model, start)
    print('Finish!')
    del model
    return

if __name__ == '__main__':
    main()



