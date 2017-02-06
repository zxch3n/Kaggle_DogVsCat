import numpy as np
import random
import os
from PIL import Image
from sklearn.cross_validation import train_test_split
import pickle


def get_files_path(shuffle=True, seed=None):
    if seed:
        random.seed(seed)
    root_path = 'input/my_train/'
    files_path = [root_path + i for i in os.listdir(root_path)]
    if shuffle:
        random.shuffle(files_path)
    return files_path


def get_train_val_generator(validation_index=None, val_split=0.15, batchsize=32):
    files = get_files_path()
    if validation_index is None:
        validation_index = int(float(len(files)) * val_split)

    def train():
        while True:
            gen = get_xy_data(files[validation_index:], True, batchsize*10)
            for X_train, y_train in gen:
                for idx in range(0, len(X_train), batchsize):
                    next_X_train = X_train[idx:idx+batchsize]
                    next_y_train = y_train[idx:idx+batchsize]
                    next_X_train = next_X_train.astype(np.float16)
                    next_X_train = (next_X_train - 128.) / 128.
                    yield next_X_train, next_y_train

    def val():
        while True:
            gen = get_xy_data(files[:validation_index], False, batchsize*10)
            for X_train, y_train in gen:
                for idx in range(0, len(X_train), batchsize):
                    next_X_train = X_train[idx:idx+batchsize]
                    next_y_train = y_train[idx:idx+batchsize]
                    next_X_train = next_X_train.astype(np.float16)
                    next_X_train = (next_X_train - 128.) / 128.
                    yield next_X_train, next_y_train
    return train, val


def get_xy_data(files_path, train=True, batchsize=320):
    """return X_train, Y_train"""

    train_pickle = 'data_pickle/'
    if any((str(batchsize) in i for i in os.listdir(train_pickle))):
        for picklename in os.listdir(train_pickle):
            if (train and 'train' in picklename) or (not train and 'val' in picklename):
                with open(train_pickle + picklename, 'rb') as f:
                    yield pickle.load(f)

    def get_label(name):
        if 'cat' in name:
            return [0, 1]
        return [1, 0]

    counter = 0
    if train:
        this_type = 'train'
    else:
        this_type = 'val'
    for i in range(0, len(files_path), batchsize):
        counter += 1
        X_train = []
        Y_train = []
        for j in range(i, min(i+batchsize, len(files_path))):
            new_img = Image.open(files_path[j])
            if new_img.size != (256, 256):
                continue
            X_train.append(np.array(new_img))
            Y_train.append(get_label(files_path[j]))
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        yield X_train, Y_train
        with open(train_pickle + this_type + '%d_%d.pickle' % (batchsize, counter), 'wb') as f:
            pickle.dump((X_train, Y_train), f)
        del X_train, Y_train


def get_test():
    """return X_test"""
    root_path = 'input/my_test/'
    files_path = [root_path + i for i in os.listdir(root_path)]
    X_test = np.array(list(map(lambda x: np.array(Image.open(x)), files_path)))
    for i, v in enumerate(X_test):
        if v.shape != (128, 128, 3):
            raise ValueError("Shape not equal to (128, 128, 3) Error!!!")
    return X_test
