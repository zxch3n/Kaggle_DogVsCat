import numpy as np
import os
from PIL import Image
from sklearn.cross_validation import train_test_split
import pickle


def get_train(shuffle=True, seed=None):
    """return X_train, Y_train"""

    data_save = 'train_data.pickle'
    if os.path.exists(data_save):
        with open(data_save, 'rb') as f:
            X_train, Y_train = pickle.load(f)
        return X_train, Y_train
    root_path = 'input/my_train/'
    files_path = [ root_path + i for i in os.listdir(root_path)]

    def get_label(name):
        if 'cat' in name:
            return [0, 1]
        return [1, 0]
    Y_train = np.array(list(map(get_label, files_path)))
    X_train = np.array(list(map(lambda x: np.array(Image.open(x)), files_path)))

    mask = np.ones(X_train.shape[0]).astype(np.bool)
    for i,v in enumerate(X_train):
        if v.shape != (128, 128, 3):
            mask[i] = False
    X_train = np.array(X_train[mask])
    Y_train = Y_train[mask]
    X_train = np.array(X_train.tolist())
    if shuffle:
        if not seed:
            X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=0.0)
        else:
            X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=0.0, random_state=seed)
    with open(data_save, 'wb') as f:
        pickle.dump([X_train, Y_train], f)
    return X_train, Y_train


def get_test():
    """return X_test"""
    root_path = 'input/my_test/'
    files_path = [ root_path + i for i in os.listdir(root_path)]
    X_test = np.array(list(map(lambda x: np.array(Image.open(x)), files_path)))
    for i,v in enumerate(X_test):
        if v.shape != (128, 128, 3):
            raise ValueError("Shape not equal to (128, 128, 3) Error!!!")
    return X_test
