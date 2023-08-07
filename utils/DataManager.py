# @Author   : ChaoQiezi
# @Time     : 2023/8/3  15:44
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to manage the Data
"""

import os
import glob

import h5py
import numpy as np
import keras
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from skimage.util import view_as_windows
from utils.utils import read_img


class DataManager:

    def __init__(self):
        self.img = type('', (), {})()  # create an empty class
        self.img.nodata = None
        self.img.rows = None
        self.img.cols = None
        self.img.transform = None
        self.img.projection = None
        self.time_steps = None

    def load_data(self, dataset_dir, wildcard='*.tif', save_path=None):
        """

        :param dataset_dir:
        :return:
        """

        dataset_paths = glob.iglob(os.path.join(dataset_dir, wildcard))
        datasets = []
        for dataset_path in dataset_paths:
            # get img and info
            dataset, img_info = read_img(dataset_path)

            # set nodata
            self.img.nodata = img_info[2]
            if self.img.nodata is np.nan:
                continue
            else:
                dataset[dataset == self.img.nodata] = np.nan

            # add
            datasets.append(dataset)
            print('>>> {} is loaded.'.format(os.path.basename(dataset_path)))
        if len(datasets) == 0:
            raise ValueError("There is no {} image in the {}".format(wildcard, dataset_dir))
        # get img info
        self.time_steps = len(datasets)
        self.img.rows = img_info[0]
        self.img.cols = img_info[1]
        self.img.transform = img_info[3]
        self.img.projection = img_info[4]

        datasets = np.array(datasets)

        if save_path:
            np.save(save_path, datasets)

        return datasets

    def read_img(self, img_path):
        return read_img(img_path)

    def nomalize(self, dataset, train_flag=False):

        if train_flag:
            for i in range(dataset.shape[-1]):  # dataset.shape = (None, rows, cols, bands)
                dataset[:, :, :, i] = (dataset[:, :, :, i] - np.nanmin(dataset[:, :, :, i])) \
                                      / (np.nanmax(dataset[:, :, :, i]) - np.nanmin(dataset[:, :, :, i]))
        else:
            dataset = (dataset - np.nanmin(dataset)) / (np.nanmax(dataset) - np.nanmin(dataset))

        return dataset

    def split_data(self, features, labels, test_size=0.2, random_state=0):

        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=test_size, random_state=random_state)

        return features_train, features_test, labels_train, labels_test


class GenerateData(Sequence):

    def __init__(self, config: dict, is_train: bool, batch_size=512, dimension=(30, 11, 11), channels=2,
                 n_class=None, shuffle=True):

        self.config = config
        self.is_train = is_train
        self.batch_size = batch_size
        self.dimension = dimension
        self.channels = channels
        self.n_class = n_class
        self.shuffle = shuffle
        if self.is_train:
            self.n_samples = self.config['train_samples']
        else:
            self.n_samples = self.config['test_samples']
        self.on_epoch_end()  # on_epoch_end() means that the function will be executed after each epoch

    def __len__(self):
        return self.n_samples // self.batch_size  # get the number of batches per epoch

    def __getitem__(self, index):
        # generate indexes of the batch
        batch_indexes = self.epoch_indexes[(index * self.batch_size):((index + 1) * self.batch_size)]

        # generate data
        features, labels = self.__data_generation(batch_indexes)

        return features, labels

    def on_epoch_end(self):
        self.epoch_indexes = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.epoch_indexes)

    def __data_generation(self, batch_indexes: list):
        # because reading an h5 file through  the indexes  requires that the indexes are in increasing order
        batch_indexes.sort()
        # open h5 file
        with h5py.File(self.config['features_labels_path'], 'r') as f:
            if self.is_train:
                features = f['features_train'][batch_indexes, :, :, :,
                           :]  # shape = (batch_size, time_steps, rows, cols, bands)
                labels = f['labels_train'][batch_indexes]  # shape = (batch_size, )
            else:
                features = f['features_test'][batch_indexes, :, :, :, :]
                labels = f['labels_test'][batch_indexes]
        if self.n_class is None:
            self.n_class = len(np.unique(labels))
        return features, keras.utils.to_categorical(labels, num_classes=self.n_class)  # one-hot encoding


class PredictGenerator(Sequence):

    def __init__(self, predict_data_path):
        self.predict_data_path = predict_data_path
        with h5py.File(self.predict_data_path, 'r') as f:
            self.rows = range(5, f['features'].shape[1] - 5)

    def __len__(self):
        # return len(self.rows)
        return 20

    def __getitem__(self, index):
        row = self.rows[index]
        return self.__data_generation(row)

    def __data_generation(self, row):
        with h5py.File(self.predict_data_path, 'r') as f:
            predict_data = f['features'][:]
            predict_data = view_as_windows((predict_data[:, row - 5:row + 6, :, :]), (30, 11, 11, 2)).squeeze()
        return predict_data


class PredictGenerator2(Sequence):

    def __init__(self, predict_data):
        self.predict_data = predict_data
        self.rows = range(5, predict_data.shape[1] - 5)

    def __len__(self):
        # return len(self.rows)
        return 10

    def __getitem__(self, index):
        row = self.rows[index]
        return view_as_windows((self.predict_data[:, row - 5:row + 6, :, :]), (30, 11, 11, 2)).squeeze()
