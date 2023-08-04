# @Author   : ChaoQiezi
# @Time     : 2023/8/3  15:44
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to manage the Data
"""

import os
import glob
import numpy as np
from pyrsgis.ml import imageChipsFromArray
from sklearn.model_selection import train_test_split

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
