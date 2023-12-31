# @Author   : ChaoQiezi
# @Time     : 2023/8/3  16:17
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import os
import yaml
import numpy as np
import h5py

from utils.DataManager import DataManager
from utils.utils import make_chips, update_config

# load the config file
config = yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)

# load and disposal the features and labels dataset
data_manager = DataManager(config)
radar_vvs = data_manager.load_data(config['radar_vv_dir'])  # shape: (time_steps, rows, cols) = (30, 10243, 11500)
radar_vhs = data_manager.load_data(config['radar_vh_dir'])
labels, labels_info = data_manager.read_img(config['labels_path'])  # shape: (rows, cols)
config['geo_info'] = labels_info  # [img_rows, img_cols, img_nodata, img_transform, img_projection]

# normalize and make chips
features = np.stack((radar_vvs, radar_vhs), axis=-1)  # shape: (time_steps, rows, cols, bands=2)
config['bands'] = 2
del radar_vvs, radar_vhs  # release the memory
features = data_manager.nomalize(features, train_flag=True)
# save the original features for prediction
config['original_features'] = os.path.join(config['save_dir'], 'original_features.h5')
with h5py.File(config['original_features'], 'w') as f:
    padding_size = config['chip_size'] // 2  # default window size is 11 * 11
    f.create_dataset(
        'features', data=np.pad(features, ((0, 0), (padding_size, padding_size), (padding_size, padding_size),
                                           (0, 0)), 'reflect'), dtype=np.float32)
# make chips
features = make_chips(features, (data_manager.time_steps, config['chip_size'], config['chip_size'], config[
    'bands']))  # features shape: (rows, cols, time_steps, chip_size, chip_size, bands=2)
# discard the invalid samples based on the labels
rows_index, cols_index = np.where(labels != labels_info[2])  # labels_info[2] is the nodata value
features = features[rows_index, cols_index, :, :, :, :]
labels = labels[rows_index, cols_index]
# get the weight of classes
config['class_weight'] = data_manager.extract_weight(labels)
config['n_class'] = len(np.unique(labels))
# split the dataset
features_labels = {
    'features_train': None,
    'features_test': None,
    'labels_train': None,
    'labels_test': None
}
features_labels['features_train'], features_labels['features_test'], features_labels['labels_train'], \
    features_labels['labels_test'] = data_manager.split_data(features, labels)
del features, labels
# save the train dataset and test dataset as h5 file
config['features_labels_path'] = os.path.join(config['save_dir'], 'features_labels.h5')
with h5py.File(config['features_labels_path'], 'w') as f:
    for key, value in features_labels.items():
        f.create_dataset(key, data=value)

config['train_samples'] = features_labels['features_train'].shape[0]
config['test_samples'] = features_labels['features_test'].shape[0]
config['time_steps'] = features_labels['features_train'].shape[1]
update_config(config)
