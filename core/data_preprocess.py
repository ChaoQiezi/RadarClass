# @Author   : ChaoQiezi
# @Time     : 2023/8/3  16:17
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import os
import yaml
import numpy as np

from utils.DataManager import DataManager
from utils.utils import make_chips

# load the config file
config = yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)

# load and disposal the features and labels dataset
data_manager = DataManager()
radar_vvs = data_manager.load_data(config['radar_vv_dir'])  # shape: (time_steps, rows, cols)
radar_vhs = data_manager.load_data(config['radar_vh_dir'])
labels, _ = data_manager.read_img(config['labels_path'])  # shape: (rows, cols)

# nomalize and make chips
features = np.stack((radar_vvs, radar_vhs), axis=-1)  # shape: (time_steps, rows, cols, bands=2)
features = data_manager.nomalize(features, train_flag=True)
features = make_chips(features, (data_manager.time_steps, config['chip_size'], config['chip_size'], 2),
                      config['stride'])
"""shape: (n_samples, time_steps, chip_size, chip_size, bands=2)"""
labels = np.reshape(labels, -1)  # shape: (rows*cols, )
# discard the invalid samples based on the labels
valid_index = ~np.isnan(labels)
features = features[valid_index]
labels = labels[valid_index]
# split the dataset
features_train, features_test, labels_train, labels_test = data_manager.split_data(features, labels)
# save the dataset
features_labels = {
    'features_train': features_train,
    'features_test': features_test,
    'labels_train': labels_train,
    'labels_test': labels_test
}
save_path = os.path.join(config['save_dir'], 'features_labels.npy')
np.save(save_path, features_labels)
