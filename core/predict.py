# @Author   : ChaoQiezi
# @Time     : 07/08/2023  17:37
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import os
import h5py
import matplotlib.pyplot as plt
import yaml
import numpy as np
from utils.utils import predict_row
import time
from skimage.util import view_as_windows
import multiprocessing
from multiprocessing import Manager
from utils.utils import update_config, write_tiff

# load the config file
config = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)

# load predict data, default window size is 11 * 11
with h5py.File(config['original_features'], 'r') as f:
    predict_data = f['features']
    labels_predict = np.zeros((predict_data.shape[1] - config['chip_size'] + 1,
                               predict_data.shape[2] - config['chip_size'] + 1), dtype=np.int8)


if __name__ == '__main__':
    t1 = time.time()
    manager = Manager()
    return_dict = manager.dict()
    original_features = h5py.File(config['original_features'], 'r')['features']
    pool = multiprocessing.Pool(processes=1)
    # change threading
    for row in range(5, original_features.shape[1] - 5):
        window_row = view_as_windows((original_features[:, row - 5:row + 6, :, :]), (30, 11, 11, 2)).squeeze()
        pool.apply_async(predict_row, args=(row, window_row, return_dict))
        if row % 20 == 0:
            pool.close()
            pool.join()
            pool = multiprocessing.Pool(processes=1)
            print('time: {}; row: {}'.format(time.time() - t1, row))
    else:
        pool.close()
        pool.join()
        print('time: ', time.time() - t1)

    for row, result in return_dict.items():
        labels_predict[row, :] = result

    plt.imshow(labels_predict)
    plt.show()
    # config['labels_predict_path'] = os.path.join(config['save_dir'], 'labels_predict_compound_with_focal.tif')
    config['labels_predict_path'] = os.path.join(config['save_dir'], 'labels_predict_lstm2d.tif')
    # config['labels_predict_path'] = os.path.join(config['save_dir'], 'labels_predict_cnn3d.tif')
    # config['labels_predict_path'] = os.path.join(config['save_dir'], 'labels_predict_compound.tif')
    write_tiff(labels_predict, config)

    update_config(config)
