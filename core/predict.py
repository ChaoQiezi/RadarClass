# @Author   : ChaoQiezi
# @Time     : 07/08/2023  17:37
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import h5py
import yaml
import numpy as np
from utils.model import _row
import time
from skimage.util import view_as_windows
import multiprocessing
from multiprocessing import Manager

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True  # 程序按需申请内存
# session = tf.compat.v1.Session(config=config)

# physical_gpus = tf.config.list_physical_devices('GPU')  # 获取GPU列表
# tf.config.experimental.set_memory_growth(physical_gpus[0], True)  # 设置GPU显存用量按需使用
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # 这是为了防止GPU显存碎片化

# load the config file
config = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)

# # load the model
# cnn = load_model(config['model_path'])

# load predict data, default window size is 11 * 11
with h5py.File(config['original_features'], 'r') as f:
    predict_data = f['features']
    labels_predict = np.zeros((predict_data.shape[1] - config['chip_size'] + 1,
                               predict_data.shape[2] - config['chip_size'] + 1), dtype=np.int8)

original_features = h5py.File(config['original_features'], 'r')['features']
if __name__ == '__main__':
    num_processes = 1
    t1 = time.time()
    manager = Manager()
    return_dict = manager.dict()
    pool = multiprocessing.Pool(processes=num_processes)
    # change threading
    for row in range(5, original_features.shape[1] - 5):
        window_row = view_as_windows((original_features[:, row - 5:row + 6, :, :]), (30, 11, 11, 2)).squeeze()
        pool.apply_async(_row, args=(row, window_row, return_dict))
        if row % 36 == 0:
            pool.close()
            pool.join()
            pool = multiprocessing.Pool(processes=num_processes)
            print('time: {}; row: {}'.format(time.time() - t1, row))
    else:
        pool.close()
        pool.join()
        print('time: ', time.time() - t1)
