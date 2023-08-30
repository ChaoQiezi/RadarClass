# @Author   : ChaoQiezi
# @Time     : 07/08/2023  17:37
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import os
import h5py
import yaml
import tensorflow as tf
from keras.models import load_model
import numpy as np
from utils.DataManager import PredictGenerator

physical_gpus = tf.config.list_physical_devices('GPU')  # 获取GPU列表
tf.config.experimental.set_memory_growth(physical_gpus[0], True)  # 设置GPU显存用量按需使用
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # 这是为了防止GPU显存碎片化
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True  # 程序按需申请内存
# session = tf.compat.v1.Session(config=config)

# load the config file
config = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)

# load the model
cnn = load_model(config['model_path'])

# load predict data, default window size is 11 * 11
with h5py.File(config['original_features'], 'r') as f:
    labels_predict = np.zeros((f['features'].shape[1] - config['chip_size'] + 1,
                               f['features'].shape[2] - config['chip_size'] + 1), dtype=np.int8)

generator = PredictGenerator(config['original_features'])
import time

if __name__ == '__main__':
    t1 = time.time()
    # predictions = cnn(generator, steps=10243, verbose=1, workers=3, use_multiprocessing=True, training=False)
    predictions = cnn.predict(generator, steps=20, verbose=0, workers=3).argmax(axis=1)
    print('time: ', time.time() - t1)
