# @Author   : ChaoQiezi
# @Time     : 2023/8/4  14:15
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import os
import yaml
from utils.DataManager import GenerateData
from utils.model import cnn3d
from utils.utils import update_config

def main():
    # load the config file
    config = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)

    # generate the train data and test dataset by batch
    training_generator = GenerateData(config, True)

    # build the model
    cnn = cnn3d()
    print(cnn.summary())

    # train model on dataset
    history = cnn.fit(x=training_generator, use_multiprocessing=True, workers=12)
    # print and view the history
    print(cnn.history.history)

    # save the model
    config['model_path'] = os.path.join(config['save_dir'], 'cnn.h5')
    cnn.save(config['model_path'])
    # 保存权重
    config['model_weight_path'] = os.path.join(config['save_dir'], 'cnn_weight.h5')
    cnn.save_weights(config['model_weight_path'])

    # save the config
    update_config(config)


if __name__ == '__main__':
    main()
