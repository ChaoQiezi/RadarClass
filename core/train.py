# @Author   : ChaoQiezi
# @Time     : 2023/8/4  14:15
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import yaml
from utils.DataManager import GenerateData
from utils.model import cnn3d

# load the config file
config = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)

# generate the train data and test dataset by batch
training_generator = GenerateData(config, True)

# build the model
cnn = cnn3d()

# train model on dataset
history = cnn.fit_generator(generator=training_generator, use_multiprocessing=True, workers=12)
# print and view the history
print(history.history)
