# @Author   : ChaoQiezi
# @Time     : 2023/8/3  16:17
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import yaml
from utils.DataManager import DataManager

# load the config file
config = yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)


# load and disposal the features and labels dataset
data_manager = DataManager()
radar_vvs = data_manager.load_data(config['radar_vv_dir'])
radar_vhs = data_manager.load_data(config['radar_vh_dir'])



