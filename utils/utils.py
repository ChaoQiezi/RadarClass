# @Author   : ChaoQiezi
# @Time     : 2023/8/3  16:44
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to save the helpful tools
"""

import numpy as np
import yaml
from skimage.util import view_as_windows
from osgeo import gdal
from keras.layers import Reshape


def read_img(img_path):
    img = gdal.Open(img_path)
    if img is None:
        raise ValueError("The image {} is not exist".format(img_path))

    img_cols = img.RasterXSize  # the cols of the image
    img_rows = img.RasterYSize  # the rows of the image
    img_nodata = img.GetRasterBand(1).GetNoDataValue()  # the nodata value of the image

    # get the transform info of the image
    img_transform = img.GetGeoTransform()
    img_projection = img.GetProjection()

    # get the bands of the image
    img_data = img.ReadAsArray(0, 0, img_cols, img_rows)

    del img

    return img_data, [img_rows, img_cols, img_nodata, img_transform, img_projection]


def make_chips(dataset, window_shape, stride=1):
    padding_size = window_shape[1] // 2
    features = np.pad(dataset, ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0)), 'reflect')
    features = view_as_windows(features, window_shape, stride)

    features = np.squeeze(features)  # shape: (rows, cols, time_steps, chip_size, chip_size, bands=2)

    return features


def update_config(config):
    with open('config.yml', 'w') as f:
        yaml.safe_dump(config, f)
