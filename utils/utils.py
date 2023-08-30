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
import tensorflow as tf
import os
from utils.model import cnn3d, lstm2d, compound_conv


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


def predict_row(row, window_row, return_dict):
    physical_gpus = tf.config.list_physical_devices('GPU')  # 获取GPU列表
    tf.config.experimental.set_memory_growth(physical_gpus[0], True)  # 设置GPU显存用量按需使用
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # 这是为了防止GPU显存碎片化

    # load the model
    model = lstm2d((30, 11, 11, 2), 6)
    model.load_weights(r'E:\Material\Object\RadarSentinelClass\Data\lstm2d_weight.h5')
    # model = cnn3d((30, 11, 11, 2), 6)
    # model.load_weights(r'E:\Material\Object\RadarSentinelClass\Data\cnn3d_weight.h5')
    # model = lstm2d((30, 11, 11, 2), 6)
    # model.load_weights(r'E:\Material\Object\RadarSentinelClass\Data\lstm2d_weight.h5')
    # model = compound_conv((30, 11, 11, 2), 6)
    # model.load_weights(r'E:\Material\Object\RadarSentinelClass\Data\compound_weight.h5')
    result = model.predict(window_row, verbose=1, batch_size=1024).argmax(axis=1)

    return_dict[row - 5] = result
    print('row: ', row)


def write_tiff(data, config):
    """
    create a tiff file
    :param data:
    :return:
    """
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(config['labels_predict_path'], data.shape[1], data.shape[0], 1, gdal.GDT_Int16)
    geo_info = config['geo_info']  # [img_rows, img_cols, img_nodata, img_transform, img_projection]
    ds.SetGeoTransform(geo_info[3])
    ds.SetProjection(geo_info[4])

    ds.GetRasterBand(1).WriteArray(data)
    ds.FlushCache()

    del ds


def update_config(config):
    with open('config.yml', 'w') as f:
        yaml.dump(config, f)
