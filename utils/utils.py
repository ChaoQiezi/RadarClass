# @Author   : ChaoQiezi
# @Time     : 2023/8/3  16:44
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to save the helpful tools
"""

from osgeo import gdal

def read_img(img_path):


    img = gdal.Open(img_path)
    img_cols = img.RasterXSize  # the cols of the image
    img_rows = img.RasterYSize  # the rows of the image
    img_nodata = img.GetRasterBand(1).GetNoDataValue()  # the nodata value of the image

    # get the transform info of the image
    img_transform = img.GetGeoTransform()
    img_projection = img.GetProjection()

    # get the bands of the image
    img_data = img.ReadAsArray(0, 0, img_width, img_height)

    del img

    return img_data, [img_rows, img_cols, img_nodata, img_transform, img_projection]
