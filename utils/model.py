# @Author   : ChaoQiezi
# @Time     : 2023/8/3  20:31
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to build the model
"""

import keras
from keras.layers import Conv2D, Dropout, Flatten, Dense, BatchNormalization


def build_cnn(input_shape, class_num, dropout_rate=0.25, kernel_size=3, activation='relu'):
    """
    this function us used to build the cnn model
    :param input_shape: the shape of the input features(single image chips)
    :param class_num: the number of the classes
    :param dropout_rate: the dropout rate of the neurons(every hidden convolutional layer is both done)
    :param kernel_size: the kernel size of every convolutional layer
    :param activation: the activation function of every hidden layer
    :return: the cnn model
    """

    # """
    # this model source from the 《基于卷积神经网络的遥感图像分类算法研究_李亚飞》
    model = keras.Sequential()  # create a sequential model
    model.add(Conv2D(32, strides=1, kernel_size=kernel_size, padding='valid', activation=activation,
                     input_shape=input_shape))  # add the convolutional layers of the model
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))  # default dropout 25% of the nodes

    model.add(Conv2D(64, strides=1, kernel_size=kernel_size, padding='valid', activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(64, strides=1, kernel_size=1, padding='valid', activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Flatten())  # flatten the input
    model.add(Dense(256, activation=activation))  # add the fully connected layers of the model
    model.add(Dense(128, activation=activation))
    model.add(Dense(class_num, activation='softmax'))  # add the output layer of the model
    # """

    """
    # this model source from the ConvPool-CNN-C, https://cloud.tencent.com/developer/article/1120167
    # this model source from the Striving for Simplicity: The All Convolutional Net：arXiv:1412.6806v3
    model = keras.Sequential()  # create a sequential model
    model.add(Conv2D(96, kernel_size=3, strides=1, padding='same', activation=activation,
                     input_shape=input_shape))  # add the convolutional layers of the model
    model.add(Conv2D(96, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
    model.add(Conv2D(192, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(Conv2D(192, kernel_size=3, strides=1, padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=3, strides=1, padding='same'))
    model.add(Conv2D(192, kernel_size=3, strides=3, padding='valid', activation=activation))
    model.add(Conv2D(192, kernel_size=1, strides=1, padding='valid', activation=activation))
    model.add(Conv2D(96, kernel_size=1, strides=1, padding='valid', activation=activation))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(class_num, activation='softmax'))  # add the output layer of the model
    """

    return model


def cnn3d():
    pass
