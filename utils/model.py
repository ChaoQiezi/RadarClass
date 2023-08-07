# @Author   : ChaoQiezi
# @Time     : 2023/8/3  20:31
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to build the model
"""
import h5py
import keras
from keras.layers import Conv2D, Dropout, Flatten, Dense, BatchNormalization, Conv3D, Activation, ConvLSTM2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt


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
    model = keras.Sequential()

    # 32@3*3*3
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(30, 11, 11, 2)))
    """input_shape: (batch_size, time_steps, rows, cols, channels)"""
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))  # default dropout 25% of the nodes(or convolution kernel) randomly

    # 64@3*3*3
    model.add(Conv3D(64, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # 128@3*3*3
    model.add(Conv3D(128, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # 256@3*3
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Flatten
    model.add(Flatten())

    # Dense 256
    model.add(Dense(256, activation='relu'))
    # Dense 128
    model.add(Dense(128, activation='relu'))
    # Class  ==> n_class=10
    model.add(Dense(2, activation='softmax'))

    # compile model
    adam = Adam(learning_rate=0.0005)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def lstm2d():
    pass


def build_cnn_lstm(input_shape, class_num, dropout_rate=0.25, activation='relu'):
    """
    this function us used to build the lstm model
    :param input_shape: the shape of the input features(single image chips) ==> (time_step, height, width, channel)
    :param class_num: the number of the classes
    :param dropout_rate: the dropout rate of the neurons(every hidden convolutional layer is both done)
    :param activation: the activation function of every hidden layer
    :return: the lstm model
    """

    model = keras.Sequential()  # create a sequential model
    model.add(ConvLSTM2D(64, strides=1, kernel_size=3, padding='valid', activation=activation, return_sequences=True,
                         input_shape=input_shape))  # add the convolutional layers of the model,Out=(None, 12, 3, 3, 32)
    model.add(Dropout(dropout_rate))  # default dropout 25% of the nodes
    model.add(ConvLSTM2D(128, strides=1, kernel_size=3, padding='valid', activation=activation))  # Out=(None, 1, 1, 64)
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(128, strides=1, kernel_size=1, padding='valid', activation=activation))  # Out=(None, 1, 1, 64)
    model.add(Dropout(dropout_rate))
    model.add(Flatten())  # flatten the input
    model.add(Dense(256, activation=activation))  # add the fully connected layers of the model
    model.add(Dense(64, activation=activation))
    model.add(Dense(class_num, activation='softmax'))  # add the output layer of the model

    return model


def lstm2d():
    model = keras.Sequential()
    """
    My shape of the single sample is (30, 11, 11, 2), and 30 means the time_steps, first 11 means the rows, second 11 
    means cols, 2 means the channels.So the input_shape of the ConvLSTM2D is """
    # 32@3*3*3
    model.add(ConvLSTM2D(32, kernel_size=(3, 3), input_shape=(30, 11, 11, 2)), return_sequences=True)
    """input_shape: (batch_size, time_steps, rows, cols, channels)"""
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))  # default dropout 25% of the nodes(or convolution kernel) randomly

    # 64@3*3*3
    model.add(ConvLSTM2D(64, kernel_size=(3, 3)), return_sequences=True)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # 128@3*3*3
    model.add(ConvLSTM2D(128, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # 256@3*3
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Flatten
    model.add(Flatten())

    # Dense 256
    model.add(Dense(256, activation='relu'))
    # Dense 128
    model.add(Dense(128, activation='relu'))
    # Class  ==> n_class=10
    model.add(Dense(2, activation='softmax'))

    # compile model
    adam = Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def build_cnn_lstm(input_shape, class_num, dropout_rate=0.25, activation='relu'):
    """
    this function us used to build the lstm model
    :param input_shape: the shape of the input features(single image chips) ==> (time_step, height, width, channel)
    :param class_num: the number of the classes
    :param dropout_rate: the dropout rate of the neurons(every hidden convolutional layer is both done)
    :param activation: the activation function of every hidden layer
    :return: the lstm model
    """

    model = keras.Sequential()  # create a sequential model
    model.add(ConvLSTM2D(64, strides=1, kernel_size=3, padding='valid', activation=activation, return_sequences=True,
                         input_shape=input_shape))  # add the convolutional layers of the model,Out=(None, 12, 3, 3, 32)
    model.add(Dropout(dropout_rate))  # default dropout 25% of the nodes
    model.add(ConvLSTM2D(128, strides=1, kernel_size=3, padding='valid', activation=activation))  # Out=(None, 1, 1, 64)
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(128, strides=1, kernel_size=1, padding='valid', activation=activation))  # Out=(None, 1, 1, 64)
    model.add(Dropout(dropout_rate))
    model.add(Flatten())  # flatten the input
    model.add(Dense(256, activation=activation))  # add the fully connected layers of the model
    model.add(Dense(64, activation=activation))
    model.add(Dense(class_num, activation='softmax'))  # add the output layer of the model

    return model


import numpy as np
from keras.models import load_model
import tensorflow as tf
import os


def _row(row, window_row, return_dict):
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
    physical_gpus = tf.config.list_physical_devices('GPU')  # 获取GPU列表
    tf.config.experimental.set_memory_growth(physical_gpus[0], True)  # 设置GPU显存用量按需使用
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # 这是为了防止GPU显存碎片化

    # load the model
    cnn = cnn3d()
    cnn.load_weights(r'E:\Material\Object\RadarSentinelClass\Data\cnn_weight.h5')
    # cnn = load_model('E:\\Material\\Object\\RadarSentinelClass\\Data\\cnn.h5')
    result = cnn.predict(window_row, verbose=1, batch_size=1024).argmax(axis=1)

    return_dict[row - 5] = result
    # tf.keras.backend.clear_session()  # 表示清除模型的状态，否则会影响下一次的预测
    print('row: ', row)

    # cnn.reset_states()  # 表示清除模型的状态，否则会影响下一次的预测
    # tf.keras.backend.clear_session()  # 表示清除模型的状态，否则会影响下一次的预测


def accuracy_loss_curve(history):
    """
    this function is used to plot the accuracy and loss curveof the model
    :param history: the history of the model
    :return: None
    """

    plt.figure(figsize=(10, 5))  # set the size of the figure
    plt.subplot(1, 2, 1)  # set the position of the first curve
    plt.title("准确率曲线 (Accuracy Curve)")  # set the title of the first curve
    plt.xlabel("迭代次数 (epoch)")  # set the xlabel of the first curve
    plt.ylabel("准确率 (accuracy)")  # set the ylabel of the first curve
    plt.plot(history.history['sparse_categorical_accuracy'], color='blue', linestyle='-', label='训练准确率')
    plt.plot(history.history['val_sparse_categorical_accuracy'], color='red', linestyle='--', label='验证准确率')
    plt.legend(loc='upper left')  # set the legend of the first curve
    plt.grid(True)

    plt.subplot(1, 2, 2)  # set the position of the second curve
    plt.title("损失曲线 (Loss Curve)")  # set the title of the second curve
    plt.xlabel("迭代次数 (epoch)")  # set the xlabel of the second curve
    plt.ylabel("损失 (loss)")  # set the ylabel of the second curve
    plt.plot(history.history['loss'], color='blue', linestyle='-', label='训练损失')
    plt.plot(history.history['val_loss'], color='red', linestyle='--', label='验证损失')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.show()  # show the figure
