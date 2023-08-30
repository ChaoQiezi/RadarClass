# @Author   : ChaoQiezi
# @Time     : 2023/8/3  20:31
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to build the model
"""
import h5py
import keras
from keras.layers import Conv2D, Dropout, Flatten, Dense, BatchNormalization, Conv3D, Activation, ConvLSTM2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.metrics import categorical_accuracy, Precision, Recall
from keras import backend as K


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


def cnn3d(input_shape, n_class):
    model = keras.Sequential()

    # 32@3*3*3
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=input_shape, activation='relu'))
    """input_shape: (batch_size, time_steps, rows, cols, channels)"""
    model.add(Dropout(0.25))  # default dropout 25% of the nodes(or convolution kernel) randomly

    # 64@3*3*3
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(Dropout(0.25))

    # 128@3*3*3
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(Dropout(0.25))

    # 256@3*3*3
    model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu'))
    model.add(Dropout(0.25))

    # Flatten
    model.add(Flatten())

    # Dense 256
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_class, activation='softmax'))

    # compile model
    adam = Adam(learning_rate=0.0005)
    model.compile(loss=categorical_crossentropy, optimizer=adam, metrics=[categorical_accuracy, Precision(), Recall()])

    return model


import tensorflow as tf


# 焦点损失函数计算
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        loss_1 = -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())
        loss_0 = -(1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon())

        sample_loss = K.mean(K.sum(loss_1 + loss_0, axis=1))

        return sample_loss

    return focal_loss_fixed


def compound_conv(input_shape, n_class):
    model = keras.Sequential()

    # 32@3*3*3
    model.add(ConvLSTM2D(32, kernel_size=(3, 3), input_shape=input_shape, return_sequences=True))
    """input_shape: (batch_size, time_steps, rows, cols, channels)"""
    model.add(Dropout(0.25))

    # 64@3*3*3
    model.add(ConvLSTM2D(64, kernel_size=(3, 3), return_sequences=True))
    model.add(Dropout(0.25))

    # 128@3*3*3
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(Dropout(0.25))  # default dropout 25% of the nodes(or convolution kernel) randomly

    # 256@3*3
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(Dropout(0.25))

    # Flatten
    model.add(Flatten())

    # Dense 256
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_class, activation='softmax'))

    # compile model
    adam = Adam(learning_rate=0.0005)
    model.compile(loss=focal_loss(), optimizer=adam, metrics=[categorical_accuracy, Precision(), Recall()])

    return model


def lstm2d(input_shape, n_class):
    model = keras.Sequential()
    """
    My shape of the single sample is (30, 11, 11, 2), and 30 means the time_steps, first 11 means the rows, second 11 
    means cols, 2 means the channels.So the input_shape of the ConvLSTM2D is """
    # 32@3*3
    model.add(ConvLSTM2D(32, kernel_size=(3, 3), input_shape=input_shape, return_sequences=True))
    """input_shape: (batch_size, time_steps, rows, cols, channels)"""
    model.add(Dropout(0.25))  # default dropout 25% of the nodes(or convolution kernel) randomly

    # 64@3*3
    model.add(ConvLSTM2D(64, kernel_size=(3, 3), return_sequences=True))
    model.add(Dropout(0.25))

    # 128@3*3
    model.add(ConvLSTM2D(128, kernel_size=(3, 3), return_sequences=True))
    model.add(Dropout(0.25))

    # 256@3*3
    model.add(ConvLSTM2D(256, kernel_size=(3, 3)))
    model.add(Dropout(0.25))

    # Flatten
    model.add(Flatten())

    # Dense 256
    model.add(Dense(256, activation='relu'))
    # Dense 128
    model.add(Dense(128, activation='relu'))
    # Class  ==> n_class=10
    model.add(Dense(n_class, activation='softmax'))

    # compile model
    adam = Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=adam,
                  metrics=[categorical_accuracy, Precision(), Recall()])

    return model


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
