# @Author   : ChaoQiezi
# @Time     : 2023/8/4  14:15
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import os
import h5py
import yaml
from utils.DataManager import GenerateData
from utils.model import cnn3d, lstm2d, compound_conv
from utils.utils import update_config
from keras.callbacks import History
from matplotlib import pyplot as plt
import keras
import pandas as pd

if __name__ == '__main__':  # Does main do anything else?
    # load the config file
    config = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)

    # generate the train data and test dataset by batch
    training_generator = GenerateData(config, True, batch_size=120)
    testing_generator = GenerateData(config, False, batch_size=120)
    with h5py.File(config['features_labels_path'], 'r') as f:
        val_features = f['features_test'][:2000:, :, :]
        val_labels = f['labels_test'][:2000]
        # one-hot encoding
        val_labels = keras.utils.to_categorical(val_labels, num_classes=config['n_class'])

    # build the model
    cnn = lstm2d((30, 11, 11, 2), 6)  # input_shape, n_class
    print(cnn.summary())
    # train model on dataset
    history = cnn.fit(x=training_generator, use_multiprocessing=True, workers=10, epochs=30,
                      class_weight=config['class_weight'], validation_data=(val_features, val_labels),
                      callbacks=[History()])

    # plt
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['precision'])
    plt.plot(history.history['recall'])
    plt.plot(history.history['loss'])
    plt.show()
    # plt val
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['val_precision'])
    plt.plot(history.history['val_recall'])
    plt.plot(history.history['val_loss'])
    plt.show()

    # save the model
    config['model_path'] = os.path.join(config['save_dir'], 'compound_with_focal.h5')
    cnn.save(config['model_path'])
    # save model weight
    config['model_weight_path'] = os.path.join(config['save_dir'], 'compound_weight_with_focal.h5')
    cnn.save_weights(config['model_weight_path'])
    """
    cnn_compound: 复合卷积
    cnn3d: 纯3D卷积
    lstm2d: LSTM2D卷积
    """

    # save history as excel
    config['history_path'] = os.path.join(config['save_dir'], 'history_compound_with_focal.xlsx')
    history_df = pd.DataFrame(history.history)
    history_df.to_excel(config['history_path'])

    # evaluate the model
    cnn.evaluate(testing_generator)

    # update the config
    update_config(config)
