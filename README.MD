# 基于Sentinel-1的多时序雷达数据的农作物分类(PIE比赛-代码提交和说明)

    使用模型：经典的3D卷积、2D卷积、LSTM2D卷积等的延申

# 01 结构

    -- core
        |
        -- data_preprocess.py
        -- predict.py
        -- train.py
    -- utils
        |
        -- DataManager.py
        -- model.py
        -- utils.py
    -- ref
    -- data
    -- notebook
    -- remote_sensing_cnn_class
    config.yml
    README.MD

## 1.1 说明

### 1.1.1 执行core代码以完成农作物的分类

data_preprocess.py用于数据的预处理；

train.py用于模型的训练和评估;

predict.py用于农作物的预测和输出；

### 1.1.2 utils用于存储函数和类

DataManager.py主要用于数据的批量读取和管理；

model.py用于模型的建立和评估；

utils.py用于存储常规函数或工具；

### 1.1.3 其他

ref为参考文件；

data为存储临时文件的位置；

notebook为存储探索数据的ipynb文件；

config.yml为参数文件；

README.MD为项目说明文档；