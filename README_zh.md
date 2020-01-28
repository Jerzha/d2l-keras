# 《动手学深度学习》 Keras 实现

* 原项目地址： https://github.com/d2l-ai/d2l-zh
* 英文版地址： https://github.com/d2l-ai/d2l-en

## 说明:

1. Keras不支持动态图，所以有些部分代码是使用numpy或者tf2.0来写的
2. 我没有拷贝整本书的内容过来，该项目仅有中文标题和keras实现的代码，可能还有一些笔记
3. 由于中途TF2.0发布了，所以后续可能代码会转向使用tf.keras，毕竟他针对运行速度的优化实在太棒了。
4. 为了方便Jupyter显示训练动态，我使用了livelossplot库


## 目录:

#### 第二章 预备知识
* [2.2 数据操作](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter2/2.2_data_manipulation.ipynb)
* [2.3 自动求梯度](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter2/2.3_automatic_differentiation.ipynb)
#### 第三章 深度学习基础
* [3.2 线性回归](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter3/3.2_linear-regression.ipynb)
* [3.5 图像分类数据集(Fashion Mnist)](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter3/3.5_fashion-mnist.ipynb)
* [3.6 Softmax回归](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter3/3.6_softmax-regression.ipynb)
* [3.9 多层感知器](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter3/3.9_multilayer-perceptrons.ipynb)
* [3.11 模型选择、过拟合和欠拟合](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter3/3.11_underfitting_overfitting.ipynb)
* [3.12 权重衰减](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter3/3.12_weight_decay.ipynb)
* [3.13 丢弃法](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter3/3.13_dropout.ipynb)
* [3.16 实战Kaggle竞赛：房价预测](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter3/3.16_kaggle_house_prices.ipynb)
#### 第四章 深度学习计算
* [4.1 模型构造](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter4/4.1_layers_and_blocks.ipynb)
* [4.2 模型参数访问、初始化和共享](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter4/4.2_parameter_management.ipynb)
* [4.3 模型延后初始化](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter4/4.3_deferred_initialization.ipynb)
* [4.4 自定义层](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter4/4.4_custom_layers.ipynb)
* [4.5 读取和存储](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter4/4.5_file_io.ipynb)
* [4.6 GPU计算](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter4/4.6_gpus.ipynb)
#### 第五章 卷积神经网络
* [5.5 Lenet](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter5/5.5_lenet.ipynb)	
* [5.6 Alexnet](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter5/5.6_alexnet.ipynb)	
* [5.7 VGG](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter5/5.7_vgg.ipynb)	
* [5.8 Network In Network](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter5/5.8_network_in_network.ipynb)
* [5.9 GoogleNet](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter5/5.9_googlenet.ipynb)
* [5.11 Resnet](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter5/5.11_resnet.ipynb)	
* [5.12 Densenet](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter5/5.12_densenet.ipynb)
#### 第六章 循环神经网络
* [6.3 语言模型数据集](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter6/6.3_language_model_dataset.ipynb)
* [6.4 循环神经网络 RNN](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter6/6.4_rnn.ipynb)
* [6.7 门控循环单元 GRU](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter6/6.7_gru.ipynb)
* [6.8 长短期记忆网络 LSTM](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter6/6.8_lstm.ipynb)
#### 第七章 优化算法
* [7.1 优化与深度学习](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter7/7.1_optimization_and_deeplearning.ipynb)	
* [7.2 梯度下降和随机梯度下降](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter7/7.2_gradient_descent.ipynb)
* [7.4 动量法](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter7/7.4_momentum.ipynb)		
* [7.5 Adagrad算法](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter7/7.5_adagrad.ipynb)		
* [7.6 Rmsprop算法](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter7/7.6_rmsprop.ipynb)
#### 第九章 计算机视觉
* [9.1 图像增广](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter9/9.1_image_augmentation.ipynb)
* [9.2 微调](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter9/9.2_fine_tuning.ipynb)
* [9.3 目标检测和边界框](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter9/9.3_bounding_box.ipynb)
* [9.4 锚框](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter9/9.4_anchor_box.ipynb)
* [9.5 多尺度目标检测](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter9/9.5_multiscale_object_detection.ipynb)
* [9.6 目标检测数据集](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter9/9.6_object_detection_data_set.ipynb)
* [9.7 单发多框检测 SSD](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter9/9.7_ssd.ipynb)
* [9.8 区域卷积神经网络 RCNN系列](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter9/9.8_rcnn.ipynb)
* [9.9 语意分割与数据集](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter9/9.9_semantic_segmentation.ipynb)
* [9.10 全卷积网络 FCN](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter9/9.10_fully_convolutional_network.ipynb)
* [9.11 样式迁移](https://nbviewer.jupyter.org/github/Jerzha/d2l-keras/blob/master/chapter9/9.11_neural_style_transfer.ipynb)
* 更新中...

## 英文版引用
BibTeX entry:

```
@book{zhang2019dive,
    title={Dive into Deep Learning},
    author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
    note={\url{http://www.d2l.ai}},
    year={2019}
}
```
