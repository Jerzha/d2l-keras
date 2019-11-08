import collections
import math
import os
import random
import sys
import tarfile
import time
import zipfile

from IPython import display
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


def show_images(imgs, num_rows, num_cols, scale=2):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape(28, 28))
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)


def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)


def metric_accuracy(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
    acc = K.equal(ytrue, ypred)
    return K.mean(acc)


def load_data_jay_lyrics():
    """Load the Jay Chou lyric data set (available in the Chinese book)."""
    with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, corpus_chars, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减1是因为输出的索引是相应输入的索引加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield np.array(X, ctx), np.array(Y, ctx)


def data_iter_consecutive(corpus_indices, batch_size, num_steps):
    corpus_indices = np.array(corpus_indices)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


def predict_rnn_gluon(prefix, num_chars, model, vocab_size, idx_to_char, char_to_idx, num_steps):
    output = np.array([char_to_idx[prefix[idx]] for idx in range(len(prefix))])
    for t in range(num_chars + len(prefix) - 1):
        # print(output)
        X = keras.utils.to_categorical(output[-num_steps : ], vocab_size)
        # print('X', X.shape, output[-num_steps : ])
        Y = model.predict(X.reshape(1, num_steps, vocab_size))  # 引入batch=1维度
        if t < len(prefix) - 1:
            # output = np.append(output, char_to_idx[prefix[t + 1]])
            pass
        else:
            output = np.append(output, int(Y.argmax(axis=-1)))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_gluon(model, vocab_size, corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    model.compile(
        optimizer='adam',  # keras.optimizers.SGD(learning_rate=lr, momentum=0, decay=0, clipvalue=1),
        loss=keras.losses.categorical_crossentropy)

    for epoch in range(num_epochs):
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps)
        for X, Y in data_iter:
            x = keras.utils.to_categorical(X, vocab_size)
            y = keras.utils.to_categorical(Y[:, -1], vocab_size)
            # print(x.shape, y.shape)
            model.train_on_batch(x.reshape(batch_size, num_steps, vocab_size), y)

        # print(epoch, pred_period)
        if (epoch + 1) % pred_period == 0:
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(
                    prefix, pred_len, model, vocab_size, idx_to_char,
                    char_to_idx, num_steps))


def train_2d(trainer):  # 本函数将保存在d2lzh包中方便以后使用
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1和s2是自变量状态，本章后续几节会使用
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results


def show_trace_2d(f, results):  # 本函数将保存在d2lzh包中方便以后使用
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')


def get_data_ch7():  # 本函数已保存在d2lzh包中方便以后使用
    data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return np.array(data[:1500, :-1]), np.array(data[:1500, -1])


class Residual(keras.layers.Layer):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, *args, **kwargs):
        super(Residual, self).__init__(*args, **kwargs)
        self.conv1 = keras.layers.Conv2D(num_channels, kernel_size=3, padding='same', strides=strides)
        self.conv2 = keras.layers.Conv2D(num_channels, kernel_size=3, padding='same')

        if use_1x1conv:
            self.conv3 = keras.layers.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None

        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        Y = keras.layers.ReLU()(self.bn1(self.conv1(inputs)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            inputs = self.conv3(inputs)
        return keras.layers.ReLU()(Y + inputs)


def resnet18(num_class):
    def resnet_block(model, num_channels, num_residuals, first_block=False):
        for i in range(num_residuals):
            if i == 0 and not first_block:
                model.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                model.add(Residual(num_channels))
        return model

    model = keras.Sequential()
    #model.add(keras.layers.Lambda(lambda img: tf.image.resize(img, (224, 224)), input_shape=(32, 32, 3)))
    model.add(keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')) # input_shape=(224, 224, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

    resnet_block(model, 64, 2, first_block=True),
    resnet_block(model, 128, 2),
    resnet_block(model, 256, 2),
    resnet_block(model, 512, 2)

    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(num_class))
    model.add(keras.layers.Softmax())
    return model


def bbox_to_rect(bbox, color):
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox, color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


# return:  (b, an, 4) 4-> (x1/w, y1/h, x2/w, y2/h)
def MultiBoxPrior(X, sizes, ratios, do_reduce=True):
    Y = []

    def foreach_radio_sizes(x, y, h, w, p):
        flag_first_size = True
        for ratio in ratios:
            flag_pass = False
            for size in sizes:
                if flag_pass:
                    continue
                if not flag_first_size and do_reduce:
                    flag_pass = True
                tw = w * size * np.sqrt(ratio)
                th = h * size / np.sqrt(ratio)

                x1 = (x - tw / 2) / w
                y1 = (y - th / 2) / h
                x2 = (x + tw / 2) / w
                y2 = (y + th / 2) / h
                p.append((x1, y1, x2, y2))
            flag_first_size = False
        return p

    # for b in X:
    batch = []
    h, w = X.shape[1:3]
    for y in range(h):
        for x in range(w):
            foreach_radio_sizes(x+0.5, y+0.5, h, w, batch)
    Y.append(batch)

    return np.array(Y)