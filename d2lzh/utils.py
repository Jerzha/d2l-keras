import random
import zipfile
import json
import pandas as pd

from IPython import display
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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


# 通过feature map锚框的sizes radios参数，生成锚框
# return:  (b, an, 4) 4-> (x1/w, y1/h, x2/w, y2/h)
def MultiBoxPrior(X, sizes, ratios, do_reduce=True):
    assert (len(X.shape) == 4)
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
    if isinstance(X, np.ndarray):
        h, w = X.shape[1:3]
    else:
        if len(X.shape.as_list()) == 4:
            h, w = X.shape.as_list()[1:3]
        else:
            h = w = 1

    for y in range(h):
        for x in range(w):
            foreach_radio_sizes(x + 0.5, y + 0.5, h, w, batch)
    Y.append(batch)

    return np.array(Y)


def iou(a, b):
    def intersection(ai, bi):
        x = max(ai[0], bi[0])
        y = max(ai[1], bi[1])
        w = min(ai[2], bi[2]) - x
        h = min(ai[3], bi[3]) - y
        if w < 0 or h < 0:
            return 0
        return w * h

    def union(au, bu, area_intersection):
        area_a = (au[2] - au[0]) * (au[3] - au[1])
        area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
        area_union = area_a + area_b - area_intersection
        return area_union

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)
    return float(area_i) / float(area_u + 1e-6)


# 根据每个锚框生成当前batch数据基于锚框的标签和偏移
# anchor: anchor box (1, an, 4)
# label: label (b, gtn, 5)  5 -> (category, x1, y1, x2, y2)
# cls_pred: anchor pred (b, clz+1, an)
# return:
# bbox_offset: anchor offsets (b, an * 4)
# bbox_mask:
# cls_labels
def MultiBoxTarget(anchor, label, thresh=0.5):
    assert (len(anchor.shape) == 3 and anchor.shape[2] == 4)
    assert (len(label[0].shape) == 2 and label[0].shape[1] == 5)

    # a = anchor
    # b = gt
    def calc_offset(idx, offset, a, b):
        ux = uy = uw = uh = 0
        ax = ay = 0.1
        aw = ah = 0.2
        wa = a[2] - a[0]
        ha = a[3] - a[1]
        wb = b[2] - b[0]
        hb = b[3] - b[1]
        xa = a[0] + wa / 2  # centor points
        ya = a[1] + ha / 2  # centor points
        xb = b[0] + wb / 2  # centor points
        yb = b[1] + hb / 2  # centor points
        x1 = ((xb - xa) / wa - ux) / ax
        y1 = ((yb - ya) / ha - uy) / ay
        w1 = (np.log(wb / wa) - uw) / aw
        h1 = (np.log(hb / ha) - uh) / ah
        offset[idx * 4 + 0] = x1
        offset[idx * 4 + 1] = y1
        offset[idx * 4 + 2] = w1
        offset[idx * 4 + 3] = h1

    bbox_offset = []
    bbox_mask = []
    cls_labels = []
    for batch_idx in range(len(label)):
        ious = []  # (label_n, anchor_n)
        offset = [0. for _ in range(len(anchor[0]) * 4)]
        mask = [1. for _ in range(len(anchor[0]) * 4)]
        clabels = [0. for _ in range(len(anchor[0]))]
        for gt in label[batch_idx]:
            ious_label = []
            for anc in anchor[0]:
                ious_label.append(iou(gt[1:5], anc))
            ious.append(ious_label)
        ious = np.array(ious)
        ious2 = np.copy(ious)

        num_posv = 0
        num_negv = 0
        for _ in range(len(label[batch_idx])):
            max_iou_idx = np.argmax(ious)
            max_iou_idx = np.unravel_index(max_iou_idx, ious.shape)
            clabels[max_iou_idx[1]] = max_iou_idx[0] + 1  # bg is 0, others + 1
            ious[max_iou_idx[0], :] = -1
            ious[:, max_iou_idx[1]] = -1
            num_posv += 1

        # second scan
        for anidx in range(len(clabels)):
            if clabels[anidx] != 0:
                calc_offset(anidx, offset, anchor[0, anidx, :], label[batch_idx][clabels[anidx] - 1, 1:5])
                continue
            max_iou_idx = np.argmax(ious2[:, anidx])
            score = ious2[max_iou_idx, anidx]
            if score > thresh:
                clabels[anidx] = max_iou_idx + 1
                calc_offset(anidx, offset, anchor[0, anidx, :], label[batch_idx][clabels[anidx] - 1, 1:5])
                num_posv += 1
            else:
                clabels[anidx] = 0
                num_negv += 1
                mask[anidx * 4 + 0] = 0.
                mask[anidx * 4 + 1] = 0.
                mask[anidx * 4 + 2] = 0.
                mask[anidx * 4 + 3] = 0.

        bbox_offset.append(offset)
        bbox_mask.append(mask)
        cls_labels.append(clabels)
    return np.array(bbox_offset), np.array(bbox_mask), np.array(cls_labels)


# cls_prob : predicted prob of each anchor (b, an, clz+1)
# anchor : anchor box (1, an * 4)
# threshold : nms threshold
# return: (b, new_an, 6)
#  6-> (category, prob, x1, y1, x2, y2)
#  category = -1 : useless
def NMS(cls_prob, bbox, threshold=0.5):
    output = []
    for bn in range(len(cls_prob)):
        # concat
        c = np.argmax(cls_prob[bn], axis=1)
        c1 = np.expand_dims(c, axis=-1)
        p = np.max(cls_prob[bn], axis=1)
        p1 = np.expand_dims(p, axis=-1)

        La = np.concatenate((c1, p1, bbox[bn]), axis=1)

        # delete bg (-1)
        id0 = np.argwhere(c == 0)
        La[id0, 0] = -1

        deleted = La[La[:, 0] == -1, :]
        La = np.delete(La, id0.reshape(-1), axis=0)
        p = np.delete(p, id0.reshape(-1), axis=0)

        # sort
        cls_sort_idx = np.argsort(-p, axis=0)
        Ls = La[cls_sort_idx]

        for topidx in range(len(Ls)):
            if topidx >= len(Ls):
                break
            box_top = Ls[topidx, 2:6]
            for othidx in range(topidx + 1, len(Ls)):
                if othidx >= len(Ls):
                    break

                box_oth = Ls[othidx, 2:6]
                IoU = iou(box_top, box_oth)
                if IoU > threshold:
                    Ls[othidx, 0] = -1

        deleted = np.concatenate((deleted, Ls[Ls[:, 0] == -1, :]))
        id_1 = np.argwhere(Ls[:, 0] == -1)
        Ls = np.delete(Ls, id_1.reshape(-1), axis=0)
        Ls[:, 0] -= 1
        output.append(np.concatenate((Ls, deleted)))
    return np.array(output)


def offset_to_loc(anchors, cls_preds, bbox_preds):
    # print(len(cls_preds), len(anchors[0]))
    bbox_preds = bbox_preds.reshape((len(cls_preds), len(anchors[0]), 4))
    # print('bbox shape=', bbox_preds.shape())
    ux = uy = uw = uh = 0
    ax = ay = 0.1
    aw = ah = 0.2

    batch_bbox = []
    for bn in range(len(cls_preds)):
        c = np.argmax(cls_preds[bn], axis=1)
        c1 = np.expand_dims(c, axis=-1)
        bbox = []
        for bbox_idx in range(len(c1)):
            an_x1 = anchors[0, bbox_idx, 0]
            an_y1 = anchors[0, bbox_idx, 1]
            an_x2 = anchors[0, bbox_idx, 2]
            an_y2 = anchors[0, bbox_idx, 3]

            # print(c1[bbox_idx])
            # if c1[bbox_idx] == 0:
            #     bbox.append([an_x1, an_y1, an_x2, an_y2])
            #     continue

            # print(an_x1, an_y1, an_x2, an_y2)

            # print(bbox_preds)
            off_xc = bbox_preds[bn, bbox_idx, 0]
            off_yc = bbox_preds[bn, bbox_idx, 1]
            off_w = bbox_preds[bn, bbox_idx, 2]
            off_h = bbox_preds[bn, bbox_idx, 3]
            # print('-', off_xc, off_yc, off_w, off_h)

            wa = an_x2 - an_x1
            ha = an_y2 - an_y1
            xa = an_x1 + wa / 2  # centor points
            ya = an_y1 + ha / 2  # centor points

            xb = (off_xc * ax + ux) * wa + xa
            yb = (off_yc * ay + uy) * ha + ya
            wb = np.exp(off_w * aw + uw) * wa
            hb = np.exp(off_h * ah + uh) * ha

            # print('>>', xb, yb, wb, hb)
            bbox.append([xb - wb / 2, yb - hb / 2, xb + wb / 2, yb + hb / 2])
        batch_bbox.append(bbox)
    return np.array(batch_bbox)


class PikachuDataGenerator(keras.utils.Sequence):
    def __init__(self, anchors, target_size=(256, 256), batch_size=32, num_class=1, shuffle=True, is_train=True, data_dir = 'data/pikachu'):
        self.anchors = anchors
        self.num_class = num_class
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gen = ImageDataGenerator()
        self.debug_y = None
        self.cls_onehot = False

        if is_train:
            json_path = data_dir + '/train/annotations.json'
            img_path = data_dir + '/train/images/'
        else:
            json_path = data_dir + '/val/annotations.json'
            img_path = data_dir + "/val/images/"

        with open(json_path) as f:
            data_list = json.load(f)
            data = [[img_path + data_list[d]['image'], np.array([[data_list[d]['class']] + data_list[d]['loc']])]
                    for d in data_list]
            dataframe = pd.DataFrame(data, columns=['image', 'label'])
            self.len = len(dataframe.index)

        self.iter = self.gen.flow_from_dataframe(
            dataframe=dataframe,
            directory=None,
            x_col='image',
            y_col='label',
            class_mode='raw',
            batch_size=batch_size,
            target_size=target_size,
            shuffle=shuffle
        )

    def __getitem__(self, index):
        x, y = self.iter.next()
        self.debug_y = y
        off, mask, cls = MultiBoxTarget(self.anchors, y)
        # loc_mask = np.stack((off, mask), axis=-1)
        # print(cls.shape, self.num_class)
        if self.cls_onehot:
            cls = np.array([self.to_categorical(batch, self.num_class+1) for batch in cls])
        return x, [cls, off, mask]

    # return: x, [cls, loc_mask]
    # x:        (b, w, h, 3)
    # cls:      (b, an)
    # loc_mask: (b, an * 4, 2) -> 2: [offset, mask]
    def getitem(self):
        return self.__getitem__(0)

    # return: (b, 5) -> 5: [class, x1, y1, x2, y2]
    def gety(self):
        return self.debug_y

    def __len__(self):
        return (self.len // self.batch_size) + 1


def load_data_pikachu(anchors, batch_size, edge_size=256, data_dir='../data/pikachu'):  # edge_size：输出图像的宽和高
    train_iter = PikachuDataGenerator(
        anchors, data_dir=data_dir,
        target_size=(edge_size, edge_size), batch_size=batch_size, shuffle=True, is_train=True)
    val_iter = PikachuDataGenerator(
        anchors, data_dir=data_dir,
        target_size=(edge_size, edge_size), batch_size=5, shuffle=False, is_train=False)
    return train_iter, val_iter
