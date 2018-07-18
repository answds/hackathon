#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import numpy as np
from skimage import io
from skimage import transform


w = 224
h = 224
c = 3


def read_img(filename):
    if not filename.endswith("jpg") or (not not filename.endswith("png")):
        return None
    img = io.imread(filename)
    # img = img / 255.0
    # assert (0 <= img).all() and (img <= 1.0).all()
    # short_edge = min(img.shape[:2])
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    reimg = transform.resize(img, (w, h, c))
    return reimg


def load_data(pos_path, neg_path, train_size=0.8, image_file="images_array.npy",
              label_file="label_array.npy"):
    """
    Generates a batch iterator for a dataset.
    """
    if not os.path.exists(image_file) or (not os.path.exists(label_file)):
        img_data_list = list()
        img_label_list = list()
        posfilelist = os.listdir(pos_path)
        for fname in posfilelist:
            filename = pos_path + fname
            img_data = read_img(filename)
            img_data_list.append(img_data)
            img_label_list.append([0, 1])

        negfilelist = os.listdir(neg_path)
        for fname in negfilelist:
            filename = neg_path + fname
            img_data = read_img(filename)
            img_data_list.append(img_data)
            img_label_list.append([1, 0])
        img_datas, img_labels = np.asarray(img_data_list, dtype=np.float32), np.asarray(img_label_list, dtype=np.int32)
        np.save(image_file, img_datas)
        np.save(label_file, img_labels)
    else:
        img_datas = np.load(image_file)
        img_labels = np.load(label_file)

    all_number = img_datas.shape[0]
    arr = np.arange(all_number)
    np.random.shuffle(arr)
    img_datas = img_datas[arr]
    img_labels = img_labels[arr]
    s = np.int(all_number * train_size)
    x_train, x_dev = img_datas[:s], img_datas[s:]
    y_train, y_dev = img_labels[:s], img_labels[s:]
    return x_train, y_train, x_dev, y_dev


def get_per_epochs_data(img_datas, img_labels, batch_size, shuffle=True):
    all_number = img_datas.shape[0]
    num_batches_per_epoch = int(all_number / batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(all_number))
        shuffled_data = img_datas[shuffle_indices]
        shuffled_label = img_labels[shuffle_indices]
    else:
        shuffled_data = img_datas
        shuffled_label = img_labels
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, all_number)
        yield shuffled_data[start_index:end_index], shuffled_label[start_index:end_index]
