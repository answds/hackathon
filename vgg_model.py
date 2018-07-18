#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]


class vgg16(object):

    def __init__(self, height, width, channel, number_classes):
        self.number_classes = number_classes
        self.input_x = tf.placeholder(tf.float32, [None, height, width, channel], name="input_x")
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.input_x)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.input_y = tf.placeholder(tf.int32, [None, number_classes], name="input_y")
        self.finaloutput = self.forward_calculation(input_x=bgr)
        self.compute_loss_acc(self.finaloutput)

    def forward_calculation(self, input_x):
        # conv1
        conv1_1 = self.convolution_layer(input_x, [3, 3, 3, 64], [64], "conv1_1")
        conv1_2 = self.convolution_layer(conv1_1, [3, 3, 64, 64], [64], "conv1_2")
        pooling_1 = self.max_pooling(conv1_2)
        print(pooling_1)

        # conv2
        conv2_1 = self.convolution_layer(pooling_1, [3, 3, 64, 128], [128], "conv2_1")
        conv2_2 = self.convolution_layer(conv2_1, [3, 3, 128, 128], [128], "conv2_2")
        pooling_2 = self.max_pooling(conv2_2)
        print(pooling_2)

        # conv3
        conv3_1 = self.convolution_layer(pooling_2, [3, 3, 128, 256], [256], "conv3_1")
        conv3_2 = self.convolution_layer(conv3_1, [3, 3, 256, 256], [256], "conv3_2")
        pooling_3 = self.max_pooling(conv3_2)
        print(pooling_3)

        # conv4
        conv4_1 = self.convolution_layer(pooling_3, [3, 3, 256, 512], [512], "conv4_1")
        conv4_2 = self.convolution_layer(conv4_1, [3, 3, 512, 512], [512], "conv4_2")
        conv4_3 = self.convolution_layer(conv4_2, [3, 3, 512, 512], [512], "conv4_3")
        pooling_4 = self.max_pooling(conv4_3)
        print(pooling_4)

        # conv5
        conv5_1 = self.convolution_layer(pooling_4, [3, 3, 512, 512], [512], "conv5_1")
        conv5_2 = self.convolution_layer(conv5_1, [3, 3, 512, 512], [512], "conv5_2")
        conv5_3 = self.convolution_layer(conv5_2, [3, 3, 512, 512], [512], "conv5_3")
        pooling_5 = self.max_pooling(conv5_3)
        print(pooling_5)

        # fc 6
        fc6_shape_1 = int(np.prod(pooling_5.get_shape()[1:]))
        pool5_flat = tf.reshape(pooling_5, [-1, fc6_shape_1])
        output_fc6 = tf.nn.relu(self.full_connect_layer(pool5_flat, [fc6_shape_1, 4096], [4096]))
        print(output_fc6)

        # fc 7
        output_fc7 = tf.nn.relu(self.full_connect_layer(output_fc6, [4096, 4096], [4096]))
        print(output_fc7)

        # fc 8
        output_fc8 = tf.nn.relu(self.full_connect_layer(output_fc7, [4096, self.number_classes], [self.number_classes]))
        print(output_fc8)

        finaloutput = tf.nn.softmax(output_fc8, name="softmax")
        print(finaloutput)

        return finaloutput

    def compute_loss_acc(self, finaloutput):
        self.predict_labels = tf.argmax(finaloutput, axis=1, name="output")
        correct_prediction = tf.equal(self.predict_labels, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=finaloutput, labels=self.input_y)
        self.loss = tf.reduce_mean(losses)

    def weight_variable(self, shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    # 卷积
    def convolution_layer(self, input_data, kernel_shape, biases_shape, scope_name):
        kernel = self.weight_variable(kernel_shape)
        conv = tf.nn.conv2d(
            input=input_data,
            filter=kernel,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )
        biases = self.bias_variable(biases_shape)
        convl = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope_name)
        return convl

    # 池化
    def max_pooling(self, conv):
        pool = tf.nn.max_pool(
            conv,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool1'
        )
        return pool

    # 全连接层
    def full_connect_layer(self, x, w_shape, b_shape):
        w = self.weight_variable(w_shape)
        b = self.bias_variable(b_shape)
        return tf.nn.bias_add(tf.matmul(x, w), b)