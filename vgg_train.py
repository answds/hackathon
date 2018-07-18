#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
import datetime
import numpy as np
import data_helpers
import tensorflow as tf
from vgg_model import vgg16

batch_size, num_epochs = 12, 50
num_classes = 2
pos_path, neg_path = "data/positive/", "data/negative/"
x_train, y_train, x_dev, y_dev = data_helpers.load_data(pos_path, neg_path)
print("load data is ok...")


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        vggmodel = vgg16(
            height=data_helpers.h,
            width=data_helpers.w,
            channel=data_helpers.c,
            number_classes=num_classes
        )
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(vggmodel.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", vggmodel.loss)
        acc_summary = tf.summary.scalar("accuracy", vggmodel.accuracy)
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())
        init = tf.global_variables_initializer()
        sess.run(init)
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join("./", "runs", timestamp))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("Writing to {}\n".format(out_dir))
        checkpoint_prefix = os.path.join(out_dir, "model")

        def train_step(x_batch, y_batch):
            feed_dict = {
                vggmodel.input_x: x_batch,
                vggmodel.input_y: y_batch
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, vggmodel.loss, vggmodel.accuracy],
                feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            # if step % 100 == 0:
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return loss, accuracy


        def dev_step(epoch, dev_batch_size=24):
            dev_batches = data_helpers.get_per_epochs_data(x_dev, y_dev, batch_size)
            loss_list, acc_list = list(), list()
            for dev_batch in dev_batches:
                x_dev_batch, y_dev_batch = dev_batch
                feed_dict = {
                    vggmodel.input_x: x_dev_batch,
                    vggmodel.input_y: y_dev_batch,
                }
                step, loss, accuracy, prediction = sess.run(
                    [global_step, vggmodel.loss, vggmodel.accuracy, vggmodel.predict_labels],
                    feed_dict)
                if not np.isnan(loss):
                    loss_list.append(loss)
                    acc_list.append(accuracy)
                else:
                    print("dev: ", x_dev_batch, y_dev_batch)
            print_message("Evaluation: ", epoch, loss_list, acc_list)


        def print_message(train_evalution, epoch, loss_list, accuracy_list):
            loss_mean = sum(loss_list) / len(loss_list)
            acc_mean = sum(accuracy_list) / len(accuracy_list)
            time_str = datetime.datetime.now().isoformat()
            print(train_evalution)
            print("{}: epoch: {}, loss {:g}, acc {:g}".format(time_str, epoch, loss_mean, acc_mean))


        for epoch in range(num_epochs):
            train_batches = data_helpers.get_per_epochs_data(x_train, y_train, batch_size)
            train_loss_list, train_accuracy_lis = list(), list()
            for batch in train_batches:
                x_batch, y_batch = batch
                batch_loss, batch_accuracy = train_step(x_batch, y_batch)
                if not np.isnan(batch_loss):
                    train_loss_list.append(batch_loss)
                    train_accuracy_lis.append(batch_accuracy)
                else:
                    print("train: ", x_batch, y_batch)

            print_message("Train: ", epoch+1, train_loss_list, train_accuracy_lis)
            dev_step(epoch+1)
            print("")
