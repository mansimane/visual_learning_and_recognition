#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 02:07:46 2018

@author: snigdha
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import os
import numpy as np
import tensorflow as tf
import argparse
# import os.path as osp
from PIL import Image
from functools import partial
from collections import defaultdict
import pickle

from eval import compute_map

# import models
log_dir = '/home/ubuntu/assignments/04_pascal_fine_tune'
tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

rdr = tf.train.NewCheckpointReader("/home/ubuntu/assignments/vgg_16.ckp")


def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    # """Model function for CNN."""
    # Input Layer

    input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])

    if mode == tf.estimator.ModeKeys.TRAIN:
        flipped = tf.map_fn(lambda image: tf.image.random_flip_left_right(image), features["x"])
        cropped = tf.map_fn(lambda image: tf.random_crop(image, size=[224, 224, 3]), features["x"])

        fets = tf.concat([features["x"], flipped, cropped], axis=0)
        # wts = tf.concat([features["w"],features["w"],features["w"]],axis = 0)
        lbls = tf.concat([labels, labels, labels], axis=0)

        feats = tf.random_shuffle(fets, seed=features["x"].shape[0] * 3)
        # wtgs = tf.random_shuffle(wts,seed = features["x"].shape[0]*3)
        lbels = tf.random_shuffle(lbls, seed=features["x"].shape[0] * 3)

        features["x"] = feats
        input_layer = features["x"]
        labels = lbels

    tf.summary.image("Training_images", input_layer)

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv1/conv1_1/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv1/conv1_1/biases'),
                                                 verify_shape=True))

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv1/conv1_2/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv1/conv1_2/biases'),
                                                 verify_shape=True))

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 
    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv2/conv2_1/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv2/conv2_1/biases'),
                                                 verify_shape=True))

    # Convolutional Layer #4 
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv2/conv2_2/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv2/conv2_2/biases'),
                                                 verify_shape=True))

    # Pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # Convolutional Layer #5 
    conv5 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv3/conv3_1/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv3/conv3_1/biases'),
                                                 verify_shape=True))

    # Convolutional Layer #6 
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv3/conv3_2/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv3/conv3_2/biases'),
                                                 verify_shape=True))

    # Convolutional Layer #7
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv3/conv3_3/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv3/conv3_3/biases'),
                                                 verify_shape=True))

    # Pooling layer 3
    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

    # Convolutional Layer #8 
    conv8 = tf.layers.conv2d(
        inputs=pool3,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv4/conv4_1/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv4/conv4_1/biases'),
                                                 verify_shape=True))

    # Convolutional Layer #9
    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv4/conv4_2/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv4/conv4_2/biases'),
                                                 verify_shape=True))

    # Convolutional Layer #10
    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv4/conv4_3/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv4/conv4_3/biases'),
                                                 verify_shape=True))

    # Pooling layer 4
    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

    # Convolutional Layer #11
    conv11 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv5/conv5_1/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv5/conv5_1/biases'),
                                                 verify_shape=True))

    # Convolutional Layer #12
    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv5/conv5_2/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv5/conv5_2/biases'),
                                                 verify_shape=True))

    # Convolutional Layer #13
    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv5/conv5_3/weights'),
                                                   verify_shape=True),
        bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/conv5/conv5_3/biases'),
                                                 verify_shape=True))

    # Pooling layer 5
    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)

    dense1 = tf.layers.conv2d(inputs=pool5,
                              activation=tf.nn.relu,
                              filters=4096,  # this specifies the number of channels in the output layer
                              kernel_size=[7, 7],
                              strides=[1, 1],
                              padding="same",
                              kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/fc6/weights'),
                                                                         verify_shape=True),
                              bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/fc6/biases'),
                                                                       verify_shape=True))

    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.conv2d(inputs=dropout1,
                              filters=4096,  # this specifies the number of channels in the output layer
                              kernel_size=[1, 1],
                              strides=[1, 1],
                              padding="same",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/fc7/weights'),
                                                                         verify_shape=True),
                              bias_initializer=tf.constant_initializer(value=rdr.get_tensor('vgg_16/fc7/biases'),
                                                                       verify_shape=True))

    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense3 = tf.layers.conv2d(inputs=dropout2,
                              filters=1000,  # this specifies the number of channels in the output layer
                              kernel_size=[1, 1],
                              strides=[1, 1],
                              padding="same",
                              activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=tf.contrib.layers.flatten(dense3), units=20)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)

    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        labels, logits=logits), name='loss')
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:



        decay_learning_rate = tf.train.exponential_decay(
            learning_rate=0.0001,
            global_step=tf.train.get_global_step(),
            decay_steps=1000,
            decay_rate=0.5,
            staircase=False,
            name=None)
        optimizer = tf.train.MomentumOptimizer(learning_rate=decay_learning_rate,
                                               momentum=0.9)

        tf.summary.scalar("learning_rate", decay_learning_rate)

        grads_and_vars = optimizer.compute_gradients(loss)

        for g, v in grads_and_vars:
            if g is not None:
                # print(format(v.name))
                tf.summary.histogram("{}/grad_histogram".format(v.name), g)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


from tensorflow.core.framework import summary_pb2


def summary_var(log_dir, name, val, step):
    writer = tf.summary.FileWriterCache.get(log_dir)
    summary_proto = summary_pb2.Summary()
    value = summary_proto.value.add()
    value.tag = name
    value.simple_value = float(val)
    writer.add_summary(summary_proto, step)
    writer.flush()


def load_pascal_afs(data_dir, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that 
            are ambiguous.
    """
    class_dict = {}
    k = len(CLASS_NAMES)
    # split ='train'

    label_path = data_dir + "ImageSets/Main/"
    im_path = data_dir + "JPEGImages/"

    for i, name in enumerate(CLASS_NAMES):
        class_dict[i] = name

        with open(label_path + name + "_" + split + ".txt") as f:
            a = f.read().split()
            class_label = (np.array(a[1::2], dtype=np.int32) > 0) * 1
            class_weight = np.abs(np.array(a[1::2], dtype=np.int32))

            if i == 0:
                image_name = a[::2]
                N = len(image_name)
                labels = np.zeros([N, k], dtype=np.int32)
                weights = np.zeros([N, k], dtype=np.int32)
                images = np.zeros([N, 224, 224, 3], dtype=np.float32)

                # get the images 
                for n in range(N):
                    I = np.asarray(Image.open(im_path + image_name[n] + ".jpg").resize([224, 224]))
                    images[n, :, :, :] = I

        labels[:, i] = class_label
        weights[:, i] = class_weight

    labels = np.array(labels, dtype=np.int32)
    weights = np.array(weights, dtype=np.int32)
    images = np.array(images, dtype=np.float32)

    return images, labels, weights


def load_pascal(data_dir, split='train'):
    '''
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that 
            are ambiguous.
   '''

    wts = []
    lbels = []
    # len_CN = len(CLASS_NAMES)
    for i in CLASS_NAMES:
        with open(data_dir + '/ImageSets/Main/' + i + '_' + split + '.txt') as f1:
            a1 = f1.read().split()
            img = a1[::2]
            print(img)
            N = len(img)
            l = a1[1::2]
            print("l", l)
            wts = np.append(wts, np.abs(np.array(l, dtype=np.int32)), axis=0)
            print("wts", wts)
            lbels = np.append(lbels, (np.array(l, dtype=np.int32) > 0) * 1, axis=0)
            print("lbels", lbels)
    weights = np.reshape(wts, (N, 20))
    labels = np.reshape(lbels, (N, 20))

    no_i = len(img)
    # print(Num)
    images = np.ndarray(shape=(no_i, 256, 256, 3), dtype=np.float32)
    for i in img:
        N = len(img)
        im1 = Image.open(data_dir + '/JPEGImages/' + i + '.jpg')
        im2 = im1.convert('RGB')
        im3 = np.asarray(im2)
        im4 = np.resize(im3, (256, 256, 3))
        images[1:no_i:1, :, :, :] = np.abs(np.array(im4, dtype=np.int32))

    print("ok")
    return (images, labels, weights)


def main():
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal_afs(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal_afs(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir=log_dir)
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=400)

    list22 = []
    for i in range(0, 10):

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data, "w": train_weights},
            y=train_labels,
            batch_size=10,
            num_epochs=None,
            shuffle=True)

        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=400,
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data, "w": eval_weights},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)

        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        pred = np.stack([p['probabilities'] for p in pred])
        rand_AP = compute_map(
            eval_labels, np.random.random(eval_labels.shape),
            eval_weights, average=None)
        print('Random AP: {} mAP'.format(np.mean(rand_AP)))
        gt_AP = compute_map(
            eval_labels, eval_labels, eval_weights, average=None)
        print('GT AP: {} mAP'.format(np.mean(gt_AP)))
        AP = compute_map(eval_labels, pred, eval_weights, average=None)
        print('Obtained {} mAP'.format(np.mean(AP)))
        print('per class:')
        for cid, cname in enumerate(CLASS_NAMES):
            print('{}: {}'.format(cname, _get_el(AP, cid)))
        list22.append(np.mean(AP))
        summary_var("pascal_vggfinetune", "mAP", np.mean(AP), i * 400)

    with open('list22.pkl', 'wb') as fr2:
        pickle.dump(list22, fr2)


if __name__ == "__main__":
    main()