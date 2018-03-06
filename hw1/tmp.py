
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

num_classes = 20
BATCH_SIZE = 10
no_of_iters = 4000
no_of_pts = 10
no_of_steps = no_of_iters / no_of_pts
log_dir = "/home/ubuntu/assignments/04_pascal_fine_tune"
rdr = tf.train.NewCheckpointReader('/home/ubuntu/assignments/vgg_16.ckpt')


def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    # """Model function for CNN."""
    # Input Layer

    input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])

    if mode == tf.estimator.ModeKeys.TRAIN:
        flipped = tf.map_fn(lambda image: tf.image.random_flip_left_right(image), features["x"])
        cropped = tf.map_fn(lambda image: tf.random_crop(image, size=[224, 224, 3]), features["x"])

        fets = tf.concat([features["x"], flipped, cropped], axis=0)
        lbls = tf.concat([labels, labels, labels], axis=0)

        feats = tf.random_shuffle(fets, seed=features["x"].shape[0] * 3)
        lbels = tf.random_shuffle(lbls, seed=features["x"].shape[0] * 3)

        features["x"] = feats
        input_layer = features["x"]
        labels = lbels

    tf.summary.image("images", input_layer)

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
        multi_class_labels=labels, logits=logits), name='loss')

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


def load_pascal(data_dir, split='train'):
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
    # Wrote this function
    H = 256
    W = 256

    N_dict = {'train': 2501, 'val': 2510, 'trainval': 5011, 'test': 4952}
    N = N_dict[split]

    images = np.zeros((N, H, W, 3), dtype=np.float32)
    labels = np.zeros((N, num_classes), dtype=np.int32)
    weights = np.zeros((N, num_classes), dtype=np.int32)

    #Load Images
    file_name = data_dir + "ImageSets/Main/" + split + '.txt'
    with open(file_name, 'r') as image_lst_file:
        i=0
        for line in image_lst_file.readlines():
            img_no = line.strip('\n')
            image_name = data_dir + 'JPEGImages/' + img_no + '.jpg'
            img = Image.open(image_name)
            img = img.resize((H, W), Image.ANTIALIAS)
            imgarr = np.asarray(img, dtype=np.float32)
            images[i, :, :, :] = imgarr
            i += 1

    for i in range(len(CLASS_NAMES)):
        cls = CLASS_NAMES[i]
        file_name = data_dir + "ImageSets/Main/" + cls + '_' + split + '.txt'
        with open(file_name, 'r') as image_lst_file:
            im_idx = 0
            for line in image_lst_file.readlines():
                _, is_present = line.strip('\n').split()
                is_present = int(is_present)
                if is_present == 1:
                    labels[im_idx][i] = 1

                if is_present != 0:
                    weights[im_idx][i] = 1
                im_idx += 1
    return images, labels, weights



def main():
    args = parse_args()

    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    np.save(os.path.join(args.data_dir, 'trainval' + '_data_images'), train_data)
    np.save(os.path.join(args.data_dir, 'trainval' + '_data_labels'), train_labels)
    np.save(os.path.join(args.data_dir, 'trainval' + '_data_weights'), train_weights)

    # train_data = np.load(os.path.join(args.data_dir, 'trainval' + '_data_images.npy'))
    # train_labels = np.load(os.path.join(args.data_dir, 'trainval' + '_data_labels.npy'))
    # train_weights = np.load(os.path.join(args.data_dir, 'trainval' + '_data_weights.npy'))

    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    np.save(os.path.join(args.data_dir, 'test' + '_data_images'), eval_data)
    np.save(os.path.join(args.data_dir, 'test' + '_data_labels'), eval_labels)
    np.save(os.path.join(args.data_dir, 'test' + '_data_weights'), eval_weights)
    # eval_data = np.load(os.path.join(args.data_dir, 'test' + '_data_images.npy'))
    # eval_labels = np.load(os.path.join(args.data_dir, 'test' + '_data_labels.npy'))
    # eval_weights = np.load(os.path.join(args.data_dir, 'test' + '_data_weights.npy'))



    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir=log_dir)

    # logging loss
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=no_of_steps)

    # summary_hook = tf.train.SummarySaverHook(
    #     SAVE_EVERY_N_STEPS,
    #     output_dir='/tmp/tf',
    #     summary_op=tf.summary.merge_all())

    # logging lr
    # tensors_to_log2 = {"learning_rate": "learning_rate"}
    # logging_hook2 = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log2, every_n_iter=100)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    for i in range(no_of_pts):
        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=no_of_steps,
            hooks=[logging_hook])
        # Evaluate the model and print results
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

        summary_var(log_dir=log_dir,
                    name="mAP", val=np.mean(AP), step=i * no_of_steps)


if __name__ == "__main__":
    main()
