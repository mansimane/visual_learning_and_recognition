from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial

from eval import compute_map
#import models
from tensorflow.core.framework import summary_pb2
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
reader = tf.train.NewCheckpointReader('/home/ubuntu/assignments/vgg_16.ckpt')
#Local
# init_path = tf.train.NewCheckpointReader('vgg_16.ckpt')


def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    # Referred :https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py#L243
    # Data Augmentation: https://stackoverflow.com/questions/38920240/tensorflow-image-operations-for-batches
    # Fine tuning: https://gist.github.com/leonidk/b0d21c8ed68c69abe985f713d0af15a1, line 268
    # ***crop size is 227?

    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

    flipped_imgs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_layer)
    distorted_image = tf.map_fn(lambda img: tf.random_crop(img, [224, 224, 3]), flipped_imgs)
    tf.summary.image("train_images", distorted_image, max_outputs=40)
    norm_imgs = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), distorted_image)

    # for old_name in reader.get_variable_to_shape_map():
    #     #print(old_name)
    #     tensor = reader.get_tensor(old_name)
    ####### BLOCK 1
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=norm_imgs,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv1/conv1_1/biases'),verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv1/conv1_1/weights'),verify_shape=True),
        use_bias=True)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv1/conv1_2/biases'), verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv1/conv1_2/weights'), verify_shape=True),
        use_bias=True)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)  #

    ####### BLOCK 2
    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv2/conv2_1/biases'), verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv2/conv2_1/weights'), verify_shape=True),
        use_bias=True)

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv2/conv2_2/biases'), verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv2/conv2_2/weights'), verify_shape=True),
        use_bias=True)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    ####### BLOCK 3###
    # Convolutional Layer #3
    conv5 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv3/conv3_1/biases'),verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv3/conv3_1/weights'),verify_shape=True),
        use_bias=True)

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv3/conv3_2/biases'),verify_shape=True ),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv3/conv3_2/weights'),verify_shape=True),
        use_bias=True)

    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv3/conv3_3/biases'), verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv3/conv3_3/weights'), verify_shape=True),
        use_bias=True)

    # Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

    ####### BLOCK 4 ###
    # Convolutional Layer #3
    conv8 = tf.layers.conv2d(
        inputs=pool3,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv4/conv4_1/biases'), verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv4/conv4_1/weights'), verify_shape=True),
        use_bias=True)

    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv4/conv4_2/biases'), verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv4/conv4_2/weights'), verify_shape=True),
        use_bias=True)

    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv4/conv4_3/biases'), verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv4/conv4_3/weights'), verify_shape=True),
        use_bias=True)

    # Pooling Layer #3
    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

    ####### BLOCK 5 ###
    # Convolutional Layer #10
    conv11 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv5/conv5_1/biases'), verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv5/conv5_1/weights'), verify_shape=True),
        use_bias=True)

    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv5/conv5_2/biases'), verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv5/conv5_2/weights'), verify_shape=True),
        use_bias=True)

    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1,
        bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv5/conv5_3/biases'), verify_shape=True),
        kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/conv5/conv5_3/weights'), verify_shape=True),
        use_bias=True)

    # Pooling Layer #3
    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)

    # Dense Layer 1, FC 6
#    pool3_flat = tf.reshape(pool2, [-1, 64 * 64 * 64]) #*** use flatten?
    #pool5_flat = tf.contrib.layers.flatten(pool5, outputs_collections=None, scope=None) #*** use flatten?


    dense1 = tf.layers.conv2d(inputs=pool5, filters=4096,
                              kernel_size=[7, 7],
                              padding="same",
                              activation=tf.nn.relu,
                             bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/fc6/biases'), verify_shape=True),
                             kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/fc6/weights'), verify_shape=True),
                             use_bias=True)

    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense Layer 2, FC7
    dense2 = tf.layers.conv2d(inputs=dropout1, filters=4096,
                              kernel_size=[7, 7],
                              padding="same",
                              activation=tf.nn.relu,
                             bias_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/fc7/biases'), verify_shape=True),
                             kernel_initializer=tf.constant_initializer(reader.get_tensor(value='vgg_16/fc7/weights'), verify_shape=True),
                             use_bias=True)

    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense3 = tf.layers.conv2d(inputs=dropout2, filters=1000,
                              activation=tf.nn.relu,
                              kernel_size=[1, 1],
                              padding="same",
                              use_bias=True)

    dropout3 = tf.layers.dropout(
        inputs=dense3, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer, FC 8
    logits = tf.layers.dense(inputs=dropout3, units=num_classes)

    probs = tf.sigmoid(logits, name="sigmoid_tensor")
    pred_float = tf.greater_equal(probs, 0.5)
    pred_int = tf.cast(pred_float, dtype="int32")

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": pred_int,
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": probs
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
#    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
         multi_class_labels=labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    #batch_no = tf.train.get_global_step()

    batch_no = tf.Variable(0, dtype=tf.float32)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        learning_rate = tf.train.exponential_decay(
            0.0001,  # Base learning rate.
            tf.train.get_global_step(),  # Current index into the dataset.
            1000,  # Decay step.
            0.05,  # Decay rate.
            staircase=False)

        tf.summary.scalar('learning_rate', learning_rate)

        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               0.9)
        #summary_var(log_dir=log_dir,
        #            name="learning_rate", val=str(learning_rate), step=tf.train.get_global_step())
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        train_summary = []
        grads_and_vars=optimizer.compute_gradients(loss)
        for g, v in grads_and_vars:
            if g is not None:
                #print(format(v.name))
                grad_hist_summary = tf.summary.histogram("{}/grad_histogram".format(v.name), g)
                train_summary.append(grad_hist_summary)

        #tf.summary.merge(train_summary)
        tf.summary.merge_all()

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)


    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    #logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=200)

    # in main logging: histograms: tf.summary, summary_saver
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


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

def summary_var(log_dir, name, val, step):
    writer = tf.summary.FileWriterCache.get(log_dir)
    summary_proto = summary_pb2.Summary()
    value = summary_proto.value.add()
    value.tag = name
    value.simple_value = float(val)
    writer.add_summary(summary_proto, step)
    writer.flush()

def summary_var_lr(log_dir, name, val, step):
    writer = tf.summary.FileWriterCache.get(log_dir)
    summary_proto = summary_pb2.Summary()
    value = summary_proto.value.add()
    value.tag = name
    value.simple_value = val
    writer.add_summary(summary_proto, step)
    writer.flush()

def main():
    args = parse_args()

    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
         args.data_dir, split='trainval')
    np.save(os.path.join(args.data_dir, 'trainval' + '_data_images'), train_data)
    np.save(os.path.join(args.data_dir, 'trainval' + '_data_labels'), train_labels)
    np.save(os.path.join(args.data_dir, 'trainval' + '_data_weights'), train_weights)

    #train_data = np.load(os.path.join(args.data_dir, 'trainval' + '_data_images.npy'))
    #train_labels = np.load(os.path.join(args.data_dir, 'trainval' + '_data_labels.npy'))
    #train_weights = np.load(os.path.join(args.data_dir, 'trainval' + '_data_weights.npy'))

    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')
    
    np.save(os.path.join(args.data_dir, 'test' + '_data_images'), eval_data)
    np.save(os.path.join(args.data_dir, 'test' + '_data_labels'), eval_labels)
    np.save(os.path.join(args.data_dir, 'test' + '_data_weights'), eval_weights)
    #eval_data = np.load(os.path.join(args.data_dir, 'test' + '_data_images.npy'))
    #eval_labels = np.load(os.path.join(args.data_dir, 'test' + '_data_labels.npy'))
    #eval_weights = np.load(os.path.join(args.data_dir, 'test' + '_data_weights.npy'))



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
                    name="mAP", val=np.mean(AP), step=i*no_of_steps)


    #######  Image logging
if __name__ == "__main__":
    main()
