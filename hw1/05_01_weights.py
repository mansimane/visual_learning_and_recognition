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
from funcs import *
tf.logging.set_verbosity(tf.logging.INFO)

def main():
    readerf = tf.train.NewCheckpointReader("/tmp/02_pascal_model_scratch_bak/model.ckpt-40000")
    readerm = tf.train.NewCheckpointReader("./plots/02_pascal_alexnet_iters60/model.ckpt-24000")
    readers = tf.train.NewCheckpointReader("./plots/02_pascal_alexnet_iters100/model.ckpt-40000")

    x30 = readerf.get_tensor("conv2d/kernel")
    x60 = readerm.get_tensor("conv2d/kernel")
    x100 = readers.get_tensor("conv2d/kernel")
    filters30=put_kernels_on_grid (tf.convert_to_tensor(x30), pad = 1)
    filters60=put_kernels_on_grid (tf.convert_to_tensor(x60), pad = 1)
    filters100=put_kernels_on_grid (tf.convert_to_tensor(x100), pad = 1)

    im_summary30=tf.summary.image('filters30', filters30)
    im_summary60=tf.summary.image('filters60', filters60)
    im_summary100=tf.summary.image('filters100', filters100)

    all_summary = tf.summary.merge([im_summary30, im_summary60, im_summary100])




    with tf.Session() as sess:
        # Run
        summary = sess.run(all_summary)
        # Write summary
        writer = tf.summary.FileWriter(DIR)
        writer.add_summary(summary)
        writer.close()

if __name__ == "__main__":
    main()
