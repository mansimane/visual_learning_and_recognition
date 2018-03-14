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
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

readerf = tf.train.NewCheckpointReader("/tmp/02_pascal_model_scratch_bak/model.ckpt-40000")

f = readerf.get_tensor("conv2d/kernel")


final_filt=put_kernels_on_grid (tf.convert_to_tensor(f), pad = 1)


im_summary=tf.summary.image('final_filt', final_filt)


all_summary = tf.summary.merge([im_summary])


with tf.Session() as sess:
    summary = sess.run(all_summary)
    writer = tf.summary.FileWriter('./fit')
    writer.add_summary(summary)
    writer.close()


