#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:23:16 2020

@author: pierre
"""

import argparse

import dnnlib
import tensorflow as tf
import util

import numpy as np


parser = argparse.ArgumentParser("Training analysis")

parser.add_argument("network_dir", help="path to network directory")
parser.add_argument("tf_train", help="path to tf train datasets")

#network_dir = "/home/pierre/cam/denoising/noise2noise/results/00024-autoencoder"
#tf_train = "/mnt/data/denoising_data/tf_pa_train_prepost.tf"

args = parser.parse_args()

if tf.get_default_session() is None:
    session = tf.Session(config=tf.ConfigProto())
    session._default_session = session.as_default()
    session._default_session.enforce_nesting = False
    session._default_session.__enter__() # pylint: disable=no-member

net = util.load_snapshot(args.network_dir + "/network_169000.pickle")

reader = tf.TFRecordReader()

feats = {'shape': tf.FixedLenFeature([3], tf.int64),
         'data1': tf.FixedLenFeature([], tf.string),
         'data2': tf.FixedLenFeature([], tf.string)}

def _parse_image_function(example_proto):
  return tf.parse_single_example(example_proto, feats)

raw_image_dataset = tf.data.TFRecordDataset(args.tf_train)
dataset = raw_image_dataset.map(_parse_image_function)
dat = dataset.make_one_shot_iterator().get_next()
print(dat)
assert(False)
try:
    errs = []
    while True:
        target_img = tf.reshape(tf.decode_raw(dat["data2"], tf.uint8), dat["shape"]).eval()
        pred_img = util.infer_image(net, target_img)

        target_img = util.clip_to_uint8(np.mean(target_img, axis=0))
        pred_img = util.clip_to_uint8(np.mean(pred_img, axis=0))

        errs.append(sum(np.sqrt((target_img - pred_img)**2).flatten()))
        print(errs)

except tf.errors.OutOfRangeError:
    pass
