# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import glob
import os
import sys
import argparse
import tensorflow as tf

from os import path

import PIL.Image

from numpy import array, shape, zeros
from numpy.random import RandomState

from datetime import datetime

from math import ceil

def load_stack(fname):
    im = PIL.Image.open(fname)
    h,w = shape(im)

    res = zeros((im.n_frames, 3, h, w))
    for i in range(im.n_frames):
        im.seek(i)
        res[i, 0, :, :] = array(im)
        res[i, 1, :, :] = res[i, 0, :, :]
        res[i, 2, :, :] = res[i, 0, :, :]

    return res.astype("uint8")

def shape_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


parser = argparse.ArgumentParser(description='generate traning dataset as tf format')
parser.add_argument("--seed", help="seed of random generator",
                    type=int, default=None)
parser.add_argument("input_folder",
                    help="folder containing training stacks")
parser.add_argument("perc",
                    help="percentage ([0,1]) of frame pairs per to select per stack",
                    type=float)
parser.add_argument("out", help="Filename of the output tfrecords file")

args = parser.parse_args()

if args.seed is None:
    rng = RandomState(datetime.now())
else:
    rng = RandomState(args.seed)


outdir = os.path.dirname(args.out)
os.makedirs(outdir, exist_ok=True)
writer = tf.python_io.TFRecordWriter(args.out)

ntrain = 0
for fname in [e for e in os.listdir(args.input_folder)
              if path.isfile(path.join(args.input_folder, e))]:
    stck = load_stack(path.join(args.input_folder, fname))
    idxs = rng.randint(0, high=stck.shape[0] - 1,
                       size=ceil(args.perc * stck.shape[0]))
    for i in idxs:
        offx = rng.randint(0, high=stck.shape[2] - 256)
        offy = rng.randint(0, high=stck.shape[3] - 256)

        feat = {'shape': shape_feature([3, 256, 256]),
                'data1': bytes_feature(tf.compat.as_bytes(stck[i,   :, offx:(offx+256), offy:(offy+256)].tostring())),
                'data2': bytes_feature(tf.compat.as_bytes(stck[i+1, :, offx:(offx+256), offy:(offy+256)].tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feat))
        writer.write(example.SerializeToString())
    print ('{}: added {} examples'.format(fname, len(idxs)))
    ntrain += len(idxs)

writer.close()

print("Training set size = {}".format(ntrain))

print("DONE")
