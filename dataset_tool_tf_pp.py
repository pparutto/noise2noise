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

import PIL.Image
import numpy as np

from collections import defaultdict

def load_stack(fname):
    im = PIL.Image.open(fname)
    h,w = np.shape(im)

    res = np.zeros((im.n_frames, 3, h, w))
    for i in range(im.n_frames):
        im.seek(i)
        res[i, 0, :, :] = np.array(im)
        res[i, 1, :, :] = np.array(im)
        res[i, 2, :, :] = np.array(im)
    res = res.astype("uint8")

    return res


def shape_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

examples='''examples:

  python %(prog)s --input-dir=./kodak --out=imagenet_val_raw.tfrecords
'''

def main():
    parser = argparse.ArgumentParser(
        description='Convert a set of image files into a TensorFlow tfrecords training set.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_stack", help="Input image stack")
    parser.add_argument("out", help="Filename of the output tfrecords file")
    args = parser.parse_args()

    print ('Loading image stack: %s' % args.input_stack)
    #np.random.RandomState(0x1234f00d).shuffle(images)

    images = load_stack(args.input_stack)

    #----------------------------------------------------------
    outdir = os.path.dirname(args.out)
    os.makedirs(outdir, exist_ok=True)
    writer = tf.python_io.TFRecordWriter(args.out)
    for i in range(images.shape[0] // 2):
        feature = {
          'shape': shape_feature(images[i].shape),
          'data1': bytes_feature(tf.compat.as_bytes(images[i,:,:,:].tostring())),
          'data2': bytes_feature(tf.compat.as_bytes(images[i+1,:,:,:].tostring()))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    print(images.shape)


if __name__ == "__main__":
    main()
