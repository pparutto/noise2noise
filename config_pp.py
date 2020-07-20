# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import dnnlib
import argparse
import sys

import dnnlib.submission.submit as submit
import dnnlib.tflib.tfutil as tfutil

import util
import validation

from PIL import Image
import numpy as np

# Submit config
# ------------------------------------------------------------------------------------------

submit_config = dnnlib.SubmitConfig()
submit_config.run_dir_root = 'results'
submit_config.run_dir_ignore += ['datasets', 'results']

desc = "autoencoder"

# Tensorflow config
# ------------------------------------------------------------------------------------------

tf_config = dnnlib.EasyDict()
tf_config["graph_options.place_pruned_graph"] = True

# Network config
# ------------------------------------------------------------------------------------------

net_config = dnnlib.EasyDict(func_name="network.autoencoder")

# Optimizer config
# ------------------------------------------------------------------------------------------

optimizer_config = dnnlib.EasyDict(beta1=0.9, beta2=0.99, epsilon=1e-8)

# Noise augmentation config
gaussian_noise_config = dnnlib.EasyDict(
    func_name='train.AugmentGaussian',
    train_stddev_rng_range=(0.0, 50.0),
    validation_stddev=25.0
)
poisson_noise_config = dnnlib.EasyDict(
    func_name='train.AugmentPoisson',
    lam_max=50.0
)

# ------------------------------------------------------------------------------------------
# Preconfigured validation sets
# datasets = {
#     'kodak':  dnnlib.EasyDict(dataset_dir='datasets/kodak'),
#     'bsd300': dnnlib.EasyDict(dataset_dir='datasets/bsd300'),
#     'set14':  dnnlib.EasyDict(dataset_dir='datasets/set14')
# }

# default_validation_config = datasets['kodak']

# corruption_types = {
#     'gaussian': gaussian_noise_config,
#     'poisson': poisson_noise_config
# }

# Train config
# ------------------------------------------------------------------------------------------

train_config = dnnlib.EasyDict(
    iteration_count=100000,
    eval_interval=1000,
    minibatch_size=4,
    run_func_name="train_pp.train",
    learning_rate=0.0003,
    ramp_down_perc=0.3,
    train_tfrecords=None,
    validation_config=None
)

# Validation run config
# ------------------------------------------------------------------------------------------
validate_config = dnnlib.EasyDict(
    run_func_name="validation.validate",
    dataset=None,
    network_snapshot=None,
    noise=gaussian_noise_config
)

# ------------------------------------------------------------------------------------------

# jhellsten quota group

def error(*print_args):
    print (*print_args)
    sys.exit(1)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ------------------------------------------------------------------------------------------
examples='''examples:

  # Train a network using the BSD300 dataset:
  python %(prog)s train --train-tfrecords=datasets/bsd300.tfrecords

  # Run a set of images through a pre-trained network:
  python %(prog)s validate --network-snapshot=results/network_final.pickle --dataset-dir=datasets/kodak
'''

if __name__ == "__main__":
    def train(args):
        if args:
            if 'long_train' in args and args.long_train:
                train_config.iteration_count = 500000
                train_config.eval_interval = 5000
                train_config.ramp_down_perc = 0.5
        else:
            print ('running with defaults in train_config')
        noise = 'gaussian'

        train_config.train_tfrecords = submit.get_path_from_template(args.tfrecords)

        print (train_config)
        dnnlib.submission.submit.submit_run(submit_config, **train_config)

    def validate(args):
        if submit_config.submit_target != dnnlib.SubmitTarget.LOCAL:
            print ('Command line overrides currently supported only in local runs for the validate subcommand')
            sys.exit(1)
        if args.dataset_dir is None:
            error('Must select dataset with --dataset-dir')
        else:
            validate_config.dataset = {
                'dataset_dir': args.dataset_dir
            }
        if args.noise not in corruption_types:
            error('Unknown noise type', args.noise)
        validate_config.noise = corruption_types[args.noise]
        if args.network_snapshot is None:
            error('Must specify trained network filename with --network-snapshot')
        validate_config.network_snapshot = args.network_snapshot
        dnnlib.submission.submit.submit_run(submit_config, **validate_config)

    def infer_stack(args):
        tmp = Image.open(args.stack)
        h,w = np.shape(tmp)

        imgs = np.zeros((tmp.n_frames, 3, h, w))
        for i in range(tmp.n_frames):
            tmp.seek(i)
            imgs[i, 0, :, :] = np.array(tmp)
            imgs[i, 1, :, :] = np.array(tmp)
            imgs[i, 2, :, :] = np.array(tmp)
        imgs = imgs.astype("float32")
        imgs = imgs / 255.0 - 0.5

        tfutil.init_tf(tf_config)
        net = util.load_snapshot(args.network)

        res = np.zeros(imgs.shape)
        for i in range(imgs.shape[0]):
            res[i,:,:,:] = util.infer_image(net, imgs[i,:,:,:])

        tmp = Image.fromarray(res[0,:,:,:].transpose([1,2,0]).astype("uint8"))
        tmp.save(args.out, format="tiff",
                 append_images=[Image.fromarray(res[i,:,:,:].transpose([1,2,0]).astype("uint8")) for i in range(1, res.shape[0])],
                 save_all=True)

    # Train by default
    parser = argparse.ArgumentParser(
        description='Train a network or run a set of images through a trained network.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--desc', default='', help='Append desc to the run descriptor string')
    parser.add_argument('--run-dir-root', help='Working dir for a training or a validation run. Will contain training and validation results.')
    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')
    parser_train = subparsers.add_parser('train', help='Train a network')

    parser_train.add_argument('--long-train', default=False, help='Train for a very long time (500k iterations or 500k*minibatch image)')
    parser_train.add_argument('tfrecords', help='Filename of the training set tfrecords file')
    parser_train.set_defaults(func=train)

    parser_validate = subparsers.add_parser('validate', help='Run a set of images through the network')
    parser_validate.add_argument('--dataset-dir', help='Load all images from a directory (*.png, *.jpg/jpeg, *.bmp)')
    parser_validate.add_argument('--network-snapshot', help='Trained network pickle')
    parser_validate.add_argument('--noise', default='gaussian', help='Type of noise corruption (one of: gaussian, poisson)')
    parser_validate.set_defaults(func=validate)

    parser_infer = subparsers.add_parser('infer', help='Run a stack through the network')
    parser_infer.add_argument('network', help='Trained network pickle')
    parser_infer.add_argument('stack', help='Image stack filename')
    parser_infer.add_argument('out', help='Output stack filename')
    parser_infer.set_defaults(func=infer_stack)

    args = parser.parse_args()
    submit_config.run_desc = desc + args.desc
    if args.run_dir_root is not None:
        submit_config.run_dir_root = args.run_dir_root
    if args.command is not None:
        args.func(args)
    else:
        # Train if no subcommand was given
        train(args)
