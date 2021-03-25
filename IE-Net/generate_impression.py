#!/usr/bin/env python3
"""
Created on 10/18/2020
@Author: DreamTale
"""
import sys
import os
import argparse
from pathlib import Path
import time
import yaml
import torch
import utils
import tqdm
import numpy as np
import logging
import torch.backends.cudnn as cudnn
import ie_net.ie
import ie_net.data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    # parse arguments
    model_path = args.resume
    save = args.save

    print(' ######################### Arguments: ############################')
    print(' +----------------------------------------------------------------')
    for arg in vars(args):
        print(' │ {:<20} : {}'.format(arg, getattr(args, arg)))
    print(' +----------------------------------------------------------------')
    # Load experiment setting
    config = utils.get_config(args.config)
    config = utils.merge_config(config, {'save_root': args.save})
    input_directory = config.input_directory
    if args.input_directory is not None:
        input_directory = args.input_directory
    
    cudnn.benchmark = True
    torch.cuda.set_device(args.gpu_id)

    # ============= LOAD MODEL AND PREPROCESSING CONFIGURATION ================
    utad = ie_net.ie.IENet(config, verbose=args.verbose, for_train=False)

    # load model and info
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        loc = 'cuda:{}'.format(args.gpu_id)
        checkpoint = torch.load(args.resume, map_location=loc)
        start_epoch = checkpoint['epoch']
        utad.load_state_dict(checkpoint, is_train=False)
        print("=> loaded checkpoint '{}' from epoch {}.".format(args.resume, 
                                                                checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    utad.cuda()
    utad.eval()

    sub_name = 'train' if args.for_latter_train else None

    test_loader = torch.utils.data.DataLoader(
        ie_net.data_loader.get_testset(input_directory, sub_name, input_size=config.input_size),
        batch_size=args.batch_size, shuffle=False, num_workers=config.workers
    )
    print(f'### ==> {len(test_loader)} batchs are found in test set <== ##')

    exp_name = config.exp_name
    print('### ==> [RNU]', exp_name, '<== ##')

    model_dir_name = os.path.basename(str(Path(model_path).parent))
    t_bar = tqdm.tqdm(test_loader)
    t_bar.set_description('Processing')
    for (imgs_test_input, labels, filenames) in t_bar:
        # predict on test images
        imgs_test_pred, latents, means, log_vars = utad.predict(imgs_test_input.cuda())
        
        # ====================== SAVE TEST RESULTS =========================

        # create directory to save test results
        if args.save is None:
            save_dir = os.path.join(
                'results',
                os.path.basename(input_directory),
                config.architecture,
                config.loss,
                model_dir_name,
                "test",
            )
        else:
            save_dir = args.save

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # save test result
        if save:
            utils.save_intermediate_images(imgs_test_input, imgs_test_pred, filenames, save_dir, is_for_train=True)
            utils.save_intermediate_latents({'latent': latents, 'mean': means, 'log_var': log_vars}, filenames, save_dir)


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(description='Pytorch UTAD Generate Impression')
    parser.add_argument('-c', '--config', type=str, default='configs/mvtec.yaml',
                        help='Path to the config file.')
    parser.add_argument(
        "-r", "--resume", type=str, required=True, help="path to saved model"
    )
    parser.add_argument('-i', '--input_directory', type=str, 
                        default=None, help="input training data path")
    parser.add_argument(
        "-s", "--save", type=str, default='/home/lyf2/results/utad_info', help="save results images",
    )
    parser.add_argument(
        '-g', '--gpu_id', type=int, default=0, help="gpu id"
    )
    parser.add_argument(
        '-b', '--batch_size', default=4, type=int, metavar='N', help='number of batch size'
    )
    parser.add_argument(
        '--verbose', action='store_true', help='show the network architecture or not'
    )
    parser.add_argument(
        '--for_latter_train', action='store_true', help='orginaze the output images and inputs to dir for latter train'
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(e)
        exit(0)

# Examples of command to initiate testing
# python3 test.py -c configs/inception.yaml -r /home/lyf2/checkpoints/utad/test_AE/utad_inceptionCAE/rgb-l2+edge/inceptionCAE_b32_e399.pth -m l2 --fast_ae
# python3 test.py -c configs/inception.yaml -r /home/lyf2/checkpoints/utad/test_AE/utad_inceptionCAE/rgb-l2+edge/inceptionCAE_b32_e399.pth -m l2 --verbose
# Example of command to initiate testing with saving intermediate images
# python3 test.py -c configs/inception.yaml -r /home/lyf2/checkpoints/utad/test_AE/utad_inceptionCAE/rgb-l2+edge/inceptionCAE_b32_e399.pth -m l2 --for_latter_train --fast_ae
# python3 test_expression.py -c configs/inception.yaml -r /home/lyf2/checkpoints/utad/test_AE/utad_inceptionCAE/rgb-l2+edge/inceptionCAE_b32_e399.pth -m l2  --fast_ae -s /home/lyf2/results/utad_info/saved_images/hazelnut/