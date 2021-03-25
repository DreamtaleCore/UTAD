"""
Network for Impression Imression Network (IE-Net)
Created on 3/23/2021
@author DreamTale
"""
import torch
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ie_net.model import inceptionCAE, discriminator
import torch.nn as nn
import ie_net.losses as losses
import logging
import yaml
import pandas as pd
import datetime
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IENet(nn.Module):
    def __init__(self, args, verbose=True, for_train=True):
        super().__init__()

        # path attrivutes
        self.args            = args
        self.input_directory = args.input_directory
        self.save_dir        = None
        self.log_dir         = None

        # model and data attributes
        self.architecture    = args.architecture
        self.color_mode      = args.color_mode
        self.loss            = args.loss
        self.batch_size      = args.batch_size
        self.input_size      = args.input_size

        # training attributes
        self.save_root       = args.save_root

        # build model and preprocessing variables
        self.model           = inceptionCAE.Model(self.color_mode, self.input_size, for_info=True)
        
        # verbosity
        self.verbose = verbose
        if verbose:
            print('Model\n' + '-' * 80)
            print(self.model)
    
    def load_state_dict(self, state_dict, strict=True, is_train=True):
        self.model.load_state_dict(state_dict['autoencoder'])
    
    @torch.no_grad()
    def predict(self, batch_image):
        return self.model(batch_image)
