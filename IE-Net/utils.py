import os
import tqdm
import yaml
import torch
import easydict
import numpy as np
import torchvision
from torch import nn
from datetime import datetime
from torch.autograd import Variable
import ie_net.postprocessing as pp
from matplotlib import pyplot as plt
from torchvision.models import vgg11, vgg19
from torchvision.transforms import transforms
from torchvision.datasets.folder import is_image_file, default_loader


def print_now():
    # With format 2016-04-07 10:25:09
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_config(config):
    with open(config, 'r') as stream:
        return easydict.EasyDict(yaml.load(stream, Loader=yaml.FullLoader))


def merge_config(config_src, config_addon):
    if type(config_addon) is dict:
        for k, v in config_addon.items():
            config_src[k] = v
    else:
        try:
            for arg in vars(config_addon):
                config_src[arg] = getattr(config_addon, arg)
        except Exception as e:
            print(f'{type(config_addon)} is not supported!')
    return config_src


def calculate_largest_areas(resmaps, thresholds):

    # initialize largest areas to an empty list
    largest_areas = []

    # initialize progress bar
    t_bar = tqdm.tqdm(range(len(thresholds)))
    t_bar.set_description('Processing')

    for index in t_bar:
        threshold = thresholds[index]
        # segment (threshold) residual maps
        resmaps_th = resmaps > threshold

        # compute labeled connected components
        _, areas_th = pp.label_images(resmaps_th)

        # retieve largest area of all resmaps for current threshold
        areas_th_total = [item for sublist in areas_th for item in sublist]
        largest_area = np.amax(np.array(areas_th_total))
        largest_areas.append(largest_area)

    return largest_areas


def get_true_classes(filenames):
    # retrieve ground truth
    y_true = [1 if "good" not in filename.split("/") else 0 for filename in filenames]
    return y_true


def get_true_classe(filename):
    # retrieve ground truth
    y_true = 1 if "good" not in filename else 0
    return y_true


def is_defective(areas, min_area):
    """Decides if image is defective given the areas of its connected components"""
    areas = np.array(areas)
    if areas[areas >= min_area].shape[0] > 0:
        return 1
    return 0


def predict_classes(resmaps, min_area, threshold):
    # threshold residual maps with the given threshold
    resmaps_th = resmaps > threshold
    # compute connected components
    _, areas_all = pp.label_images(resmaps_th)
    # Decides if images are defective given the areas of their connected components
    y_pred = [is_defective(areas, min_area) for areas in areas_all]
    return y_pred


def save_segmented_images(resmaps, threshold, filenames, save_dir):
    # threshold residual maps with the given threshold
    resmaps_th = resmaps > threshold
    # create directory to save segmented resmaps
    seg_dir = os.path.join(save_dir, "segmentation")
    if not os.path.isdir(seg_dir):
        os.makedirs(seg_dir)
    # save segmented resmaps
    for i, resmap_th in enumerate(resmaps_th):
        fname = os.path.basename(os.path.dirname(filenames[i])) + '-' +\
                os.path.basename(filenames[i]).split('.')[0] + "_seg.png"
        fpath = os.path.join(seg_dir, fname)
        plt.imsave(fpath, resmap_th, cmap="gray")
    return


def save_intermediate_images(inputs, predictions, filenames, save_dir, is_for_train=False):    
    if is_for_train:
        image_a_dir = os.path.join(save_dir, 'Input')
        image_b_dir = os.path.join(save_dir, 'Impression')
        if not os.path.isdir(image_a_dir):
            os.makedirs(image_a_dir)
        if not os.path.isdir(image_b_dir):
            os.makedirs(image_b_dir)
        
        for idx in range(inputs.shape[0]):
            fname = os.path.basename(os.path.dirname(filenames[idx])) + '-' +\
                    os.path.basename(filenames[idx]).split('.')[0] + ".png"
            grid_a = torchvision.utils.make_grid(inputs[idx].cpu(),      padding=0)
            grid_b = torchvision.utils.make_grid(predictions[idx].cpu(), padding=0)
            torchvision.utils.save_image(grid_a, os.path.join(image_a_dir, fname))
            torchvision.utils.save_image(grid_b, os.path.join(image_b_dir, fname))
    else:
        image_dir = os.path.join(save_dir, 'images')
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        for idx in range(inputs.shape[0]):
            fname = os.path.basename(os.path.dirname(filenames[idx])) + '-' +\
                    os.path.basename(filenames[idx]).split('.')[0] + ".png"
            grid = torchvision.utils.make_grid([inputs[idx].cpu(), predictions[idx].cpu()])
            torchvision.utils.save_image(grid, os.path.join(image_dir, fname))


def save_intermediate_latents(latent_dict, filenames, save_dir):
    latent_dir = os.path.join(save_dir, 'latent_codes')
    if not os.path.isdir(latent_dir):
        os.makedirs(latent_dir)
    
    for i, (latent, mean, log_var) in enumerate(zip(latent_dict['latent'], latent_dict['mean'], latent_dict['log_var'])):
        fname = os.path.basename(os.path.dirname(filenames[i])) + '-' +\
                os.path.basename(filenames[i]).split('.')[0] + ".pth"
        torch.save({'latent': latent, 'mean': mean, 'log_var': log_var}, os.path.join(latent_dir, fname))
        pass



class LoadMetaAnormalyFiles(list):
    def __init__(self, in_dir, origin_dir=None):
        super().__init__()
        self.root = in_dir
        self.origin_dir      = origin_dir
        self.input_name      = 'trainA'
        self.impression_name = 'trainB'
        self.latent_name     = 'latent_codes'
        self.naive_impression = 'naive_impressions'
        self.reconstruction   = 'reconstructions'

        self.file_names = self._get_file_names(os.path.join(self.root, self.input_name))
        self.modality_names = self._get_multimodality_names(os.path.join(self.root, self.naive_impression))

    def _get_file_names(self, s_dir):
        names = [x.split('.')[0] for x in os.listdir(s_dir) if is_image_file(x)]
        return names
    
    def _get_multimodality_names(self, s_dir):
        names = [x for x in os.listdir(s_dir) if os.path.isdir(os.path.join(s_dir, x))]
        names.sort()
        return names
    
    def __len__(self):
        return len(self.file_names)

    def _check_img(self, im_in, im_ref):
        h, w = im_ref.height, im_ref.width
        return im_in.resize((w, h))
    
    def _img_loader(self, pwd, im_ref=None):
        img = default_loader(pwd)
        if im_ref is None:
            return img
        else:
            return self._check_img(img, im_ref)
    
    def __getitem__(self, index):
        fname = self.file_names[index]
        if self.origin_dir is not None:
            img_input = self._img_loader(os.path.join(self.origin_dir, fname.split('-')[0], fname.split('-')[1] + '.png'))
        else:
            img_input = self._img_loader(os.path.join(self.root, self.input_name, fname + '.png'))
        # import pdb; pdb.set_trace()
        img_impre = self._img_loader(os.path.join(self.root, self.impression_name, fname + '.png'), img_input)
        # latent_co = torch.load(os.path.join(self.root, self.latent_name, fname + '.pth'), map_location='cuda:0')
        latent_co = None
        fast_impr = [self._img_loader(os.path.join(self.root, self.naive_impression, x, fname + '.png'), img_input)
                        for x in self.modality_names]
        expert_re = [self._img_loader(os.path.join(self.root, self.reconstruction, x, fname + '.png'), img_input)
                        for x in self.modality_names]
        label_id  = 0 if 'good' in fname else 1 
        return easydict.EasyDict({
            'input':             img_input,
            'file_name':         fname,
            'label_id':          label_id,
            'impression':        img_impre,
            'latent':            latent_co,
            'naive_impressions': fast_impr,
            'reconstructions':   expert_re
        })


class Vgg11EncoderMS(nn.Module):
    """Vgg encoder wiht multi-scales"""

    def __init__(self, pretrained):
        super().__init__()
        features = list(vgg11(pretrained=pretrained).features)
        self.backbone = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1',
                       'conv2_1',
                       'conv3_1', 'conv3_2',
                       'conv4_1', 'conv4_2',
                       'conv5_1', 'conv5_2']
        idx = 0
        for ii, model in enumerate(self.backbone):
            x = model(x)
            if ii in {0, 3, 6, 8, 11, 13, 16, 18}:
                result_dict[layer_names[idx]] = x
                idx += 1

        # return {'conv5_1': result_dict['conv5_1'], 'conv5_2': result_dict['conv5_2'],
        #         'conv5_3': result_dict['conv5_3'], 'conv5_4': result_dict['conv5_4']}
        return result_dict


class Vgg19EncoderMS(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        features = list(vgg19(pretrained=pretrained).features)
        self.backbone = nn.ModuleList(features)

    def forward(self, x):
        # here we assume x is normalized in [-1, 1]
        result_dict = {}
        layer_names = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        idx = 0
        for ii, model in enumerate(self.backbone):
            x = model(x)
            if ii in {0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34}:
                result_dict[layer_names[idx]] = x
                idx += 1

        # return {'conv5_1': result_dict['conv5_1'], 'conv5_2': result_dict['conv5_2'],
        #         'conv5_3': result_dict['conv5_3'], 'conv5_4': result_dict['conv5_4']}
        return result_dict


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # subtract mean
    return batch


img2tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])