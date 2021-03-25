"""
Test code for Expert Network (Expert-Net)
Created on 10/13/2020
@author DreamTale
"""
import os
import tqdm
import torch
import argparse
import numpy as np
from torch import nn
from data import ImageFolder
import torch.nn.functional as F
from scipy.stats import entropy
import torchvision.utils as vutils
from torch.autograd import Variable
from trainer import ExpertNet_Trainer
from utils import get_config, get_data_loader_folder


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/mvtec_ad.yaml', help='Path to the config file.')
parser.add_argument('-i', '--input_folder', type=str, help="input image folder")
parser.add_argument('--refer_folder', type=str, default=None, help="reference image folder, usually is the input image dir")
parser.add_argument('-o', '--output_folder', type=str, help="output image folder")
parser.add_argument('-r', '--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--i2m', type=int, help="1 for i2m and 0 for m2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_detail',type=int, default=10, help="number of details to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized detail code or not")
parser.add_argument('--gpu_id', type=int, default=0, help="gpu id")

opts = parser.parse_args()

print(' ######################### Arguments: ############################')
print(' +----------------------------------------------------------------')
for arg in vars(opts):
    print(' │ {:<20} : {}'.format(arg, getattr(opts, arg)))
print(' +----------------------------------------------------------------')

torch.cuda.set_device(opts.gpu_id)

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.i2m else config['input_dim_b']
if opts.refer_folder is None:
    opts.refer_folder = os.path.join(os.path.dirname(opts.input_folder),
        os.path.basename(opts.input_folder).replace('Input', 'Impression') if opts.i2m else os.path.basename(opts.input_folder).replace('Impression', 'Input'))

# Setup model and data loader
image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
data_loader = get_data_loader_folder(opts.input_folder, 1, False, new_size=config['new_size'], crop=False)
refe_loader = get_data_loader_folder(opts.refer_folder, 1, False, new_size=config['new_size'], crop=False)

detail_dim = config['gen']['detail_dim']
trainer = ExpertNet_Trainer(config)

print('Test on GPU: ', opts.gpu_id)
state_dict = torch.load(opts.checkpoint, map_location={'cuda:{}'.format(opts.gpu_id): 'cuda:%d' % opts.gpu_id})
# state_dict = torch.load(opts.checkpoint)
trainer.gen_i2m.load_state_dict(state_dict['i2m'])
trainer.gen_m2i.load_state_dict(state_dict['m2i'])

trainer.cuda()
trainer.eval()

encode        = trainer.gen_i.encode if opts.i2m else trainer.gen_m.encode # encode function
encode_detail = trainer.gen_m.encode if opts.i2m else trainer.gen_i.encode # encode function
decode        = trainer.gen_m.decode if opts.i2m else trainer.gen_i.decode # decode function


# Start testing
detail_fixed = Variable(torch.randn(opts.num_detail, detail_dim, 1, 1).cuda())
t_bar = tqdm.tqdm(zip(data_loader, refe_loader, image_names))
for i, (images, refers, names) in enumerate(t_bar):
    t_bar.set_description('Processing {:>20}'.format(os.path.basename(names[1]))) 
    images = Variable(images.cuda())
    refers = Variable(refers.cuda())
    content, _ = encode(images)

    # Given reference 
    _, detail = encode_detail(refers)
    outputs = decode(content, detail)
    outputs = (outputs + 1) / 2.
    
    basename = os.path.basename(names[1]).split('.')[0] + '.png'
    path = os.path.join(opts.output_folder, basename)

    # Random generation for image to impression
    if opts.i2m:
        detail = detail_fixed if opts.synchronized else Variable(torch.randn(opts.num_detail, detail_dim, 1, 1).cuda())
    for j in range(opts.num_detail):
        s = detail[j].unsqueeze(0)
        outputs = decode(content, s)
        outputs = (outputs + 1) / 2.
        # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
        path = os.path.join(opts.output_folder, "rand_%02d" % j, basename)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        vutils.save_image(outputs.data, path, padding=0, normalize=True)


# python test_batch.py --config configs/mvtec_ad.yaml --input_folder /home/lyf2/results/UTAD_info/saved_images/hazelnut/images_train/trainA/ --output_folder /home/lyf2/results/UTAD_info/saved_images/hazelnut/good/reconstructioni2m/ --checkpoint /home/lyf2/checkpoints/UTAD/post_train/outputs/hazelnut/checkpoints/gen_00100000.pt --i2m 1 --output_only