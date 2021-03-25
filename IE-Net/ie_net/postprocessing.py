"""
Created on 10/19/2020
@Author: Dreatale
Inspired by https://github.com/AdneneBoumessouer/MVTec-Anomaly-Detection/blob/master/processing/postprocessing.py
"""
import os
import cv2
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.util import img_as_ubyte
import logging
import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Segmentation Parameters
# float + SSIM
THRESH_MIN_FLOAT_SSIM = 0.10
THRESH_STEP_FLOAT_SSIM = 0.002
# float + L2
THRESH_MIN_FLOAT_L2 = 0.005
THRESH_STEP_FLOAT_L2 = 0.0005
# uint8 + SSIM
THRESH_MIN_UINT8_SSIM = 20
THRESH_STEP_UINT8_SSIM = 1
# uint8 + L2 (generally uneffective combination)
THRESH_MIN_UINT8_L2 = 5
THRESH_STEP_UINT8_L2 = 1


class PassImages(object):
    def __init__(
        self,
        img_input,
        img_pred,
        fast_impression,
        expert_reconstr,
        latent,
        method,
        dtype="float64",
        filename=None,
    ):
        assert dtype in ["float64", "uint8"]
        assert method in ["l2", "ssim"]
        self.method = method
        self.dtype = dtype
        self.filename = filename

        # compute resmaps
        self.img_input  = np.asarray(img_input)
        self.img_pred   = np.asarray(img_pred)
        self.imgs_fast  = [np.asarray(x) for x in fast_impression]
        self.imgs_exper = [np.asarray(x) for x in expert_reconstr]
        self.latent     = latent 

        # self.score, self.resmap = calculate_residual_map(
        #     self.img_input, self.img_pred, self.imgs_fast, self.imgs_exper, method, dtype
        # )

        # # compute maximal threshold based on resmap
        # self.thresh_max = np.amax(self.resmap)

        # # set parameters for future segmentation of resmap
        # if dtype == "float64":
        #     self.vmin_resmap = 0.0
        #     self.vmax_resmap = 1.0
        #     if method in ["ssim", "mssim"]:
        #         self.thresh_min = THRESH_MIN_FLOAT_SSIM
        #         self.thresh_step = THRESH_STEP_FLOAT_SSIM
        #     elif method == "l2":
        #         self.thresh_min = THRESH_MIN_FLOAT_L2
        #         self.thresh_step = THRESH_STEP_FLOAT_L2

        # elif dtype == "uint8":
        #     self.vmin_resmap = 0
        #     self.vmax_resmap = 255
        #     if method in ["ssim", "mssim"]:
        #         self.thresh_min = THRESH_MIN_UINT8_SSIM
        #         self.thresh_step = THRESH_STEP_UINT8_SSIM
        #     elif method == "l2":
        #         self.thresh_min = THRESH_MIN_UINT8_L2
        #         self.thresh_step = THRESH_STEP_UINT8_L2
        
    def generate_inspection_plots(self, save_dir=None):
        self.plot_input_pred_resmap(save_dir=save_dir)
    
    ### plottings methods for inspection

    def plot_input_pred_resmap(self, save_dir=None):
        fig, axarr = plt.subplots(3, 1)
        fig.set_size_inches((4, 9))

        axarr[0].imshow(
            self.img_input, cmap=self.cmap, vmin=0, vmax=255,
        )
        axarr[0].set_title("input")
        axarr[0].set_axis_off()
        # fig.colorbar(im00, ax=axarr[0])

        axarr[1].imshow(
            self.imgs_pred, cmap=self.cmap, vmin=0, vmax=255
        )
        axarr[1].set_title("pred")
        axarr[1].set_axis_off()
        # fig.colorbar(im10, ax=axarr[1])

        im20 = axarr[2].imshow(
            self.resmap,
            cmap="inferno",
            vmin=self.vmin_resmap,
            vmax=self.vmax_resmap,
        )
        axarr[2].set_title(
            "resmap_"
            + self.method
            + "_"
            + self.dtype
            + "\n{}_".format(self.method)
            + f"score = {self.score:.2E}"
        )
        axarr[2].set_axis_off()
        fig.colorbar(im20, ax=axarr[2])

        if save_dir is not None:
            plot_name = get_plot_name(self.filename, suffix="inspection")
            fig.savefig(os.path.join(save_dir, plot_name))
            plt.close(fig=fig)
        return
    
    def __call__(self):
        """
        Return the anomaly map, anomaly score and anomaly classification ID
        """
        im_i = self.img_input
        im_p = self.img_pred

        im_f = normalize_pix2pix_results(self.imgs_fast, im_i)
        im_e = normalize_pix2pix_results(self.imgs_exper, im_p)
        embd = self.latent

        dir_save = r'F:\tmp\mvtec_all_test\mutual\cable\test\tmp_images'
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        
        save_img('im_i', im_i)
        save_img('im_p', im_p)

        for i in range(len(im_f)):
            save_img('im_f-{}'.format(i), im_f[i])
            save_img('im_e-{}'.format(i), im_e[i])

        # the difference between {input} and {impression}
        s_ip, m_ip = calculate_resmap(im_i, im_p, self.method, 'uint8')
        save_img('m_ip', m_ip, to_heatmap=True)
        # mean and var of {fast impression} from invertatible pix2pix
        im_f_mean, im_f_var = get_mean_var(im_f)
        save_img('im_f_mean', np.uint8(im_f_mean))
        save_img('im_f_var',  im_f_var, to_heatmap=True)
        # mean and var of {expert reconstuction} from invertatible pix2pix
        im_e_mean, im_e_var = get_mean_var(im_e)
        save_img('im_e_mean', np.uint8(im_e_mean))
        save_img('im_e_var',  im_e_var, to_heatmap=True)
        # the difference between {epxer reconstruction} and {impression}
        s_ep, m_ep = calculate_resmap(im_e_mean, im_p, self.method, 'uint8') 
        save_img('m_ep',  m_ep, to_heatmap=True)
        # the difference between {input} and {mean expert reconstuction}
        s_ie, m_ie = calculate_resmap(im_i, im_e_mean, self.method, 'uint8')
        save_img('m_ie', m_ie, to_heatmap=True)
        # the difference between {impression} and {mean fast impression}
        s_pf, m_pf = calculate_resmap(im_p, im_f_mean, self.method, 'uint8')
        save_img('m_pf', m_pf, to_heatmap=True)
        # the difference between {input} and {mean fast impression}
        s_if, m_if = calculate_resmap(im_i, im_f_mean, self.method, 'uint8')
        save_img('m_if', m_if, to_heatmap=True)
        # a brute force combination
        # import pdb; pdb.set_trace()
        m_final = m_ie + m_if - np.mean(im_e_var, axis=2)  - m_ep
        m_final = np.clip(m_final, 0, m_final.max())

        save_img('m_final', np.uint8(normalize_to_01(m_final)*255), to_heatmap=True)


        return None, None, None, None


def save_img(f_name='tmp', img=None, dir_save='', ord='rgb', to_heatmap=False):
    if len(img.shape) == 3 and ord == 'rgb':
        cv2.imwrite(os.path.join(dir_save, f_name + '.png'), img[..., ::-1])
    else:
        if len(img.shape) == 2 and to_heatmap:
            cv2.imwrite(os.path.join(dir_save, f_name + '.png'), cv2.applyColorMap(img, cv2.COLORMAP_JET))
        elif len(img.shape) == 3 and to_heatmap:
            cv2.imwrite(os.path.join(dir_save, f_name + '.png'), cv2.applyColorMap(np.mean(img, axis=2), cv2.COLORMAP_JET))
        else:
            cv2.imwrite(os.path.join(dir_save, f_name + '.png'), img)


def normalize_pix2pix_results(inputs, reference):
        rets = []
        reference = np.asarray(reference)
        if len(reference.shape) == 3:
            # For RGB images
            channel_max = [np.max(reference[..., i]) for i in range(3)]
            channel_min = [np.min(reference[..., i]) for i in range(3)]
            h, w = reference.shape[:2]
            for im in inputs:
                im = cv2.resize(np.asarray(im), (w, h))
                im[..., 0] = (im[..., 0] - im[..., 0].min()) / (im[..., 0].max() - im[..., 0].min()) * (channel_max[0] - channel_min[0]) + channel_min[0]
                im[..., 1] = (im[..., 1] - im[..., 1].min()) / (im[..., 1].max() - im[..., 1].min()) * (channel_max[1] - channel_min[1]) + channel_min[1]
                im[..., 2] = (im[..., 2] - im[..., 2].min()) / (im[..., 2].max() - im[..., 2].min()) * (channel_max[2] - channel_min[2]) + channel_min[2]
                rets.append(Image.fromarray(im))
        else:
            # For gray images
            channel_max = np.max(reference)
            channel_min = np.min(reference)
            h, w = reference.shape[:2]
            for im in inputs:
                im = cv2.resize(np.asarray(im), (w, h))
                im = (im - im.min()) / (im.max() - im.min()) * (channel_max[0] - channel_min[0]) + channel_min[0]
                rets.append(Image.fromarray(im))
        return rets


def normalize_to_01(im):
    if len(im.shape) == 3:
        im = sum([np.float32(im[..., x]) for x in range(3)])
    ret = (im - im.min()) / (im.max() - im.min())
    return ret


def get_plot_name(filename, suffix):
    filename_new, ext = os.path.splitext(filename)
    filename_new = "_".join(filename_new.split("/")) + "_" + suffix + ext
    return filename_new


def get_mean_var(a_list):
    mean = sum([np.float32(x) for x in a_list]) / len(a_list)
    var  = sum([((x - mean) ** 2) for x in a_list]) / len(a_list)
    return mean, var


## Functions for generating Resmaps

def to_gray(mat):
    # if RGB, transform to grayscale and reduce tensor dimension to 3
    if len(mat.shape) == 3:
        rgb_weights = [0.2989, 0.5870, 0.1140]
        ret = sum([mat[:, :, x] * rgb_weights[x] for x in range(3)])
    else:
        ret = mat
    return ret / 255.


def calculate_resmap(img_input, img_pred, method, dtype="float64"):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    img_input_gray = to_gray(img_input)
    img_pred_gray  = to_gray(img_pred)

    # calculate remaps
    if method == "l2":
        scores, resmaps = resmaps_l2(img_input_gray, img_pred_gray)
    elif method in ["ssim", "mssim"]:
        scores, resmaps = resmaps_ssim(img_input_gray, img_pred_gray)
    if dtype == "uint8":
        resmaps = img_as_ubyte(resmaps)
    return scores, resmaps


def calculate_residual_map(img_input, img_pred, imgs_fast, imgs_exper, method, dtype='float64'):
    im_mean, im_std = np.mean(imgs_fast),  np.std(imgs_fast)
    re_mean, re_std = np.mean(imgs_exper), np.std(imgs_exper)

    s0, r0 = calculate_resmap(img_input, img_pred, method, dtype)
    s1, r1 = calculate_resmap(img_input, re_mean,  method, dtype)
    s2, r2 = calculate_resmap(img_pred,  im_mean,  method, dtype)
    
    return s0 + s1 + s2, r0 + r1 + r2


def calculate_residual_score(self, res_map, res_score, latents):
    pass


def resmaps_ssim(imgs_input, imgs_pred):
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    scores = []
    for index in range(len(imgs_input)):
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
        score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )
        # resmap = np.expand_dims(resmap, axis=-1)
        resmaps[index] = 1 - resmap
        scores.append(score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    return scores, resmaps


def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2
    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps


## functions for processing resmaps


def label_images(images_th):
    """
    Segments images into images of connected components (regions).
    Returns segmented images and a list of lists, where each list 
    contains the areas of the regions of the corresponding image. 
    
    Parameters
    ----------
    images_th : array of binary images
        Thresholded residual maps.
    Returns
    -------
    images_labeled : array of labeled images
        Labeled images.
    areas_all : list of lists
        List of lists, where each list contains the areas of the regions of the corresponding image.
    """
    images_labeled = np.zeros(shape=images_th.shape)
    areas_all = []
    for i, image_th in enumerate(images_th):
        # close small holes with binary closing
        # bw = closing(image_th, square(3))

        # remove artifacts connected to image border
        cleared = clear_border(image_th)

        # label image regions
        image_labeled = label(cleared)

        # image_labeled = label(image_th)

        # append image
        images_labeled[i] = image_labeled

        # compute areas of anomalous regions in the current image
        regions = regionprops(image_labeled)

        if regions:
            areas = [region.area for region in regions]
            areas_all.append(areas)
        else:
            areas_all.append([0])

    return images_labeled, areas_all

