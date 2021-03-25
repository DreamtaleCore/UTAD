#!/usr/bin/env python3
"""
Created on 10/22/2020
@Author: DreamTale
"""
import os
import sys
import logging
import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


IE_WS = '/home/lyf/ws/ZAD'
EN_WS = '/home/lyf/ws/ZAD/3rdparty/ExpertNet'


def print_now():
    # With format 2016-04-07 10:25:09
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def switch_workspace_to(s_dir):
    os.chdir(s_dir)


def find_lastest_pth(s_dir, postfix='.pth', prefix=''):
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(s_dir):
        for filename in filenames:
            if filename.endswith(postfix) and filename.startswith(prefix): 
                list_of_files[filename] = os.sep.join([dirpath, filename])
    ret_k = sorted(list_of_files.keys())[-1]
    return list_of_files[ret_k]


def main(args):
    gpu_id   = args.gpu_id
    save_dir = args.save_dir
    pwd_resume_IE = find_lastest_pth(os.path.join(args.checkpoints_path, 'IE'))
    pwd_resume_EN = find_lastest_pth(os.path.join(args.checkpoints_path, 'EN'), prefix='gen_', postfix='.pt')

    print('==> Find lastest IE pth at <{}>'.format(pwd_resume_IE))
    print('==> Find lastest EN pth at <{}>'.format(pwd_resume_EN))

    # At first, use ImpressionExtractor `test_expression` generate the impression of the test image
    switch_workspace_to(IE_WS)
    cmd = f'python generate_impression.py -c {args.configIE} ' + \
                                         '-s {} '.format(save_dir) + \
                                         '-r {} '.format(pwd_resume_IE) + \
                                         '-i {} '.format(args.input_dir) + \
                                         '-g {}'.format(gpu_id)
    print('[RUN] ', '=' * 40 + '>')
    print(print_now())
    print(cmd)
    os.system(cmd)
    # When this done, the image results are saved in the `save_dir/trainA` and `save_dir/trainB`, ends with `*.png`
    # the latent results are saved in the `save_dir/latent_codes`, ends with `*.pth`

    # Then, use the invertable ExpertNet to generate the reconstruction image with impression as input
    switch_workspace_to(EN_WS)
    cmd = f'python test_en.py --config {args.configEN} ' + \
                              '--input {} '.format(os.path.join(save_dir, 'trainA')) + \
                              '--refer_folder {} '.format(os.path.join(save_dir, 'trainB')) + \
                              '--output_folder {} '.format(os.path.join(save_dir, 'naive_impressions')) + \
                              '--checkpoint {} '.format(pwd_resume_EN) + \
                              '--a2b 1 --gpu_id {} '.format(gpu_id)
    print('[RUN] ', '=' * 40 + '>')
    print(print_now())
    print(cmd)
    os.system(cmd)
    # When this done, the image results are saved in the `save_dir/reconstructionA2B/00, 01, ...` and `save_dir/reconstructionB2A/00, 01, ...`
    
    cmd = f'python test_en.py --config {args.configEN} ' + \
                              '--input {} '.format(os.path.join(save_dir, 'trainB')) + \
                              '--refer_folder {} '.format(os.path.join(save_dir, 'trainA')) + \
                              '--output_folder {} '.format(os.path.join(save_dir, 'reconstructions')) + \
                              '--checkpoint {} '.format(pwd_resume_EN) + \
                              '--a2b 0 --gpu_id {} '.format(gpu_id)
    print('[RUN] ', '=' * 40 + '>')
    print(print_now())
    print(cmd)
    os.system(cmd)
    # When this done, the image results are saved in the `save_dir/reconstructionB2A/00, 01, ...` and `save_dir/reconstructionB2A/00, 01, ...`

    # Next, Generate the results and generate the anomaly detection score and segemntation
    switch_workspace_to(IE_WS)
    test_result_dir = os.path.join(save_dir, 'results')
    cmd = f'python test_pm.py -i {save_dir} ' + \
                             '-s {} -g {} -v'.format(test_result_dir, gpu_id)
    print('[RUN] ', '=' * 40 + '>')
    print(print_now())
    print(cmd)
    os.system(cmd)
    # When this done, the anomaly segmentation results are saved in the `save_dir/results/<type>-<file_name>.png`
    #                 the anomaly classification results are saved in the `save_dir/results/cls.txt` 
    #                                                with each line <type>-<file_name>.png 0: normal, 1: abnormal

    if not args.eval:
        print('All done.')
        return

    # Next, Evaluate the results and generate the anomaly detection score and segemntation
    switch_workspace_to(IE_WS)
    input_dir = os.path.join(save_dir, 'results')
    gt_dir    = os.path.join(args.input_dir, 'ground_truth')
    cmd       = f'python evaluate.py -i {input_dir} -g {gt_dir}'
    print('[RUN] ', '=' * 40 + '>')
    print(print_now())
    print(cmd)
    os.system(cmd)
    # When this done, the anomaly segmentation results are reported in `save_dir/results/evaluation.log`

    print(print_now())
    print('All done.')


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(description='Pytorch ZAD Testing')
    parser.add_argument('-c_ie', '--configIE', type=str, default='configs/mvtec.yaml',
                        help='Relative path to the ImpressionExtractor config file.')
    parser.add_argument('-c_en', '--configEN', type=str, default='configs/mvtec.yaml',
                        help='Relative path to the ExpertNet config file.')
    parser.add_argument('-ckpt', '--checkpoints_path', type=str, 
                        default='checkpoints/', help="pretrained checkpoints path")
    parser.add_argument('-i',   '--input_dir', type=str, 
                        default='data/to/the/test/set/', help="input test dir")
    parser.add_argument('-s',    '--save_dir', type=str, 
                        default='data/to/the/saving/dir/', help="save result dir")
    parser.add_argument('-g',    '--gpu_id', type=int, 
                        default=0, help="GPU ID for testing")
    parser.add_argument('-v', "--eval", action='store_true',
                        help='evaluate the test result')

    args = parser.parse_args()

    main(args)




