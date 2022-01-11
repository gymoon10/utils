from __future__ import print_function
import argparse
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from batch_transformers import *
import matplotlib.pyplot as pp
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from data import get_training_set, get_test_set, get_validation_set
#from data import get_test_set
from functools import reduce
import numpy as np

#from scipy.misc import imsave
# import scipy.io as sio
import time
import cv2
import math
import pdb
# from sec_CRMEF_model import input4_AE4
# from Common_Residual_MEF_Decomposition_Test import CRMEF_Decomposition_Test
from testblock0223 import network1
from loss import IOU_calc

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=21, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
#paser.add_argument('--data_dir', type=str, default='./Vid4')
parser.add_argument('--epoch', type=int, default=600, help='number of epochs to train for')

parser.add_argument('--data_dir', type=str, default='./test')
parser.add_argument('--test_dir', type=str, default='./vis_test')
parser.add_argument('--fused_dir', type=str, default='./fused')
parser.add_argument('--Common_dir', type=str, default='./Common')
parser.add_argument('--Residual_dir', type=str, default='./Residual')
parser.add_argument('--Common_bst_dir', type=str, default='./Common_bst')

parser.add_argument('--data_dir10', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/test/bin/1')
parser.add_argument('--data_dir20', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/test/bin/1')
parser.add_argument('--data_dir30', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/test/bin/1')
parser.add_argument('--data_dir40', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/test/bin/1')
parser.add_argument('--data_dir50', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/test/rgb/1')
parser.add_argument('--data_dir60', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/test/rgb/1')
parser.add_argument('--data_dir70', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/test/rgb/1')
# parser.add_argument('--data_dir80', type=str, default='C:/Users/EunjiRyu/Desktop/Plant/test/shape/4')


parser.add_argument('--file_list', type=str, default='epoch600.txt')
parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=4)
parser.add_argument('--model_type', type=str, default='Common_Residual_MEF')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--output', default='Result/', help='Location to save checkpoint models')
parser.add_argument('--init_model', default='init_weight/2x_DESKTOP-KDSK8FLMK_Common_Residual_MEFMK_epoch_299_lr(5).pth', help='first model directory')
parser.add_argument('--sec_model', default='sec_weight/2x_DESKTOP-LNOFA51MK_Common_Residual_MEFMK_epoch_219_lr(5).pth', help='second model directory')
parser.add_argument('--third_model', default='third_weight/2x_DESKTOP-LNOFA51MK_Common_Residual_MEFMK_epoch_239_lr(5).pth', help='third model directory')
parser.add_argument('--fourth_model', default='fourth_weight/2x_DESKTOP-LNOFA51MK_Common_Residual_MEFMK_epoch_159_lr(5).pth', help='fourth model directory')

parser.add_argument('--model', default='', help='mef pretrained base model')

opt = parser.parse_args()
gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')


test_set = get_test_set(opt.data_dir10, opt.data_dir20, opt.data_dir30, opt.data_dir40, opt.data_dir50, opt.data_dir60, opt.data_dir70)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model ', opt.model_type)
if opt.model_type == 'Common_Residual_MEF':
    init_model = network1()
    # sec_model = network2()
    # third_model = network3()
    # fourth_model = network4()

# if cuda:
    # init_model = torch.nn.DataParallel(init_model, device_ids=gpus_list)
    # sec_model = torch.nn.DataParallel(sec_model, device_ids=gpus_list)
    # third_model = torch.nn.DataParallel(third_model, device_ids=gpus_list)
    # fourth_model = torch.nn.DataParallel(fourth_model, device_ids=gpus_list)
    # model = torch.nn.DataParallel(model, device_ids=gpus_list)

init_model.load_state_dict(torch.load(opt.init_model, map_location=lambda storage, loc: storage))
# sec_model.load_state_dict(torch.load(opt.sec_model, map_location=lambda storage, loc: storage))
# third_model.load_state_dict(torch.load(opt.third_model, map_location=lambda storage, loc: storage))
# fourth_model.load_state_dict(torch.load(opt.fourth_model, map_location=lambda storage, loc: storage))

# model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained MEF model is loaded.')

if cuda:
    init_model = init_model.cuda(gpus_list[0])
    # sec_model = sec_model.cuda(gpus_list[0])
    # third_model = third_model.cuda(gpus_list[0])
    # fourth_model = fourth_model.cuda(gpus_list[0])
    # model = model.cuda(gpus_list[0])

def eval():
    init_model.eval()
    count=1
    EPS = 1e-8
    avg_psnr_predicted = 0.0
    L1 = 0
    MSE = 0
    PSNR = 0
    PSNR2 = 0
    SSIM1=0
    LOSS1=0
    LOSS2=0

    for batch in testing_data_loader:
        target10,target20,target30,target40,target50,target60, target70 = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        #target10, target20, target30= batch[0], batch[1], batch[2]

        # target10 = target10.unsqueeze(0)
        target10 = target10[:, :, :, :] #[2,1,0]
        target20 = target20[:, :, :, :]
        target30 = target30[:, :, :, :]
        target40 = target40[:, :, :, :]
        target50 = target50[:, [2,1,0], :, :]
        target60 = target60[:, [2,1,0], :, :]
        target70 = target70[:, :, :, :]
        # target80 = target80[:, :, :, :]

        target10 = target10.cuda(gpus_list[0])
        target20 = target20.cuda(gpus_list[0])
        target30 = target30.cuda(gpus_list[0])
        target40 = target40.cuda(gpus_list[0])
        target50 = target50.cuda(gpus_list[0])
        target60 = target60.cuda(gpus_list[0])
        target70 = target70.cuda(gpus_list[0])
        # target80 = target80.cuda(gpus_list[0])

        t0 = time.time()

        with torch.no_grad():
            shape5 = init_model(target50)

        shape5 = shape5*1.2

        criterion_L1 = nn.L1Loss()
        criterion_MSE = torch.nn.MSELoss()
        criterion_SSIM = SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=True)
        criterion_IoU = IOU_calc()
        t1 = time.time()

        # LOSS1 = criterion_L1(shape5,target40)
        # LOSS2 += LOSS1

        LOSS1 = criterion_IoU(shape5, target40)
        LOSS2 += LOSS1

        # print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
        print(LOSS1)
        print(LOSS2/len(testing_data_loader))
        print(len(testing_data_loader))
        save_img((shape5).cpu(), str(count), True)
        count += 1

def save_img_vis_test(img1, img2, img_name, pred_flag):
    save_dir = os.path.join(opt.output, opt.data_dir, opt.test_dir, os.path.splitext(opt.file_list)[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pred_flag:
        save_fn1 = save_dir + '/' + img_name + '_1_' + opt.model_type + '_' + str(opt.epoch) + '_res_fused.png'
        save_fn2 = save_dir + '/' + img_name + '_2_' + opt.model_type + '_' + str(opt.epoch) + '_com_fused.png'
    else:
        save_fn1 = save_dir + '/' + img_name + '_1_' + opt.model_type + '_' + str(opt.epoch) + '_res_fused.png'
        save_fn2 = save_dir + '/' + img_name + '_2_' + opt.model_type + '_' + str(opt.epoch) + '_com_fused.png'

    save_image(img1[:, :, :, :], save_fn1)
    save_image(img2[:, :, :, :], save_fn2)



def save_img(img, img_name, pred_flag):

    save_dir = os.path.join(opt.output, opt.data_dir, os.path.splitext(opt.file_list)[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pred_flag:
        save_fn = save_dir +'/'+ img_name+'_'+opt.model_type+'_'+str(opt.epoch)+'_Residual_mef.png'
    else:
        save_fn = save_dir +'/'+ img_name+'.png'

    save_image(img[:, :, :, :], save_fn)
    
def rgb2ycbcr(rgb):

    Y = (16 + 65.481*rgb[:, 0, :, :] + 128.553*rgb[:, 1, :, :] + 24.966*rgb[:, 2, :, :]).unsqueeze(1)
    cb = (128 - 37.797*rgb[:, 0, :, :] - 74.203*rgb[:, 1, :, :] + 112*rgb[:, 2, :, :]).unsqueeze(1)
    cr = (128 + 112*rgb[:, 0, :, :] - 93.786*rgb[:, 1, :, :] - 18.214*rgb[:, 2, :, :]).unsqueeze(1)
    Ycbcr = torch.cat((Y, cb, cr), 1)
    return Ycbcr


def save_img_multi_fused_patch(img1, img2, img3, img4, img_name, pred_flag):
    save_dir = os.path.join(opt.output, opt.data_dir, opt.fused_dir, os.path.splitext(opt.file_list)[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pred_flag:
        save_fn1 = save_dir + '/' + img_name + '_1_'+opt.model_type+'_'+str(opt.epoch)+'_Fused.png'
        save_fn2 = save_dir + '/' + img_name + '_2_' + opt.model_type + '_' + str(opt.epoch) + '_Fused.png'
        save_fn3 = save_dir + '/' + img_name + '_3_' + opt.model_type + '_' + str(opt.epoch) + '_Fused.png'
        save_fn4 = save_dir + '/' + img_name + '_4_' + opt.model_type + '_' + str(opt.epoch) + '_Fused.png'
    else:
        save_fn1 = save_dir + '/' + img_name + '_1_' + opt.model_type + '_' + str(opt.epoch) + '_Fused.png'
        save_fn2 = save_dir + '/' + img_name + '_2_' + opt.model_type + '_' + str(opt.epoch) + '_Fused.png'
        save_fn3 = save_dir + '/' + img_name + '_3_' + opt.model_type + '_' + str(opt.epoch) + '_Fused.png'
        save_fn4 = save_dir + '/' + img_name + '_4_' + opt.model_type + '_' + str(opt.epoch) + '_Fused.png'

    save_image(img1[:, :, :, :], save_fn1)
    save_image(img2[:, :, :, :], save_fn2)
    save_image(img3[:, :, :, :], save_fn3)
    save_image(img4[:, :, :, :], save_fn4)


def save_img_multi_common_patch(img1, img2, img3, img4, img_name, pred_flag):
    save_dir = os.path.join(opt.output, opt.data_dir, opt.Common_dir, os.path.splitext(opt.file_list)[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pred_flag:
        save_fn1 = save_dir + '/' + img_name + '_1_' + opt.model_type + '_' + str(opt.epoch) + '_Common.png'
        save_fn2 = save_dir + '/' + img_name + '_2_' + opt.model_type + '_' + str(opt.epoch) + '_Common.png'
        save_fn3 = save_dir + '/' + img_name + '_3_' + opt.model_type + '_' + str(opt.epoch) + '_Common.png'
        save_fn4 = save_dir + '/' + img_name + '_4_' + opt.model_type + '_' + str(opt.epoch) + '_Common.png'
    else:
        save_fn1 = save_dir + '/' + img_name + '_1_' + opt.model_type + '_' + str(opt.epoch) + '_Common.png'
        save_fn2 = save_dir + '/' + img_name + '_2_' + opt.model_type + '_' + str(opt.epoch) + '_Common.png'
        save_fn3 = save_dir + '/' + img_name + '_3_' + opt.model_type + '_' + str(opt.epoch) + '_Common.png'
        save_fn4 = save_dir + '/' + img_name + '_4_' + opt.model_type + '_' + str(opt.epoch) + '_Common.png'

    save_image(img1[:, :, :, :], save_fn1)
    save_image(img2[:, :, :, :], save_fn2)
    save_image(img3[:, :, :, :], save_fn3)
    save_image(img4[:, :, :, :], save_fn4)

def save_img_multi_common_boost_patch(img1, img2, img3, img4, img_name, pred_flag):
    save_dir = os.path.join(opt.output, opt.data_dir, opt.Common_bst_dir, os.path.splitext(opt.file_list)[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pred_flag:
        save_fn1 = save_dir + '/' + img_name + '_1_' + opt.model_type + '_' + str(opt.epoch) + '_Common_Boosted.png'
        save_fn2 = save_dir + '/' + img_name + '_2_' + opt.model_type + '_' + str(opt.epoch) + '_Common_Boosted.png'
        save_fn3 = save_dir + '/' + img_name + '_3_' + opt.model_type + '_' + str(opt.epoch) + '_Common_Boosted.png'
        save_fn4 = save_dir + '/' + img_name + '_4_' + opt.model_type + '_' + str(opt.epoch) + '_Common_Boosted.png'
    else:
        save_fn1 = save_dir + '/' + img_name + '_1_' + opt.model_type + '_' + str(opt.epoch) + '_Common_Boosted.png'
        save_fn2 = save_dir + '/' + img_name + '_2_' + opt.model_type + '_' + str(opt.epoch) + '_Common_Boosted.png'
        save_fn3 = save_dir + '/' + img_name + '_3_' + opt.model_type + '_' + str(opt.epoch) + '_Common_Boosted.png'
        save_fn4 = save_dir + '/' + img_name + '_4_' + opt.model_type + '_' + str(opt.epoch) + '_Common_Boosted.png'

    bst_cst = torch.ones_like(img1)
    # print(bst_cst)
    bst_cst = bst_cst * 0.2
    bst_mul = 4
    img1_bst = img1*bst_mul+bst_cst
    img2_bst = img2*bst_mul+bst_cst
    img3_bst = img3*bst_mul+bst_cst
    img4_bst = img4*bst_mul+bst_cst

    save_image(img1_bst[:, :, :, :], save_fn1)
    save_image(img2_bst[:, :, :, :], save_fn2)
    save_image(img3_bst[:, :, :, :], save_fn3)
    save_image(img4_bst[:, :, :, :], save_fn4)



def save_img_multi_residual_patch(img1, img2, img3, img4, img_name, pred_flag):
    save_dir = os.path.join(opt.output, opt.data_dir, opt.Residual_dir, os.path.splitext(opt.file_list)[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pred_flag:
        save_fn1 = save_dir + '/' + img_name + '_1_' + opt.model_type + '_' + str(opt.epoch) + '_Residual.png'
        save_fn2 = save_dir + '/' + img_name + '_2_' + opt.model_type + '_' + str(opt.epoch) + '_Residual.png'
        save_fn3 = save_dir + '/' + img_name + '_3_' + opt.model_type + '_' + str(opt.epoch) + '_Residual.png'
        save_fn4 = save_dir + '/' + img_name + '_4_' + opt.model_type + '_' + str(opt.epoch) + '_Residual.png'
    else:
        save_fn1 = save_dir + '/' + img_name + '_1_' + opt.model_type + '_' + str(opt.epoch) + '_Residual.png'
        save_fn2 = save_dir + '/' + img_name + '_2_' + opt.model_type + '_' + str(opt.epoch) + '_Residual.png'
        save_fn3 = save_dir + '/' + img_name + '_3_' + opt.model_type + '_' + str(opt.epoch) + '_Residual.png'
        save_fn4 = save_dir + '/' + img_name + '_4_' + opt.model_type + '_' + str(opt.epoch) + '_Residual.png'

    save_image(img1[:, :, :, :], save_fn1)
    save_image(img2[:, :, :, :], save_fn2)
    save_image(img3[:, :, :, :], save_fn3)
    save_image(img4[:, :, :, :], save_fn4)




def save_img_multi(img1, img2, img3, img4, img_name, pred_flag):
    save_img1 = img1.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    save_img2 = img2.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    save_img3 = img3.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    save_img4 = img4.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

    # save img
    save_dir=os.path.join(opt.output, opt.data_dir, os.path.splitext(opt.file_list)[0]+'_'+str(opt.upscale_factor)+'x')
    # save_dir = os.path.join(opt.output, opt.data_dir)#, os.path.splitext(opt.file_list)[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pred_flag:
        save_fn1 = save_dir + '/' + 'm0' + '/' + img_name + '_' + opt.model_type + '_' + str(opt.epoch) + '_m0.png'
        save_fn2 = save_dir + '/' + 'm1' + '/' + img_name + '_' + opt.model_type + '_' + str(opt.epoch) + '_m1.png'
        save_fn3 = save_dir + '/' + 'm2' + '/' + img_name + '_' + opt.model_type + '_' + str(opt.epoch) + '_m2.png'
        save_fn4 = save_dir + '/' + 'm3' + '/' + img_name + '_' + opt.model_type + '_' + str(opt.epoch) + '_m3.png'
    else:
        save_fn = save_dir + '/' + img_name + '.png'
    # cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(save_fn1, save_img1 * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(save_fn2, save_img2 * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(save_fn3, save_img3 * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(save_fn4, save_img4 * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def save_img_multi_Y(img1, img2, img3, img4, img_name, pred_flag):
    save_dir = os.path.join(opt.output, opt.data_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pred_flag:
        save_fn1 = save_dir + '/' + 's0_mask' + '/' + img_name + '_' + opt.model_type + '_' + str(opt.epoch) + 's0_mask.png'
        save_fn2 = save_dir + '/' + 's1_mask' + '/' + img_name + '_' + opt.model_type + '_' + str(opt.epoch) + 's1_mask.png'
        save_fn3 = save_dir + '/' + 's2_mask' + '/' + img_name + '_' + opt.model_type + '_' + str(opt.epoch) + 's2_mask.png'
        save_fn4 = save_dir + '/' + 's3_mask' + '/' + img_name + '_' + opt.model_type + '_' + str(opt.epoch) + 's3_mask.png'
    else:
        save_fn1 = save_dir + '/' + 's0_mask' + '/' + img_name + '_' + opt.model_type + '_' + str(
            opt.epoch) + 's0_mask.png'
        save_fn2 = save_dir + '/' + 's1_mask' + '/' + img_name + '_' + opt.model_type + '_' + str(
            opt.epoch) + 's1_mask.png'
        save_fn3 = save_dir + '/' + 's2_mask' + '/' + img_name + '_' + opt.model_type + '_' + str(
            opt.epoch) + 's2_mask.png'
        save_fn4 = save_dir + '/' + 's3_mask' + '/' + img_name + '_' + opt.model_type + '_' + str(
            opt.epoch) + 's3_mask.png'

    save_image(img1[:, :, :, :], save_fn1)
    save_image(img2[:, :, :, :], save_fn2)
    save_image(img3[:, :, :, :], save_fn3)
    save_image(img4[:, :, :, :], save_fn4)



def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    gt = gt[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
    
def chop_forward(x, neigbor, flow, model, scale, shave=8, min_size=2000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        [x[:, :, 0:h_size, 0:w_size], [j[:, :, 0:h_size, 0:w_size] for j in neigbor], [j[:, :, 0:h_size, 0:w_size] for j in flow]],
        [x[:, :, 0:h_size, (w - w_size):w], [j[:, :, 0:h_size, (w - w_size):w] for j in neigbor], [j[:, :, 0:h_size, (w - w_size):w] for j in flow]],
        [x[:, :, (h - h_size):h, 0:w_size], [j[:, :, (h - h_size):h, 0:w_size] for j in neigbor], [j[:, :, (h - h_size):h, 0:w_size] for j in flow]],
        [x[:, :, (h - h_size):h, (w - w_size):w], [j[:, :, (h - h_size):h, (w - w_size):w] for j in neigbor], [j[:, :, (h - h_size):h, (w - w_size):w] for j in flow]]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = inputlist[i]#torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                output_batch = model(input_batch[0], input_batch[1], input_batch[2])
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch[0], patch[1], patch[2], model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


##Eval Start!!!!
if __name__ == '__main__':

    eval()
