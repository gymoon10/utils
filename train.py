import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import os
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from loss import color_angular_error
from torchvision.utils import save_image, make_grid
from batch_transformers import *
import matplotlib.pyplot as pp
import socket
import time
from testblock0223 import network1
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from data import get_training_set, get_validation_set

parser = argparse.ArgumentParser(description='MK_Common_Residual_MEF_Retry_pytorch')
parser.add_argument('--upscale_factor', type=int, default=2, help='super resolution upscale factor')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')                              # default 7
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--val_batchsize', type=int, default=1, help='testing batch size')

parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--init_nEpochs', type=int, default=300, help='number of epochs to train for')

parser.add_argument('--snapshots', type=int, default=2, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=223, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./vimeo_septuplet/sequences')
parser.add_argument('--colorweight', type=int, default=0, help='colorweight')

parser.add_argument('--data_gt', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/bin/5')
parser.add_argument('--data_s1', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/bin/1') #mask1
parser.add_argument('--data_s2', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/bin/2')
parser.add_argument('--data_s3', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/bin/3')
parser.add_argument('--data_s4', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/bin/4')

parser.add_argument('--data_s5', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/rgb/1') #mask1
parser.add_argument('--data_s6', type=str, default='C:/PytorchProject/UnetSeg/segmentationcode/dataset5/rgb/1')
parser.add_argument('--data_s7', type=str, default='C:/Users/EunjiRyu/Desktop/Plant2/NewDataset/train/shape/2')
parser.add_argument('--data_s8', type=str, default='C:/Users/EunjiRyu/Desktop/Plant2/NewDataset/train/shape/3')
parser.add_argument('--data_s9', type=str, default='C:/Users/EunjiRyu/Desktop/Plant2/NewDataset/train/shape/4')

parser.add_argument('--nFrames', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=128, help='0 to use original frame size') # 160
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='MK_Common_Residual_MEF')
parser.add_argument('--residual', type=bool, default=False)

parser.add_argument('--pretrained_sr1', default='init_weight/2x_DESKTOP-LNOFA51MK_Common_Residual_MEFMK_epoch_84_lr(5).pth', help='pretrained_base_model')
parser.add_argument('--pretrained_init', type=bool, default=False)
parser.add_argument('--init_save_folder', default='init_weight/', help='first training for Curriculum Learning weight')


parser.add_argument('--save_folder', default='weight/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='MK', help='Location to save checkpoint models')
parser.add_argument('--patch_dir', default='patch_val', help='Location to save Val Patch')
parser.add_argument('--GF_dir', default='GF_vis', help='Location to save Global Feature visualized')
parser.add_argument('--LF_dir', default='LF_vis', help='Location to save Local Feature visualized')
parser.add_argument('--FF_dir', default='FF_vis', help='Location to save Local Feature visualized')

parser.add_argument('--init_model', default='init_weight/2x_DESKTOP-LNOFA51MK_Common_Residual_MEFMK_epoch_299_lr(5).pth', help='first model directory')


opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True


def initial_train(epoch):
    checkpoint_dir = opt.init_save_folder
    if not os.path.exists(checkpoint_dir + '/losslist.txt'):
        f = open(checkpoint_dir + '/losslist.txt', 'w')
        print('new text file made...')
    else:
        f = open(checkpoint_dir + '/losslist.txt', 'a')
        print('use the previous text file..."loss_list.txt"')

    ## epoch_Loss들
    epoch_loss = 0
    init_model.train()
    e0 = time.time()
    for iteration, batch in enumerate(training_data_loader, 1):
        n_batches = len(training_data_loader)
        GT = batch[0]
        E1 = batch[1]
        E2 = batch[2]
        E3 = batch[3]
        E4 = batch[4]
        E5 = batch[5]
        E6 = batch[6]

        GT = GT[:, [0], :, :]
        E1 = E1[:, [0], :, :]
        E2 = E2[:, [0], :, :]
        E3 = E3[:, [0], :, :]
        E4 = E4[:, [0], :, :]
        E5 = E5[:, [2, 1, 0], :, :]
        E6 = E6[:, [2, 1, 0], :, :]

        if cuda:
            GT = GT.cuda(gpus_list[0])
            E1 = E1.cuda(gpus_list[0])
            E2 = E2.cuda(gpus_list[0])
            E3 = E3.cuda(gpus_list[0])
            E4 = E4.cuda(gpus_list[0])
            E5 = E5.cuda(gpus_list[0]) #RGB-GT
            E6 = E6.cuda(gpus_list[0])

        init_optimizer.zero_grad()
        t0 = time.time()

        shape1 = init_model(E5)
        loss_l1 = criterion_L1(shape1, E1)
        loss = loss_l1

        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        init_optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}, || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                  len(training_data_loader),
                                                                                  loss.item(), (t1 - t0)))
        
    e1 = time.time()
    print('epoch {} done'.format(epoch))
    print("===> Epoch {} Complete: Avg. Loss: {:.8f} || Processing time: {:.4f}".format(epoch, (epoch_loss / len(training_data_loader)),
                                                                                        (e1 - e0)))

    text = "Epoch {} eq_mse: {:.8f} vis_e2e_mse: {:.8f}  ||||| Loss: {:.8f} \n".format(
        epoch, (epoch_loss / len(training_data_loader)), (epoch_loss / len(training_data_loader)),
        (epoch_loss / len(training_data_loader)))
    f.write(text)
    f.close()

def save_img_patch(img, img_name, num):
    # save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    # save img
    #save_dir=os.path.join(opt.output, opt.data_dir, os.path.splitext(opt.file_list)[0]+'_'+str(opt.upscale_factor)+'x')
    save_dir = os.path.join(opt.patch_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = save_dir +'/'+ img_name+'_patch_'+str(num)+'.png'

    save_image(img[:, :, :, :], save_fn)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def init_checkpoint(epoch):
    model_out_path = opt.init_save_folder+str(opt.upscale_factor)+'x_'+hostname+opt.model_type+opt.prefix+"_epoch_{}_lr(5).pth".format(epoch)
    torch.save(init_model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == '__main__':
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('=================> Loading Datasets from opt.data')

    train_set = get_training_set(opt.data_gt, opt.data_s1, opt.data_s2, opt.data_s3, opt.data_s4, opt.data_s5, opt.data_s6, opt.patch_size)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    # valid_set = get_validation_set(opt.datavalid_s1, opt.datavalid_s2, opt.datavalid_s3, opt.datavalid_s4, opt.patch_size)
    # validation_data_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=opt.val_batchsize, shuffle=False)
    print('Loading Datasets Success!')

    # Model 선언
    print('===> Building model ', opt.model_type)
    if opt.model_type == 'MK_Common_Residual_MEF':
        init_model = network1()
    # init_model = torch.nn.DataParallel(init_model, device_ids=gpus_list)

    # Loss 선언
    criterion_L1 = nn.L1Loss()
    criterion_SSIM_1ch = SSIM(data_range=1, size_average=True, channel=1, nonnegative_ssim=True)
    criterion_SSIM = SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=True)
    # criterion_MEF_MSSSIM = MEF_MSSSIM(is_lum=True)
    criterion_MSE = torch.nn.MSELoss()
    criterion_CA = color_angular_error(color_weight=opt.colorweight)

    print('-----------------Networks architecture-------------------')
    print_network(init_model)
    print('---------------------------------------------------------')
            
    if cuda:
        init_model = init_model.cuda(gpus_list[0])

        criterion_L1 = criterion_L1.cuda(gpus_list[0])
        # criterion_MEF_MSSSIM = criterion_MEF_MSSSIM.cuda(gpus_list[0])
        criterion_SSIM = criterion_SSIM.cuda(gpus_list[0])
        criterion_color = criterion_CA.cuda(gpus_list[0])

    init_optimizer = optim.Adam(init_model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)


    for epoch in range(opt.start_epoch, opt.init_nEpochs + 1):
        initial_train(epoch)
        print(1)
        if (epoch + 1) % (opt.init_nEpochs/8) == 0: #5
            for param_group in init_optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('Learning rate decay: lr={}'.format(init_optimizer.param_groups[0]['lr']))

        if (epoch + 1) % (opt.snapshots) == 0:
            init_checkpoint(epoch)














