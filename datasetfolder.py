import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
#import pyflow
import cv2
# from skimage import img_as_float
from random import randrange
import os.path

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "bmp"])

def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    #random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]

        for i in seq:
            index = int(filepath[char_len-7:char_len-4])-i
            file_name=filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png'
            
            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png').convert('RGB'),scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath,'im'+str(nFrames)+'.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for j in reversed(seq)]
    
    return target, input, neigbor

def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames/2)
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]
        if nFrames%2 == 0:
            seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt,tt+1) if x!=0]
        #random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len-7:char_len-4])+i
            file_name1=filepath[0:char_len-7]+'{0:03d}'.format(index1)+'.png'
            
            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame- is not exist')
                temp=input
                neigbor.append(temp)
            
    else:
        target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4-tt,5+tt) if x!=4]
        #random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
    return target, input, neigbor

def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,0,1)
    return flow

def rescale_flow(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo);
    iw = iw - (iw%modulo);
    img = img.crop((0, 0, ih, iw))
    return img


def get_patch(img_tar, img_tar1, img_tar2, img_tar3, img_tar4, patch_size, ix=-1, iy=-1):
    # img_tar : org, img_tar1 : NR30, img_tar2 : NR20, img_tar3 : NR10
    ih = img_tar.shape[0]
    iw = img_tar.shape[1]

    if ix == -1:
        ix = random.randrange(0, iw - patch_size + 1)
    if iy == -1:
        iy = random.randrange(0, ih - patch_size + 1)

    # img_in = img_in.crop((iy,ix,iy + ip, ix + ip))
    img_tar = img_tar[iy:iy + patch_size, ix:ix + patch_size]
    img_tar1 = img_tar1[iy:iy + patch_size, ix:ix + patch_size]
    img_tar2 = img_tar2[iy:iy + patch_size, ix:ix + patch_size]
    img_tar3 = img_tar3[iy:iy + patch_size, ix:ix + patch_size]
    img_tar4 = img_tar4[iy:iy + patch_size, ix:ix + patch_size]

    info_patch = {'ix': ix, 'iy': iy}

    return img_tar, img_tar1, img_tar2, img_tar3, img_tar4, info_patch

def get_valid_patch(img_tar1, img_tar2, img_tar3, img_tar4, patch_size, ix=250, iy=300):
    # img_tar : org, img_tar1 : NR30, img_tar2 : NR20, img_tar3 : NR10
    ih = img_tar1.shape[0]
    iw = img_tar1.shape[1]

    if ix == -1:
        ix = random.randrange(0, iw - patch_size + 1)
    if iy == -1:
        iy = random.randrange(0, ih - patch_size + 1)

    # img_in = img_in.crop((iy,ix,iy + ip, ix + ip))
    # img_tar = img_tar[iy:iy + patch_size, ix:ix + patch_size]
    img_tar1 = img_tar1[iy:iy + patch_size, ix:ix + patch_size]
    img_tar2 = img_tar2[iy:iy + patch_size, ix:ix + patch_size]
    img_tar3 = img_tar3[iy:iy + patch_size, ix:ix + patch_size]
    img_tar4 = img_tar4[iy:iy + patch_size, ix:ix + patch_size]

    info_patch = {'ix': ix, 'iy': iy}

    return img_tar1, img_tar2, img_tar3, img_tar4, info_patch


def augment(img_in, img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn, info_aug
    
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, image_dir10, image_dir20, image_dir30, image_dir40, image_dir50, image_dir60, patch_size, transform=None):
    #def __init__(self, image_dir, image_dir10, image_dir20, image_dir30, patch_size, data_augmentation,transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_filenames10 = [join(image_dir10, x10) for x10 in listdir(image_dir10) if is_image_file(x10)]
        self.image_filenames20 = [join(image_dir20, x20) for x20 in listdir(image_dir20) if is_image_file(x20)]
        self.image_filenames30 = [join(image_dir30, x30) for x30 in listdir(image_dir30) if is_image_file(x30)]
        self.image_filenames40 = [join(image_dir40, x40) for x40 in listdir(image_dir40) if is_image_file(x40)]

        self.image_filenames50 = [join(image_dir50, x50) for x50 in listdir(image_dir50) if is_image_file(x50)]
        self.image_filenames60 = [join(image_dir60, x60) for x60 in listdir(image_dir60) if is_image_file(x60)]


        self.patch_size = patch_size
        self.transform = transform
        #self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        target = cv2.imread(self.image_filenames[index], -1)
        target10 = cv2.imread(self.image_filenames10[index], -1)
        target20 = cv2.imread(self.image_filenames20[index], -1)
        target30 = cv2.imread(self.image_filenames30[index], -1)
        target40 = cv2.imread(self.image_filenames40[index], -1)
        target50 = cv2.imread(self.image_filenames50[index], -1)
        target60 = cv2.imread(self.image_filenames60[index], -1)

        if self.transform:
            target = self.transform(target)
            target10 = self.transform(target10)
            target20 = self.transform(target20)
            target30 = self.transform(target30)
            target40 = self.transform(target40)
            target50 = self.transform(target50)
            target60 = self.transform(target60)

        return target, target10, target20, target30, target40, target50, target60

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderValid(data.Dataset):
    def __init__(self, image_dir10, image_dir20, image_dir30, image_dir40, patch_size, transform=None):
    #def __init__(self, image_dir, image_dir10, image_dir20, image_dir30, patch_size, data_augmentation,transform=None):
        super(DatasetFromFolderValid, self).__init__()
        # self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_filenames10 = [join(image_dir10, x10) for x10 in listdir(image_dir10) if is_image_file(x10)]
        self.image_filenames20 = [join(image_dir20, x20) for x20 in listdir(image_dir20) if is_image_file(x20)]
        self.image_filenames30 = [join(image_dir30, x30) for x30 in listdir(image_dir30) if is_image_file(x30)]
        self.image_filenames40 = [join(image_dir40, x40) for x40 in listdir(image_dir40) if is_image_file(x40)]

        self.patch_size = patch_size
        self.transform = transform
        #self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        # target = cv2.imread(self.image_filenames[index], -1)
        target10 = cv2.imread(self.image_filenames10[index], -1)
        target20 = cv2.imread(self.image_filenames20[index], -1)
        target30 = cv2.imread(self.image_filenames30[index], -1)
        target40 = cv2.imread(self.image_filenames40[index], -1)

        # target = np.float32(target)
        # target10 = np.float32(target10)
        # target20 = np.float32(target20)
        # target30 = np.float32(target30)

        target, target10, target20, target30, target40, _ = get_valid_patch(target10, target20, target30, target40, self.patch_size)

        #if self.data_augmentation:
        #    target, target1, target2,target30, _ = augment(target, target1, target2, target3)

        if self.transform:
            # target = self.transform(target)
            target10 = self.transform(target10)
            target20 = self.transform(target20)
            target30 = self.transform(target30)
            target40 = self.transform(target40)

        return target10, target20, target30, target40

    def __len__(self):
        return len(self.image_filenames10)

class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir10, image_dir20, image_dir30, image_dir40, image_dir50, image_dir60, image_dir70, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_filenames10 = [join(image_dir10, x10) for x10 in listdir(image_dir10) if is_image_file(x10)]
        self.image_filenames20 = [join(image_dir20, x20) for x20 in listdir(image_dir20) if is_image_file(x20)]
        self.image_filenames30 = [join(image_dir30, x30) for x30 in listdir(image_dir30) if is_image_file(x30)]
        self.image_filenames40 = [join(image_dir40, x40) for x40 in listdir(image_dir40) if is_image_file(x40)]
        self.image_filenames50 = [join(image_dir50, x50) for x50 in listdir(image_dir50) if is_image_file(x50)]
        self.image_filenames60 = [join(image_dir60, x60) for x60 in listdir(image_dir60) if is_image_file(x60)]
        self.image_filenames70 = [join(image_dir70, x70) for x70 in listdir(image_dir70) if is_image_file(x70)]
        # self.image_filenames80 = [join(image_dir80, x80) for x80 in listdir(image_dir80) if is_image_file(x80)]

        #self.patch_size = patch_size
        self.transform = transform
        # self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        #target = cv2.imread(self.image_filenames[index], -1)
        target10 = cv2.imread(self.image_filenames10[index], -1)
        target20 = cv2.imread(self.image_filenames20[index], -1)
        target30 = cv2.imread(self.image_filenames30[index], -1)
        target40 = cv2.imread(self.image_filenames40[index], -1)
        target50 = cv2.imread(self.image_filenames50[index], -1)
        target60 = cv2.imread(self.image_filenames60[index], -1)
        target70 = cv2.imread(self.image_filenames70[index], -1)
        # target80 = cv2.imread(self.image_filenames80[index], -1)

        if self.transform:
            #target = self.transform(target)
            target10 = self.transform(target10)
            target20 = self.transform(target20)
            target30 = self.transform(target30)
            target40 = self.transform(target40)
            target50 = self.transform(target50)
            target60 = self.transform(target60)
            target70 = self.transform(target70)
            # target80 = self.transform(target80)

        return target10, target20, target30, target40, target50, target60, target70 #, target80

    def __len__(self):
        return len(self.image_filenames10)


'''
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
            
        flow = [get_flow(input,j) for j in neigbor]

        bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]
            
        return input, target, neigbor, flow, bicubic
      
    def __len__(self):
        return len(self.image_filenames)
'''