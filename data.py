from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, ToTensor

from datasetfolder import DatasetFromFolderTest, DatasetFromFolder, DatasetFromFolderValid

def transform():
    return Compose([
        ToTensor(),
    ])

#def get_training_set(data_dir, data_dir10, data_dir20, data_dir30, data_augmentation, file_list, other_dataset, patch_size):
#return DatasetFromFolder(data_dir, data_dir10, data_dir20, data_dir30, data_augmentation, file_list, other_dataset, patch_size,
#                             transform=transform())

def get_training_set(data_dir, data_dir10, data_dir20, data_dir30, data_dir40, data_dir50, data_dir60, patch_size):
    return DatasetFromFolder(data_dir, data_dir10, data_dir20, data_dir30, data_dir40, data_dir50, data_dir60, patch_size, transform=transform())

def get_validation_set(data_dir10, data_dir20, data_dir30, data_dir40, patch_size):
    return DatasetFromFolderValid(data_dir10, data_dir20, data_dir30, data_dir40, patch_size, transform=transform())


def get_eval_set(data_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame):
    return DatasetFromFolder(data_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,future_frame,
                             transform=transform())

#def get_test_set(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame):
#    return DatasetFromFolderTest(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=transform())


def get_test_set(data_dir10, data_dir20, data_dir30, data_dir40, data_dir50, data_dir60, data_dir70):
    return DatasetFromFolderTest(data_dir10, data_dir20, data_dir30, data_dir40, data_dir50, data_dir60, data_dir70,
                             transform=transform())