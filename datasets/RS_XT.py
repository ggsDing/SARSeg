import os
import torch
import numpy as np
from skimage import io, exposure
from torch.utils import data
import utils.transform as transform
import glob

num_classes = 7
XT_COLORMAP = [[125, 125, 125],[0, 255, 255], [255, 255, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255], [0, 0, 0]]
XT_CLASSES  = ['Invalid',       'Water',      'Built-up',    'Factory',    'Grassland', 'Barren',   'others',        'background']

XT_MEAN = np.array([134.19, 181.63, 179.26, 141.37])
XT_STD  = np.array([141.34, 166.50, 164.96, 138.58])

data_dir = '/YOUR_DATA_DIR/'

def normalize_image(img):
    img = img.astype(np.float32)
    return (img - XT_MEAN) / XT_STD

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(XT_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Index2Color(pred):
    colormap = np.asarray(XT_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap <= num_classes)
    return IndexMap.astype(np.uint8)

# util function for reading gt
def load_gt(path_gt):
    gt = io.imread(path_gt)
    gt = Color2Index(gt)
    return gt

# util function for reading s1 data
def load_sar(path):
    hh = io.imread(path)
    hh = np.clip(hh, 0, 1000)
    #hh = exposure.rescale_intensity(hh, out_range=(0,255))
    hh = np.squeeze(hh)
    
    path_hv = path.replace("HH", "HV")
    hv = io.imread(path_hv)
    hv = np.clip(hv, 0, 1000)
    #hv = exposure.rescale_intensity(hv, out_range=(0,255))
    hv = np.squeeze(hv)
    
    path_vh = path.replace("HH", "VH")
    vh = io.imread(path_vh)
    vh = np.clip(vh, 0, 1000)
    #vh = exposure.rescale_intensity(vh, out_range=(0,255))
    vh = np.squeeze(vh)
    
    path_vv = path.replace("HH", "VV")
    vv = io.imread(path_vv)
    vv = np.clip(vv, 0, 1000)
    #vv = exposure.rescale_intensity(vv, out_range=(0,255))
    vv = np.squeeze(vv)
    
    #sar = np.array([hh, hv, vh, vv])  #C*W*H
    sar  = np.dstack((hh, hv, vh, vv)) #W*H*C
    #sar = np.clip(sar, 0, 1000)
    # normaliza -1~1
    sar = normalize_image(sar)
    return sar


class PolSAR(data.Dataset):
    def __init__(self, mode='train', random_flip=False, crop_size=False):
        assert mode in ['train', 'val', 'test']
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.data = []
        self.label = []
        self.mask_name_list = []
        img_dir = os.path.join(data_dir, mode)
        img_path_list = glob.glob(os.path.join(img_dir, "*HH.tiff"), recursive=True)
        self.len = len(img_path_list)
        count=1
        for img_path in img_path_list:
            self.data.append(load_sar(img_path))
            mask_name = os.path.basename(os.path.splitext(img_path)[0]).replace("HH", "gt") + '.png'
            gt_path = os.path.join(data_dir, 'all', mask_name)
            self.label.append(load_gt(gt_path))
            self.mask_name_list.append(mask_name)
            if not count%50: print('%d/%d '%(count, self.len) + mode + ' images loaded.')
            count+=1

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]

        if self.crop_size:
            data, label = transform.random_crop(data, label, size=[self.crop_size, self.crop_size])
        if self.random_flip:
            data, label = transform.rand_flip(data, label)

        data = torch.from_numpy(data.transpose((2, 0, 1)))
        return data, label

    def __len__(self):
        return self.len

class PolSAR_test(data.Dataset):
    def __init__(self, path):
        self.data = []
        self.mask_name_list = []
        img_path_list = glob.glob(os.path.join(path, "*HH.tiff"), recursive=True)
        for data_path in img_path_list:
            self.data.append(load_sar(data_path))
            file_name = os.path.basename(os.path.splitext(data_path)[0])
            file_name = file_name.replace("HH", "gt")
            self.mask_name_list.append( file_name + '.png')
        self.len = len(img_path_list)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        data = self.data[idx]
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        return data

    def __len__(self):
        return self.len