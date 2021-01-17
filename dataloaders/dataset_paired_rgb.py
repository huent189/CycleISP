import torch
import torch.nn as nn
import glob
import os
import PIL
import random
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class PairedRGBDataset(nn.Module):
    def __init__(self, data_dir, crop_size):
        super(PairedRGBDataset, self).__init__()
        print('data dir:', os.path.join(data_dir, "**/*_NOISY_SRGB*PNG"))
        self.crop_size = crop_size
        self.input_imgs = glob.glob(os.path.join(data_dir, "**/*_NOISY_SRGB*PNG"))
        self.input_imgs = sorted(self.input_imgs)
        im = PIL.Image.open(self.input_imgs[0])
        self.input_size = im.size
    def __getitem__(self, idx):
        noisy_path = self.input_imgs[idx]
        clean_path = noisy_path.replace('_NOISY_', '_GT_')
        w, h = self.crop_size
        img_w, img_h = self.input_size
        left = random.randint(0, img_w - w -1)
        top = random.randint(0, img_h - h -1)
        noisy = self._get_cropped_image(noisy_path, (left, top))
        clean = self._get_cropped_image(clean_path, (left, top))
        return noisy, clean, noisy_path, clean_path
    def _get_cropped_image(self, path, position):
        im = PIL.Image.open(path)
        w, h = self.crop_size
        left, top = position
        im = im.crop((left, top, left + w, top + h))
        im = torch.from_numpy(np.array(im))
        im = im / 255.0
        im = im.permute(2, 0, 1)
        return im
    def __len__(self):
        return len(self.input_imgs)
class ValPairedRGBDataset(nn.Module):
    def __init__(self, data_dir, crop_size):
        super(ValPairedRGBDataset, self).__init__()
        self.crop_size = crop_size
        self.input_imgs = glob.glob(os.path.join(data_dir, "**/*_NOISY_SRGB*PNG"))
        self.input_imgs = sorted(self.input_imgs)
        im = PIL.Image.open(self.input_imgs[0])
        self.im_per_row = int(im.size[0] / crop_size[0])
        self.im_per_col = int(im.size[1] / crop_size[1])
    def __getitem__(self, idx):
        im_idx = int(idx / (self.im_per_col * self.im_per_row))
        im_r = idx % (self.im_per_col * self.im_per_row)
        im_col = im_r % self.im_per_row
        im_r = int(im_r / self.im_per_row)
        noisy_path = self.input_imgs[im_idx]
        clean_path = noisy_path.replace('_NOISY_', '_GT_')
        w, h = self.crop_size
        left = w * im_r
        top = h * im_col
        noisy = self._get_cropped_image(noisy_path, (left, top))
        clean = self._get_cropped_image(clean_path, (left, top))
        return noisy, clean, noisy_path, clean_path
    def _get_cropped_image(self, path, position):
        im = PIL.Image.open(path)
        w, h = self.crop_size
        left, top = position
        im = im.crop((left, top, left + w, top + h))
        im = torch.from_numpy(np.array(im))
        im = im / 255.0
        im = im.permute(2, 0, 1)
        # assert im.shape[1] == self.crop_size[0], 'size mismatch {}'.format(im.shape)
        # assert im.shape[2] == self.crop_size[1], 'size mismatch {}'.format(im.shape)
        return im
    def __len__(self):
        return len(self.input_imgs) * self.im_per_col * self.im_per_row