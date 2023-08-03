#%%
import copy
import os

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os import listdir
import logging
import torch
from PIL import Image

from torchvision import transforms
from torchvision.transforms.transforms import RandomCrop

import skimage.measure

def torch_to_numpy(image):
    img=image.numpy()
    img=img.transpose((1, 2, 0))
    return img

def numpy_to_torch(img):
    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img).type(torch.FloatTensor)

#%%
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, label_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.label_dir = label_dir

        self.transform = transform
        self.transform = A.Compose([t for t in transform if not isinstance(t, (A.Normalize, ToTensorV2))])

        list_files = listdir(imgs_dir)
        list_masks = listdir(masks_dir)
        list_label = listdir(label_dir)

        self.ids = [file for file in list_masks if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess2(cls, pil_img, pil_mask):
        newW, newH = pil_img.size
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.convert(mode='L')
        pil_mask = pil_mask.convert(mode='L')
        # pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)
        mask_nd = np.array(pil_mask)

        value_y = np.max(mask_nd, axis=0)
        value_x = np.max(mask_nd, axis=1)
        x_cut = np.argwhere(value_x>=0.5)
        y_min = np.min(x_cut) - 30
        if y_min < 0:
            y_min = 0
        y_max = np.max(x_cut) + 30
        if y_max >= newH:
            y_max = newH - 1

        y_cut = np.argwhere(value_y>=0.5)
        x_min = np.min(y_cut) - 30
        if x_min < 0:
            x_min = 0
        x_max = np.max(y_cut) + 30
        if x_max >= newW:
            x_max = newW - 1

        img = img_nd[x_min:x_max+1, y_min:y_max+1]
        mask = mask_nd[x_min:x_max+1, y_min:y_max+1]

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = self.masks_dir + idx
        label_file = self.label_dir + idx
        img_file = self.imgs_dir + idx[:-7] + idx[-4:]

        #assert len(mask_file) == 1, \
        #    f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        #assert len(img_file) == 1, \
        #    f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file)
        maskClass = 1
        img = Image.open(img_file)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_imgs=False)
        mask = self.preprocess(mask, self.scale, is_imgs=False)
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'img_file': img_file,
            'mask_file': mask_file,
            'file_name' : idx
        }

#%%
# images = ..\dataset_dir\types_dir\images
# masks = ..\dataset_dir\types_dir\masks
# if style special tag, do as image_mask.png

class TiRadsDataset(Dataset):
    def __init__(self, imgs_dir = 'images', masks_dir = 'masks', labels_dir = 'labels', transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.labels_dir = labels_dir

        self.transform = transform
        self.transform = A.Compose([t for t in transform if not isinstance(t, (A.Normalize, ToTensorV2))])

        list_masks = listdir(masks_dir)
        self.ids = [file for file in list_masks if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess_mask(self, image_nd, mask_nd):
        newH, newW = image_nd.shape
        value_y = np.max(mask_nd, axis=0)
        value_x = np.max(mask_nd, axis=1)
        x_cut = np.argwhere(value_x>=0.5)
        y_min = np.min(x_cut) - 50
        if y_min < 0:
            y_min = 0
        y_max = np.max(x_cut) + 50
        if y_max >= newH:
            y_max = newH - 1

        y_cut = np.argwhere(value_y>=0.5)
        x_min = np.min(y_cut) - 50
        if x_min < 0:
            x_min = 0
        x_max = np.max(y_cut) + 50
        if x_max >= newW:
            x_max = newW - 1

        img = image_nd[y_min:y_max+1, x_min:x_max+1]
        mask = mask_nd[y_min:y_max+1, x_min:x_max+1]

        width = int(img.shape[1] * 2)
        height = int(img.shape[0] * 2)
        dim = (width, height)

        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        maxpool_img = skimage.measure.block_reduce(resized, (2,2), np.max)

        return img, mask, maxpool_img

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = self.masks_dir + idx
        label_file = self.labels_dir + idx[:-4] + '.txt'
        img_file = self.imgs_dir + idx[:-7] + idx[-4:]
        #print(img_file)
        #print(mask_file)

        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        image, mask, maxpool_img = self.preprocess_mask(image, mask)

        merge = cv2.merge((image, maxpool_img, mask))

        if self.transform is not None:
            transformed = self.transform(image=merge, mask=mask)
            merge = transformed["image"]
            
        else:
            merge = numpy_to_torch(merge) / 255.0

        f = open(label_file, 'r')
        text = f.read()
        txt = text.split()
        
        composition = int(txt[0])
        echoginicity = int(txt[1])
        shape = int(txt[2])
        margin = int(txt[3])
        macrocal = int(txt[4])
        peripheral = int(txt[5])
        microcal = (int(txt[6]) + int(txt[7]))
        microcal = (1 if microcal >= 1 else 0)
        comet = int(txt[8])

        sizeNodule = int(txt[9])

        return {
            'merge' : numpy_to_torch(merge),
            'img_file': img_file,
            'mask_file': mask_file,
            'composition' : composition,
            'echoginicity' : echoginicity,
            'shape' : shape,
            'margin' : margin,
            'macrocal' : macrocal,
            'peripheral' : peripheral,
            'microcal' : microcal,
            'comet' : comet,
            'sizeNodule' : sizeNodule,
            'file_name' : idx,
        }

#%%

class TiRadsDataset2(Dataset):
    def __init__(self, imgs_dir = 'images', masks_dir = 'masks', labels_dir = 'labels', transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.labels_dir = labels_dir

        self.transform = transform
        self.transform = A.Compose([t for t in transform if not isinstance(t, (A.Normalize, ToTensorV2))])

        list_masks = listdir(masks_dir)
        self.ids = [file for file in list_masks if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess_mask(self, image_nd, mask_nd):
        newH, newW = image_nd.shape
        value_y = np.max(mask_nd, axis=0)
        value_x = np.max(mask_nd, axis=1)
        x_cut = np.argwhere(value_x>=0.5)
        y_min = np.min(x_cut) - 50
        if y_min < 0:
            y_min = 0
        y_max = np.max(x_cut) + 50
        if y_max >= newH:
            y_max = newH - 1

        y_cut = np.argwhere(value_y>=0.5)
        x_min = np.min(y_cut) - 50
        if x_min < 0:
            x_min = 0
        x_max = np.max(y_cut) + 50
        if x_max >= newW:
            x_max = newW - 1

        img = image_nd[y_min:y_max+1, x_min:x_max+1]
        mask = mask_nd[y_min:y_max+1, x_min:x_max+1]

        width = int(img.shape[1] * 2)
        height = int(img.shape[0] * 2)
        dim = (width, height)

        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        maxpool_img = skimage.measure.block_reduce(resized, (2,2), np.max)

        return img, mask, maxpool_img

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = self.masks_dir + idx
        label_file = self.labels_dir + idx[:-4] + '.txt'
        img_file = self.imgs_dir + idx[:-7] + idx[-4:]
        #print(img_file)
        #print(mask_file)

        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        #img2 = cv2.merge((image, image, image))
        #img2 = np.float32(img2) / 255
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        image, mask, maxpool_img = self.preprocess_mask(image, mask)

        merge = cv2.merge((image, maxpool_img, mask))

        if self.transform is not None:
            transformed = self.transform(image=merge, mask=mask)
            merge = transformed["image"]
            
        else:
            merge = numpy_to_torch(merge) / 255.0

        f = open(label_file, 'r')
        text = f.read()
        txt = text.split()
        
        if "benign" in img_file.lower():
            is_malignant = 0
        elif "malignant" in img_file.lower():
            is_malignant = 1
        else:
            is_malignant = -1
        composition = int(txt[0])
        echoginicity = int(txt[1])
        shape = int(txt[2])
        margin = int(txt[3])
        macrocal = int(txt[4])
        peripheral = int(txt[5])
        microcal = (int(txt[6]) + int(txt[7]))
        microcal = (1 if microcal >= 1 else 0)
        comet = int(txt[8])

        sizeNodule = int(txt[9])

        return {
            'merge' : numpy_to_torch(merge),
            'img_file': img_file,
            'mask_file': mask_file,
            'is_malignant' : is_malignant,
            'composition' : composition,
            'echoginicity' : echoginicity,
            'shape' : shape,
            'margin' : margin,
            'macrocal' : macrocal,
            'peripheral' : peripheral,
            'microcal' : microcal,
            'comet' : comet,
            'sizeNodule' : sizeNodule,
            'file_name' : idx,
            #'image' : img2,
        }

class TiRadsDataset4GradCam(Dataset):
    def __init__(self, imgs_dir = 'images', masks_dir = 'masks', labels_dir = 'labels', transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.labels_dir = labels_dir

        self.transform = transform
        self.transform = A.Compose([t for t in transform if not isinstance(t, (A.Normalize, ToTensorV2))])

        list_masks = listdir(masks_dir)
        self.ids = [file for file in list_masks if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess_mask(self, image_nd, mask_nd):
        newH, newW = image_nd.shape
        value_y = np.max(mask_nd, axis=0)
        value_x = np.max(mask_nd, axis=1)
        x_cut = np.argwhere(value_x>=0.5)
        y_min = np.min(x_cut) - 50
        if y_min < 0:
            y_min = 0
        y_max = np.max(x_cut) + 50
        if y_max >= newH:
            y_max = newH - 1

        y_cut = np.argwhere(value_y>=0.5)
        x_min = np.min(y_cut) - 50
        if x_min < 0:
            x_min = 0
        x_max = np.max(y_cut) + 50
        if x_max >= newW:
            x_max = newW - 1

        img = image_nd[y_min:y_max+1, x_min:x_max+1]
        mask = mask_nd[y_min:y_max+1, x_min:x_max+1]

        width = int(img.shape[1] * 2)
        height = int(img.shape[0] * 2)
        dim = (width, height)

        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        maxpool_img = skimage.measure.block_reduce(resized, (2,2), np.max)

        return img, mask, maxpool_img

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = self.masks_dir + idx
        label_file = self.labels_dir + idx[:-4] + '.txt'
        img_file = self.imgs_dir + idx[:-7] + idx[-4:]
        #print(img_file)
        #print(mask_file)

        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.merge((image, image, image))
        img2 = np.float32(img2) / 255
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        image, mask, maxpool_img = self.preprocess_mask(image, mask)

        merge = cv2.merge((image, maxpool_img, mask))

        if self.transform is not None:
            transformed = self.transform(image=merge, mask=mask)
            merge = transformed["image"]
            
        else:
            merge = numpy_to_torch(merge) / 255.0

        f = open(label_file, 'r')
        text = f.read()
        txt = text.split()
        
        if "benign" in img_file.lower():
            is_malignant = 0
        elif "malignant" in img_file.lower():
            is_malignant = 1
        else:
            is_malignant = -1
        composition = int(txt[0])
        echoginicity = int(txt[1])
        shape = int(txt[2])
        margin = int(txt[3])
        macrocal = int(txt[4])
        peripheral = int(txt[5])
        microcal = (int(txt[6]) + int(txt[7]))
        microcal = (1 if microcal >= 1 else 0)
        comet = int(txt[8])

        sizeNodule = int(txt[9])

        return {
            'merge' : numpy_to_torch(merge),
            'img_file': img_file,
            'mask_file': mask_file,
            'is_malignant' : is_malignant,
            'composition' : composition,
            'echoginicity' : echoginicity,
            'shape' : shape,
            'margin' : margin,
            'macrocal' : macrocal,
            'peripheral' : peripheral,
            'microcal' : microcal,
            'comet' : comet,
            'sizeNodule' : sizeNodule,
            'file_name' : idx,
            'image' : img2,
        }

#%%
def visualize_augmentations(dataset, idx=0, samples=5):
    #dataset = copy.deepcopy(dataset)
    #dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=3, figsize=(10, 24))
    for i in range(samples):
        batch = dataset[idx]
        image, img_file, mask_file, composition = batch['merge'], batch['img_file'], batch['mask_file'], batch['composition']
        print(img_file)
        print(mask_file)
        print(image.shape)
        nd_image = torch_to_numpy(image)
        ax[i, 0].imshow(nd_image[:,:,0])
        ax[i, 1].imshow(nd_image[:,:,1])
        ax[i, 2].imshow(nd_image[:,:,2], cmap='gray')
        # ax[i, 2].imshow(torch_to_numpy(mask), cmap='jet', alpha=0.5)
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 2].set_title("Augmented pooling image")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    plt.tight_layout()
    plt.show()

def create_transform(image_size=None, is_train=False):
    if is_train:
        if image_size is not None:
            transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.15, rotate_limit=30, p=0.6, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    #A.RandomCrop(image_size, image_size, always_apply=False, p=1.0),
                    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            transform = A.Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
    else: # eval image
        if image_size is not None:
            transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
                    ToTensorV2()
                ]
            )
        else:
            transform = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
                    ToTensorV2()
                ]
            )
    return transform

#%%
if __name__=='__main__':
    # example thyroid dataset
    dir_img = '/home/alisa/TiradThyroid/Train/Images/'
    dir_mask = '/home/alisa/TiradThyroid/Train/Masks/'
    dir_label = '/home/alisa/TiradThyroid/Train/Labels/'
    transform = create_transform(256, True)
    dataset = TiRadsDataset(dir_img, dir_mask, dir_label, transform)

    batch = dataset[15]
    image, img_file, mask_file, composition = batch['merge'], batch['img_file'], batch['mask_file'], batch['composition']
    print(img_file)
    print(mask_file)
    print(composition)
    print(image.shape)
    print(image)

    # %%
    visualize_augmentations(dataset, idx=5)
    #visualize_augmentations(dataset, idx=55)
# %%
