import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)

import time
import os
import sys
from dataset import create_transform, visualize_augmentations, TiRadsDataset, TiRadsDataset2
import numpy as np
from tqdm import tqdm
import warnings
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from efficientnet.model import EfficientNet

import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet50
import cv2

dir_img = '/home/alisa/TiradThyroid/Train_Mix/Images/'
dir_mask = '/home/alisa/TiradThyroid/Train_Mix/Masks/'
dir_label = '/home/alisa/TiradThyroid/Train_Mix/Labels/'

dir_img_test = '/home/alisa/TiradThyroid/Test/Images/'
dir_mask_test = '/home/alisa/TiradThyroid/Test/Masks/'
dir_label_test = '/home/alisa/TiradThyroid/Test/Labels/'

load_path = "TIRAD_efficientnet-b3_latest.pth"

checkpoints_dir='./checkpoints'
name = 'test_efficientnet-b3'
gpu_id = 1
device = "cuda:" + str(gpu_id)

torch.cuda.empty_cache()

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        print("No folder ", dir_path, 'exist. Create the folder')
        os.mkdir(dir_path)
        print("Create directory finished")

# types
num_classes = 4 + 4 + 1 * 2 + 4 + 4* 2 + 2
# num_classes = 2
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
print(model)
for param in model.parameters():
    param.requires_grad = True
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
# model.fc = nn.Linear(512, 8) # assuming that the fc7 layer has 512 neurons, otherwise change it 

if load_path is not None:
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)

# print(model)
model.to(device)

create_dir(checkpoints_dir + '/' + name)
n_epochs = 1000
batch_size = 8
save_latest_freq = 10000
save_epoch_freq = 5
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
image_size = 512

transform_train = create_transform(512, True)
transform_test = create_transform(512, False)
dataset = TiRadsDataset2(dir_img, dir_mask, dir_label, transform_train)
dataset_test = TiRadsDataset2(dir_img_test, dir_mask_test, dir_label_test, transform_test)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

n_train = len(train_loader)
n_val = len(val_loader)

criterion_binary = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
# parameters = weights
params_to_update = model.parameters()
# use Adam optimization
optimizer = optim.Adam(params_to_update, lr=0.0005)
# lr scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

total_iters = 0                # the total number of training iterations
global_step = 0

total_iters_test = 0                # the total number of test iterations

loss_history_test = []
loss_history_epoch_test = []
acc_composition_epoch_test = []
acc_echoginicity_epoch_test = []
acc_shape_epoch_test = []
acc_margin_epoch_test = []
acc_macrocal_epoch_test = []
acc_peripheral_epoch_test = []
acc_microcal_epoch_test = []
acc_comet_epoch_test = []
acc_malignant_epoch_test = []

best_acc = 0
best_acc_test = 0

writer = SummaryWriter(name)

epoch_iter_test = 0
running_loss_test = 0.0
running_composition_corrects_test = 0
running_echoginicity_corrects_test = 0
running_shape_corrects_test = 0
running_margin_corrects_test = 0
running_macrocal_corrects_test = 0
running_peripheral_corrects_test = 0
running_microcal_corrects_test = 0
running_comet_corrects_test = 0
running_malignant_corrects_test = 0

# validate
with tqdm(total=n_val * batch_size, desc=f'Test', unit='img') as pbar:
    for batch in val_loader:
        model.eval()
        iter_start_time = time.time()

        total_iters_test += batch_size

        image, img_file, mask_file, is_malignant, composition, echoginicity, shape, margin, macrocal, peripheral, microcal, comet, sizeNodule = (batch['merge'],
                                                                                                                                    batch['img_file'], 
                                                                                                                                    batch['mask_file'],
                                                                                                                                    batch['is_malignant'],
                                                                                                                                    batch['composition'],
                                                                                                                                    batch['echoginicity'],
                                                                                                                                    batch['shape'], 
                                                                                                                                    batch['margin'], 
                                                                                                                                    batch['macrocal'],
                                                                                                                                    batch['peripheral'],
                                                                                                                                    batch['microcal'], 
                                                                                                                                    batch['comet'], 
                                                                                                                                    batch['sizeNodule'])
        
        epoch_iter_test += image.shape[0]

        # print(image.shape)
        image = image.to(device)
        is_malignant = is_malignant.to(device)
        composition = composition.to(device)
        echoginicity = echoginicity.to(device)
        shape = shape.to(device)
        margin = margin.to(device)
        macrocal = macrocal.to(device)
        peripheral = peripheral.to(device)
        microcal = microcal.to(device)
        comet = comet.to(device)

        optimizer.zero_grad()
        classify_output = model(image)

        # print(classify_output.shape)
        pred_composition = classify_output[:,0:4]
        pred_echoginicity = classify_output[:,4:8]
        pred_shape = classify_output[:,8:10]
        pred_margin = classify_output[:,10:14]
        pred_macrocal = classify_output[:,14:16]
        pred_peripheral = classify_output[:,16:18]
        pred_microcal = classify_output[:,18:20]
        pred_comet = classify_output[:,20:22]
        pred_is_malignant = classify_output[:,22:24]

        loss_malignant = criterion(pred_is_malignant, is_malignant)
        loss_composition = criterion(pred_composition, composition)
        loss_echo = criterion(pred_echoginicity, echoginicity)
        loss_shape = criterion(pred_shape, shape)
        loss_margin = criterion(pred_margin, margin)
        loss_macrocal = criterion(pred_macrocal, macrocal)
        loss_peripheral = criterion(pred_peripheral, peripheral)
        loss_microcal = criterion(pred_microcal, microcal)
        loss_comet = criterion(pred_comet, comet)

        loss = (loss_composition + loss_echo + loss_shape + loss_margin + \
                    loss_margin + loss_macrocal + loss_peripheral + loss_microcal + loss_comet + loss_malignant) / 9.0
        loss.backward()     # for what? dunno but need

        _, preds_composition = torch.max(pred_composition, 1)
        _, preds_echoginicity = torch.max(pred_echoginicity, 1)
        _, preds_shape = torch.max(pred_shape, 1)
        _, preds_margin = torch.max(pred_margin, 1)
        _, preds_macrocal = torch.max(pred_macrocal, 1)
        _, preds_peripheral = torch.max(pred_peripheral, 1)
        _, preds_microcal = torch.max(pred_microcal, 1)
        _, preds_comet = torch.max(pred_comet, 1)
        _, pred_is_malignant = torch.max(pred_is_malignant, 1)

        running_loss_test += loss.item() * image.size(0)

        running_malignant_corrects_test += torch.sum(pred_is_malignant == is_malignant.data)

        if writer is not None:
            writer.add_scalar(f'Loss_Overall', loss, global_step)
            writer.add_scalar(f'loss_malignant', loss_malignant, global_step)

        loss_history_test.append(loss.item() * image.size(0))

        # GRAD_CAM
        target_layers = [model.Linear]

        rgb_img = cv2.imread("example01.jpeg", 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        # Note: input_tensor can be a batch tensor with several images!

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

        # You can also use it within a with statement, to make sure it is freed,
        # In case you need to re-create it inside an outer loop:
        # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
        #   ...

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category
        # will be used for every image in the batch.
        # Here we use ClassifierOutputTarget, but you can define your own custom targets
        # That are, for example, combinations of categories, or specific outputs in a non standard model.

        # 281: tabby, tabby cat
        # 229: Old English sheepdog, bobtail
        targets = [ClassifierOutputTarget(281)]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # -----------------END GRADCAM--------------------------------

        iter_elapsed = iter_start_time - time.time()

        loss_str = 'Loss_all: %.4f' % running_loss_test
        pbar.set_postfix(**{'loss (batch)': loss_str})
        pbar.update(image.shape[0])

loss_history_epoch_test.append(running_loss_test)

acc_malignant_epoch_test.append(running_malignant_corrects_test / epoch_iter_test)

overall_acc = running_malignant_corrects_test



