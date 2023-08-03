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

dir_img = '/home/alisa/TiradThyroid/Train_Mix/Images/'
dir_mask = '/home/alisa/TiradThyroid/Train_Mix/Masks/'
dir_label = '/home/alisa/TiradThyroid/Train_Mix/Labels/'

dir_img_test = '/home/alisa/TiradThyroid/Test/Images/'
dir_mask_test = '/home/alisa/TiradThyroid/Test/Masks/'
dir_label_test = '/home/alisa/TiradThyroid/Test/Labels/'

#transform = create_transform(256, True)
#dataset = TiRadsDataset2(dir_img, dir_mask, dir_label, transform)
#dataset_test = TiRadsDataset2(dir_img_test, dir_mask_test, dir_label_test, transform)

load_path = None

checkpoints_dir='./checkpoints'
name = 'Benign_efficientnet-b3'
gpu_id = 1
device = "cuda:" + str(gpu_id)

torch.cuda.empty_cache()

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        print("No folder ", dir_path, 'exist. Create the folder')
        os.mkdir(dir_path)
        print("Create directory finished")

# types
#num_classes = 4 + 4 + 1 * 2 + 4 + 4* 2 + 2
num_classes = 2
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

loss_history = []
loss_history_epoch = []
acc_composition_epoch = []
acc_echoginicity_epoch = []
acc_shape_epoch = []
acc_margin_epoch = []
acc_macrocal_epoch = []
acc_peripheral_epoch = []
acc_microcal_epoch = []
acc_comet_epoch = []
acc_malignant_epoch = []

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

for epoch in range(1, n_epochs + 1):
    epoch_start_time = time.time()  # timer for entire epoch
    #iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    epoch_iter_test = 0                  # the number of test iterations in current epoch, reset to 0 every epoch

    running_loss = 0.0
    running_composition_corrects = 0
    running_echoginicity_corrects = 0
    running_shape_corrects = 0
    running_margin_corrects = 0
    running_macrocal_corrects = 0
    running_peripheral_corrects = 0
    running_microcal_corrects = 0
    running_comet_corrects = 0
    running_malignant_corrects = 0

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

    model.train()
    with tqdm(total=n_train * batch_size, desc=f'Epoch {epoch}/{n_epochs}', unit='img') as pbar:
        for batch in train_loader:
            iter_start_time = time.time()

            total_iters += batch_size
            global_step += 1

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
            
            epoch_iter += image.shape[0]

            # print(image.shape)
            image = image.to(device)
            is_malignant = is_malignant.to(device)
            # composition = composition.to(device)
            # echoginicity = echoginicity.to(device)
            # shape = shape.to(device)
            # margin = margin.to(device)
            # macrocal = macrocal.to(device)
            # peripheral = peripheral.to(device)
            # microcal = microcal.to(device)
            # comet = comet.to(device)

            optimizer.zero_grad()

            classify_output = model(image)

            # print(classify_output.shape)
            # pred_composition = classify_output[:,0:4]
            # pred_echoginicity = classify_output[:,4:8]
            # pred_shape = classify_output[:,8:10]
            # pred_margin = classify_output[:,10:14]
            # pred_macrocal = classify_output[:,14:16]
            # pred_peripheral = classify_output[:,16:18]
            # pred_microcal = classify_output[:,18:20]
            # pred_comet = classify_output[:,20:22]
            # pred_is_malignant = classify_output[:,22:24]
            pred_is_malignant = classify_output

            # loss_composition = criterion(pred_composition, composition)
            # loss_echo = criterion(pred_echoginicity, echoginicity)
            # loss_shape = criterion(pred_shape, shape)
            # loss_margin = criterion(pred_margin, margin)
            # loss_macrocal = criterion(pred_macrocal, macrocal)
            # loss_peripheral = criterion(pred_peripheral, peripheral)
            # loss_microcal = criterion(pred_microcal, microcal)
            # loss_comet = criterion(pred_comet, comet)
            loss_malignant = criterion(pred_is_malignant, is_malignant)

            # loss = (loss_composition + loss_echo + loss_shape + loss_margin + \
            #             loss_margin + loss_macrocal + loss_peripheral + loss_microcal + loss_comet + loss_malignant) / 9.0
            loss = loss_malignant

            # print(loss.item())

            # _, preds_composition = torch.max(pred_composition, 1)
            # _, preds_echoginicity = torch.max(pred_echoginicity, 1)
            # _, preds_shape = torch.max(pred_shape, 1)
            # _, preds_margin = torch.max(pred_margin, 1)
            # _, preds_macrocal = torch.max(pred_macrocal, 1)
            # _, preds_peripheral = torch.max(pred_peripheral, 1)
            # _, preds_microcal = torch.max(pred_microcal, 1)
            # _, preds_comet = torch.max(pred_comet, 1)
            _, pred_is_malignant = torch.max(pred_is_malignant, 1)

            running_loss += loss.item() * image.size(0)

            # running_composition_corrects += torch.sum(preds_composition == composition.data)
            # running_echoginicity_corrects += torch.sum(preds_echoginicity == echoginicity.data)
            # running_shape_corrects += torch.sum(preds_shape == shape.data)
            # running_margin_corrects += torch.sum(preds_margin == margin.data)
            # running_macrocal_corrects += torch.sum(preds_macrocal == macrocal.data)
            # running_peripheral_corrects += torch.sum(preds_peripheral == peripheral.data)
            # running_microcal_corrects += torch.sum(preds_microcal == microcal.data)
            # running_comet_corrects += torch.sum(preds_comet == comet.data)
            running_malignant_corrects += torch.sum(pred_is_malignant == is_malignant.data)

            if writer is not None:
                writer.add_scalar(f'Loss_Overall_Train', loss, global_step)
                # writer.add_scalar(f'loss_composition_Train', loss_composition, global_step)
                # writer.add_scalar(f'loss_echo_Train', loss_echo, global_step)
                # writer.add_scalar(f'loss_shape_Train', loss_shape, global_step)
                # writer.add_scalar(f'loss_margin_Train', loss_margin, global_step)
                # writer.add_scalar(f'loss_macrocal_Train', loss_macrocal, global_step)
                # writer.add_scalar(f'preds_peripheral_Train', loss_peripheral, global_step)
                # writer.add_scalar(f'loss_microcal_Train', loss_microcal, global_step)
                # writer.add_scalar(f'loss_comet_Train', loss_comet, global_step)
                writer.add_scalar(f'loss_malignant_Train', loss_malignant, global_step)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item() * image.size(0))

            iter_elapsed = iter_start_time - time.time()

            loss_str = 'Loss_all: %.4f' % running_loss
            pbar.set_postfix(**{'loss (batch)': loss_str})
            pbar.update(image.shape[0])
        
    scheduler.step()
    print(f'epoch {epoch}' , "   lr  {:.8f}".format(scheduler.get_last_lr()[0]))

    loss_history_epoch.append(running_loss)

    # acc_composition_epoch.append(running_composition_corrects / epoch_iter)
    # acc_echoginicity_epoch.append(running_echoginicity_corrects / epoch_iter)
    # acc_shape_epoch.append(running_shape_corrects / epoch_iter)
    # acc_margin_epoch.append(running_margin_corrects / epoch_iter)
    # acc_macrocal_epoch.append(running_macrocal_corrects / epoch_iter)
    # acc_peripheral_epoch.append(running_peripheral_corrects / epoch_iter)
    # acc_microcal_epoch.append(running_microcal_corrects / epoch_iter)
    # acc_comet_epoch.append(running_comet_corrects / epoch_iter)
    acc_malignant_epoch.append(running_malignant_corrects / epoch_iter)

    # overall_acc = running_composition_corrects + running_echoginicity_corrects + running_shape_corrects + running_margin_corrects + \
    #                 running_macrocal_corrects + running_peripheral_corrects + running_microcal_corrects + running_comet_corrects + running_malignant_corrects
    overall_acc = running_malignant_corrects

    if overall_acc > best_acc:
        best_acc = overall_acc
        # torch.save(model.state_dict(), name + "_best.pth")
    if epoch % save_epoch_freq == 0:
        torch.save(model.state_dict(), name + "_latest.pth")

    if writer is not None:
        writer.add_scalar(f'Loss_Overall_epoch_Train', running_loss, epoch)
        # writer.add_scalar(f'acc_composition_corrects_Train', (running_composition_corrects / epoch_iter), epoch)
        # writer.add_scalar(f'acc_echoginicity_corrects_Train', running_echoginicity_corrects / epoch_iter, epoch)
        # writer.add_scalar(f'acc_shape_corrects_Train', running_shape_corrects / epoch_iter, epoch)
        # writer.add_scalar(f'acc_margin_corrects_Train', running_margin_corrects / epoch_iter, epoch)
        # writer.add_scalar(f'acc_macrocal_corrects_Train', running_macrocal_corrects / epoch_iter, epoch)
        # writer.add_scalar(f'acc_peripheral_corrects_Train', running_peripheral_corrects / epoch_iter, epoch)
        # writer.add_scalar(f'acc_microcal_corrects_Train', running_microcal_corrects / epoch_iter, epoch)
        # writer.add_scalar(f'acc_comet_corrects_Train', running_comet_corrects / epoch_iter, epoch)
        writer.add_scalar(f'acc_malignant_corrects_Train', running_malignant_corrects / epoch_iter, epoch)

    # validate
    with tqdm(total=n_val * batch_size, desc=f'Epoch {epoch}/{n_epochs}', unit='img') as pbar:
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
            # composition = composition.to(device)
            # echoginicity = echoginicity.to(device)
            # shape = shape.to(device)
            # margin = margin.to(device)
            # macrocal = macrocal.to(device)
            # peripheral = peripheral.to(device)
            # microcal = microcal.to(device)
            # comet = comet.to(device)

            optimizer.zero_grad()
            classify_output = model(image)

            # print(classify_output.shape)
            # pred_composition = classify_output[:,0:4]
            # pred_echoginicity = classify_output[:,4:8]
            # pred_shape = classify_output[:,8:10]
            # pred_margin = classify_output[:,10:14]
            # pred_macrocal = classify_output[:,14:16]
            # pred_peripheral = classify_output[:,16:18]
            # pred_microcal = classify_output[:,18:20]
            # pred_comet = classify_output[:,20:22]
            # pred_is_malignant = classify_output[:,22:24]
            pred_is_malignant = classify_output

            # loss_composition = criterion(pred_composition, composition)
            # loss_echo = criterion(pred_echoginicity, echoginicity)
            # loss_shape = criterion(pred_shape, shape)
            # loss_margin = criterion(pred_margin, margin)
            # loss_macrocal = criterion(pred_macrocal, macrocal)
            # loss_peripheral = criterion(pred_peripheral, peripheral)
            # loss_microcal = criterion(pred_microcal, microcal)
            # loss_comet = criterion(pred_comet, comet)
            loss_malignant = criterion(pred_is_malignant, is_malignant)

            # loss = (loss_composition + loss_echo + loss_shape + loss_margin + \
            #             loss_margin + loss_macrocal + loss_peripheral + loss_microcal + loss_comet + loss_malignant) / 9.0
            loss = loss_malignant
            loss.backward()     # for what? dunno but need

            # _, preds_composition = torch.max(pred_composition, 1)
            # _, preds_echoginicity = torch.max(pred_echoginicity, 1)
            # _, preds_shape = torch.max(pred_shape, 1)
            # _, preds_margin = torch.max(pred_margin, 1)
            # _, preds_macrocal = torch.max(pred_macrocal, 1)
            # _, preds_peripheral = torch.max(pred_peripheral, 1)
            # _, preds_microcal = torch.max(pred_microcal, 1)
            # _, preds_comet = torch.max(pred_comet, 1)
            _, pred_is_malignant = torch.max(pred_is_malignant, 1)

            running_loss_test += loss.item() * image.size(0)

            # running_composition_corrects_test += torch.sum(preds_composition == composition.data)
            # running_echoginicity_corrects_test += torch.sum(preds_echoginicity == echoginicity.data)
            # running_shape_corrects_test += torch.sum(preds_shape == shape.data)
            # running_margin_corrects_test += torch.sum(preds_margin == margin.data)
            # running_macrocal_corrects_test += torch.sum(preds_macrocal == macrocal.data)
            # running_peripheral_corrects_test += torch.sum(preds_peripheral == peripheral.data)
            # running_microcal_corrects_test += torch.sum(preds_microcal == microcal.data)
            # running_comet_corrects_test += torch.sum(preds_comet == comet.data)
            running_malignant_corrects_test += torch.sum(pred_is_malignant == is_malignant.data)

            if writer is not None:
                writer.add_scalar(f'Loss_Overall', loss, global_step)
                # writer.add_scalar(f'loss_composition', loss_composition, global_step)
                # writer.add_scalar(f'loss_echo', loss_echo, global_step)
                # writer.add_scalar(f'loss_shape', loss_shape, global_step)
                # writer.add_scalar(f'loss_margin', loss_margin, global_step)
                # writer.add_scalar(f'loss_macrocal', loss_macrocal, global_step)
                # writer.add_scalar(f'preds_peripheral', loss_peripheral, global_step)
                # writer.add_scalar(f'loss_microcal', loss_microcal, global_step)
                # writer.add_scalar(f'loss_comet', loss_comet, global_step)
                writer.add_scalar(f'loss_malignant', loss_malignant, global_step)

            loss_history_test.append(loss.item() * image.size(0))

            iter_elapsed = iter_start_time - time.time()

            loss_str = 'Loss_all: %.4f' % running_loss_test
            pbar.set_postfix(**{'loss (batch)': loss_str})
            pbar.update(image.shape[0])

    loss_history_epoch_test.append(running_loss_test)

    # acc_composition_epoch_test.append(running_composition_corrects_test / epoch_iter_test)
    # acc_echoginicity_epoch_test.append(running_echoginicity_corrects_test / epoch_iter_test)
    # acc_shape_epoch_test.append(running_shape_corrects_test / epoch_iter_test)
    # acc_margin_epoch_test.append(running_margin_corrects_test / epoch_iter_test)
    # acc_macrocal_epoch_test.append(running_macrocal_corrects_test / epoch_iter_test)
    # acc_peripheral_epoch_test.append(running_peripheral_corrects_test / epoch_iter_test)
    # acc_microcal_epoch_test.append(running_microcal_corrects_test / epoch_iter_test)
    # acc_comet_epoch_test.append(running_comet_corrects_test / epoch_iter_test)
    acc_malignant_epoch_test.append(running_malignant_corrects_test / epoch_iter_test)

    # overall_acc = running_composition_corrects_test + running_echoginicity_corrects_test + running_shape_corrects_test + running_margin_corrects_test + \
    #                 running_macrocal_corrects_test + running_peripheral_corrects_test + running_microcal_corrects_test + running_comet_corrects_test + \
    #                 running_malignant_corrects_test
    overall_acc = running_malignant_corrects_test

    if overall_acc > best_acc_test:
        best_acc_test = overall_acc
        torch.save(model.state_dict(), name + "_best.pth")

    if writer is not None:
        writer.add_scalar(f'Loss_Overall_epoch_Test', running_loss_test, epoch)
        # writer.add_scalar(f'acc_composition_corrects_Test', (running_composition_corrects_test / epoch_iter_test), epoch)
        # writer.add_scalar(f'acc_echoginicity_corrects_Test', running_echoginicity_corrects_test / epoch_iter_test, epoch)
        # writer.add_scalar(f'acc_shape_corrects_Test', running_shape_corrects_test / epoch_iter_test, epoch)
        # writer.add_scalar(f'acc_margin_corrects_Test', running_margin_corrects_test / epoch_iter_test, epoch)
        # writer.add_scalar(f'acc_macrocal_corrects_Test', running_macrocal_corrects_test / epoch_iter_test, epoch)
        # writer.add_scalar(f'acc_peripheral_corrects_Test', running_peripheral_corrects_test / epoch_iter_test, epoch)
        # writer.add_scalar(f'acc_microcal_corrects_Test', running_microcal_corrects_test / epoch_iter_test, epoch)
        # writer.add_scalar(f'acc_comet_corrects_Test', running_comet_corrects_test / epoch_iter_test, epoch)
        writer.add_scalar(f'acc_malignant_corrects_Test', running_malignant_corrects_test / epoch_iter_test, epoch)

    epoch_elapsed = epoch_start_time - time.time()

def plot_data(data, label, title):
    plt.plot(data, label = label)
    plt.title(title)
    plt.legend()
    plt.show()

plot_data(loss_history, "Train", "Loss history iteration")
plot_data(loss_history_epoch, "Train", "Loss history epoch")
# plot_data(acc_composition_epoch, "Train", "accuracy composition")
# plot_data(acc_echoginicity_epoch, "Train", "accuracy echoginicy")
# plot_data(acc_shape_epoch, "Train", "accuracy shape")
# plot_data(acc_margin_epoch, "Train", "accuracy margin")
# plot_data(acc_macrocal_epoch, "Train", "accuracy macrocalcification")
# plot_data(acc_peripheral_epoch, "Train", "accuracy peripheral calcification")
# plot_data(acc_microcal_epoch, "Train", "accuracy microcalcification")
# plot_data(acc_comet_epoch, "Train", "accuracy comet-tail")
plot_data(acc_malignant_epoch, "Train", "accuracy benign/malignant")

plot_data(loss_history_test, "Test", "Loss history iteration")
plot_data(loss_history_epoch_test, "Test", "Loss history epoch")
# plot_data(acc_composition_epoch_test, "Test", "accuracy composition")
# plot_data(acc_echoginicity_epoch_test, "Test", "accuracy echoginicy")
# plot_data(acc_shape_epoch_test, "Test", "accuracy shape")
# plot_data(acc_margin_epoch_test, "Test", "accuracy margin")
# plot_data(acc_macrocal_epoch_test, "Test", "accuracy macrocalcification")
# plot_data(acc_peripheral_epoch_test, "Test", "accuracy peripheral calcification")
# plot_data(acc_microcal_epoch_test, "Test", "accuracy microcalcification")
# plot_data(acc_comet_epoch_test, "Test", "accuracy comet-tail")
plot_data(acc_malignant_epoch_test, "Test", "accuracy benign/malignant")

