import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timm
import torch
import numpy as np
import gc
import os
import time
import random
from datetime import datetime
import shutil
from PIL import Image
from tqdm.notebook import tqdm
from sklearn import model_selection, metrics
from shutil import copyfile
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
import torch


class Affinity_Loss(nn.Module):
    def __init__(self, lambd):
        super(Affinity_Loss, self).__init__()
        self.lamda = lambd
        # self.batch_size = batch_size
    def forward(self, y_pred_plusone, y_true_plusone):
        # print("y_true_plusone :", y_true_plusone.shape)
        # print("y_pred_plusone :", y_pred_plusone.shape)
        y_true_plusone = torch.unsqueeze(y_true_plusone, 1)  # Divs
        size = len(y_true_plusone)
        # print("After changing y_true_plusone :", y_true_plusone.shape)
        x = torch.nn.Softmax(dim=1)
        y_pred_plusone = x(y_pred_plusone)
        tensor_add = torch.zeros([size, 2], dtype=torch.int32)
        tensor_add = tensor_add.numpy()
        g = 0

        for a in y_true_plusone:
            if a == [0]:
                s = [1, 0]
                tensor_add[g] = s
            else:
                s = [0, 1]
                tensor_add[g] = s
            g = g + 1

        y_true_plusone= tensor_add
        y_true_plusone = torch.from_numpy(y_true_plusone)

        onehot = y_true_plusone[:, :-1]

        # print("y_true_plusone :", y_true_plusone.shape)
        # print("y_pred_plusone :", y_pred_plusone.shape)
        # print("onehot :", onehot.shape)
        distance = y_pred_plusone[:, :-1]
        # print("distance :", distance.shape)
        rw = torch.mean(y_pred_plusone[:, -1])
        # print("rw=torch.mean(y_pred_plusone[:,-1])   :", rw.shape)
        d_fi_wyi = torch.sum(onehot * distance, -1).unsqueeze(1)
        # print("d_fi_wyi   :", d_fi_wyi.shape)
        losses = torch.clamp(self.lamda + distance - d_fi_wyi,
                             min=0)  # Divs torch.clamp(d_fi_wyi-self.lamda+distance,min=0)
        # print("losses after torch clamp", losses.shape)
        # print("y_true_plusone :",y_true_plusone.shape)
        L_mm = torch.sum(losses * (1.0 - onehot), -1) / y_true_plusone.size(0)  # Divs 1.0-onehot
        # print(" L_mm+rw", (torch.sum(L_mm + rw).item()))
        loss = torch.sum(L_mm + rw, -1)  # Divs loss=torch.sum(L_mm+rw,-1)
        return loss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("Saving the best model !")
        shutil.copyfile(filename, 'model_affinity_best_sixray10_transformer.pth.tar')


def load_data(data_folder, batch_size, phase='train', train_val_split=True, train_ratio=.8):
    transform_dict = {
        'train': transforms.Compose(
            [
                transforms.Resize([384, 384]),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                                   std=[0.229, 0.224, 0.225]),
            ]),
        'test': transforms.Compose(
            [
                transforms.Resize([384, 384]),
                transforms.ToTensor(),
                #              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                                   std=[0.229, 0.224, 0.225]),

            ])}

    data = datasets.ImageFolder(root=data_folder, transform=transform_dict[phase])

    if phase == 'train':
        if train_val_split:
            train_size = int(train_ratio * len(data))
            print("train_size : ", train_size)
            test_size = len(data) - train_size
            print("test_size : ", test_size)
            data_train, data_val = torch.utils.data.random_split(data, [train_size, test_size])
            train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True,
                                                       num_workers=1)
            val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, drop_last=False,
                                                     num_workers=1)
            return [train_loader, val_loader]
        else:
            train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                       num_workers=1)
            print("train_size : ", len(data))
            return train_loader
    else:
        test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False,
                                                  num_workers=1)
        print("test_size : ", len(data))
        return test_loader


class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_small_patch16_384", pretrained)
        # self.model = timm.create_model("vit_large_patch16_224", pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)  # Classification head

    def forward(self, x):
        x = self.model(x)
        return x

        # VIT model with patch32


class ViTBase32(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase32, self).__init__()

        self.model = timm.create_model("vit_small_patch32_384", pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)  # Classification head

    def forward(self, x):
        x = self.model(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs, _ = torch.max(inputs, -1)
        targets = targets.to(torch.float32)
        inputs = inputs.to(torch.float32)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_folder = r'C:\Users\fta71\PycharmProjects\pythonProject2\SIXray10\Training'
    test_folder = r'C:\Users\fta71\PycharmProjects\pythonProject2\SIXray10\Training'

    train_loader = load_data(train_folder, 8, phase='train', train_val_split=False)
    val_loader = load_data(test_folder, 8, phase='test', train_val_split=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    print(len(train_loader))
    print(len(val_loader))

    Sixray10_data = datasets.ImageFolder(train_folder)
    print(Sixray10_data.classes)

    model = ViTBase16(n_classes=2, pretrained=True)

    model = model.to(device)

    train_loss_array = []
    train_acc_array = []
    val_loss_array = []
    val_acc_array = []
    best_acc1 = 0
    best_f1 = 0

    LR = 2e-5
    epochs = 40
    check_every = 100
    #criterion = FocalLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    # #optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0,
    #                                 initial_accumulator_value=0, eps=1e-10)

    for epoch in range(epochs):
        print("I am going for training")
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        i = 0
        for counter, (data, target) in enumerate(train_loader):
            i += 1
            data, target = data.to(device, dtype=torch.float32), target.to(device,
                                                                           dtype=torch.int64)  # load data to device
            # clear the gradients of all optimizable variables
            optimizer.zero_grad()
            # compute outputs by passing input to the model
            output = model(data)
            criterion = Affinity_Loss(0.32)
            # print("Inside run",output.shape)
            # print("Inside run",target.shape)
            # the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Calculating accuracy
            accuracy = (output.argmax(dim=1) == target).float().mean()

            # update training loss and accuracy
            epoch_loss += loss
            epoch_accuracy += accuracy
            optimizer.step()

            if (i % check_every == 0) or (i ==(len(train_loader))):
                # keep track of validation loss
                valid_loss = 0.0
                valid_accuracy = 0.0
                pred = []
                g_t = []
                with torch.no_grad():
                    model.eval()
                    for data, target in val_loader:
                        data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.int64)
                        output = model(data)
                        loss = criterion(output, target)
                        pred.extend(output.argmax(dim=1))
                        g_t.extend(target)
                        accuracy = (output.argmax(dim=1) == target).float().mean()
                        # update average validation loss and accuracy
                        valid_loss += loss
                        valid_accuracy += accuracy

                # Score transfer to CPU
                valid_loss_cpu = valid_loss.cpu().detach().numpy()
                valid_accuracy_cpu = valid_accuracy.cpu().detach().numpy()
                epoch_loss_cpu = epoch_loss.cpu().detach().numpy()
                epoch_accuracy_cpu = epoch_accuracy.cpu().detach().numpy()

                val_loss_array.append(valid_loss_cpu / len(val_loader))
                val_acc_array.append(valid_accuracy_cpu / len(val_loader))
                train_loss_array.append(epoch_loss_cpu / (counter + 1))
                train_acc_array.append(epoch_accuracy_cpu / (counter + 1))
                pred_array = [int(i.cpu().detach().numpy()) for i in pred]
                g_t_array = [int(i.cpu().detach().numpy()) for i in g_t]
                f1 = f1_score(g_t_array, pred_array)
                print(
                    "[{} epoch {} batch] Train Loss: {:.3f}\t Acc: {:.3f}\t Valid loss: {:.3f}\t Valid Acc: {:.3f}\t F1: {:.3f}".format(
                        epoch + 1,
                        i,
                        epoch_loss_cpu / (counter + 1),
                        epoch_accuracy_cpu / (counter + 1),
                        valid_loss_cpu / len(val_loader),
                        valid_accuracy_cpu / len(val_loader),
                        f1))
                val_loss = valid_loss / len(val_loader)
                acc1 = valid_accuracy / len(val_loader)
                is_best = f1 > best_f1
                best_f1 = max(f1, best_f1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best)

                # Learning rate scheduler step
            #             scheduler.step(val_loss)
            model.train()

    print("Finish Training!")
