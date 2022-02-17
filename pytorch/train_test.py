from __future__ import print_function
from __future__ import division

import argparse
import glob
import sys, os
import math
import timm
from sklearn import metrics
from sklearn.preprocessing import label_binarize

import gc
import random
from datetime import datetime
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

from utils import CB_loss,patient_aggregation
from typing import Dict, Any

sys.path.insert(0, '../..')
import imageio
import h5py

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import girder_client
from skimage.transform import resize
from matplotlib import pylab as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler


def train_valid_model(net, configs=None, optimizer=None,dataloaders=None, criterion=None, a_device=False):

    save_model_name = 'best_model'
    best_path = os.path.join(configs["best_dir"], save_model_name + '.pt')

    best_metric = 0.0
    best_pa_auc = 0
    no_of_classes = configs['no_of_classes']
    beta = configs['beta']
    gamma = configs['gamma']

    loss_type = "focal"
    for epoch in range(configs["epochs"]):

        #for debugging usage only
        #for phase in ['val']:

        if epoch % 1 == 0:
            PHASES = [ 'train','val']

        else:
            PHASES =['train']


        for phase in PHASES:
            print('Phase {}'.format(phase))
            if phase == 'train':
                net.train()  # Set model to training mode
                #adjust_learning_rate(optimizer, epoch, configs['lr'], configs['epochs'], num_cycles=5)
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            itertor = iter(dataloaders[phase])
            total_step = len(dataloaders[phase])
            print(total_step)

            # record corrects for each class
            # init two dicts

            total_dict = {i:0 for i in range(no_of_classes)}
            hit_dict = {i: 0 for i in range(no_of_classes)}

            patient_all = []
            predictions_all = []
            label_all = []

            for step in range(total_step):

                imgs, labels, patients = next(itertor)
                imgs = imgs.to(a_device)

                total_dict = {i: total_dict[i]+labels.tolist().count(i) for i in range(no_of_classes)}

                labels = labels.to(device=a_device, dtype=torch.int64)
                # reset parameter gradients for training
                optimizer.zero_grad()
                # forward propagation
                logps = net(imgs)



                # caluclate loss
                samples_per_cls = [labels.tolist().count(i) for i in range(no_of_classes)]

                if configs["opt"] == "CE" or phase == 'val':
                    loss1 = criterion(logps, labels)
                if configs["opt"] == "CB" and phase == 'train' :
                    cb_loss = CB_loss(labels, logps, samples_per_cls, no_of_classes, loss_type, beta, gamma, device= a_device)
                    loss1 = cb_loss
                loss = loss1

                if step % 50 == 0:
                    print(samples_per_cls)
                    print("Total Loss {:.4f}, Loss1 {:.4f} ".format(loss, loss1))

                if phase == 'train':
                    # back propagation
                    loss.backward()
                    # update model parameters
                    optimizer.step()

                # Update running statistics
                running_loss += loss.item() * imgs.size(0)
                probs = torch.nn.functional.softmax(logps, dim=1) # probabilities

                # Running count of correctly identified classes
                ps = torch.exp(logps)
                _, predictions = ps.topk(1, dim=1)  # top predictions
                equals = predictions == labels.view(*predictions.shape)

                if len(predictions_all) == 0:
                    predictions_all = ps.detach().cpu().numpy()
                    label_all = labels.tolist()
                    patient_all = list(patients)
                    probs_all = probs.detach().cpu().numpy()
                else:
                    predictions_all = np.vstack((predictions_all, ps.detach().cpu().numpy()))
                    probs_all = np.vstack((probs_all, probs.detach().cpu().numpy()))
                    label_all.extend(labels.tolist())
                    patient_all.extend(list(patients))



                all_hits = equals.view(equals.shape[0]).tolist()  # get all T/F indices
                all_corrects = labels[all_hits]

                hit_dict = {i: hit_dict[i]+all_corrects.tolist().count(i) for i in range(no_of_classes)}
                running_corrects += torch.sum(equals.type(torch.FloatTensor)).item()

            # Calculate phase statistics
            phase_loss = running_loss / sum(total_dict.values())
            phase_acc = running_corrects / sum(total_dict.values())


            y_true = label_all.copy()
            y_pred = np.argmax(predictions_all, axis=1)
            metrics.confusion_matrix(y_true, y_pred)
            print(metrics.classification_report(y_true, y_pred, digits=3))

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            if no_of_classes == 2:
                fpr, tpr, _ = metrics.roc_curve(y_true, probs_all[:,1])
                roc_auc[0] = metrics.auc(fpr, tpr)
                if phase == 'val':
                    print("Binary AUC: {:.4f}".format(roc_auc[0]))

                roc_auc["micro"] = roc_auc[0]
            else:
                y_true = label_binarize(np.array(y_true), classes=range(no_of_classes))
                for i in range(no_of_classes):
                    fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:,i], probs_all[:,i])
                    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
                    if phase == 'val':
                        print("Class {} AUC: {:.4f}".format(i, roc_auc[i]))


                fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), probs_all.ravel())
                roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

            #####

            if phase == 'val':
                print("Micro Tile AUC: {:.4f}".format(roc_auc["micro"]))
                if no_of_classes ==2:
                    pa_auc = patient_aggregation(patient_all, probs_all, label_all,binary=True)

                else:
                    pa_auc = patient_aggregation(patient_all,probs_all,label_all)

            print("\n".join("class: {}\t counts: {}".format(k, v) for k, v in total_dict.items()))
            print("\n".join("class: {}\t counts: {}".format(k, v) for k, v in hit_dict.items()))

            # Save best model weights if accuracy improved
            if configs["metric"] == "AUC":
                # tile AUC as metric
                cur_metric = roc_auc["micro"]

            elif configs['metric'] == "LOSS": # use the minimal validation loss as a determiner.
                cur_metric = phase_loss

            else:
                cur_metric = phase_acc

            cur_model_path= os.path.join(configs["best_dir"], save_model_name + f'_e{epoch}.pt')

            cur_model_state_dict = copy.deepcopy(net.state_dict())
            #                 best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            torch.save(cur_model_state_dict, cur_model_path)

            if "du" in configs['name']:
                if phase == 'val' and cur_metric > best_metric and pa_auc> best_pa_auc:
                    print("Micro +Patient auc")
                    now2 = datetime.datetime.now()
                    print("Saving model date and time : ")
                    print(now2.strftime("%Y-%m-%d %H:%M:%S"))

                    best_metric = cur_metric
                    best_pa_auc = pa_auc
                    best_model_state_dict = copy.deepcopy(net.state_dict())
                    #                 best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
                    torch.save(best_model_state_dict, best_path)
                    print("at Epoch {}, model updated".format(epoch))

            else:
                if phase == 'val' and cur_metric > best_metric:
                    print("Only micro auc")
                    now2 = datetime.datetime.now()
                    print("Saving model date and time : ")
                    print(now2.strftime("%Y-%m-%d %H:%M:%S"))

                    best_metric = cur_metric
                    best_pa_auc = pa_auc
                    best_model_state_dict = copy.deepcopy(net.state_dict())
                    #                 best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
                    torch.save(best_model_state_dict, best_path)
                    print("at Epoch {}, model updated".format(epoch))



            print('PHASE {} Loss: {:.4f} Acc: {:.4f}'.format(phase, phase_loss, phase_acc))

    net.load_state_dict(best_model_state_dict)

    return net


def test_best_model(net, no_of_classes = 2, testloader= None, a_device = False,SAVE_CSV_PATH=None):
    itertor = iter(testloader)
    total_step = len(testloader)

    print(total_step)
    net.eval()
    patient_all = []
    predictions_all = []
    label_all = []
    running_corrects = 0

    total_dict = {i: 0 for i in range(no_of_classes)}
    hit_dict = {i: 0 for i in range(no_of_classes)}

    for step in range(total_step):
        if step % 50 ==0:
            print("Current Step {}".format(step))
        imgs, labels, patients = next(itertor)

        imgs = imgs.to(a_device)
        labels = labels.to(device=a_device, dtype=torch.int64)
        logps = net(imgs)
        total_dict = {i: total_dict[i] + labels.tolist().count(i) for i in range(no_of_classes)}

        probs = torch.nn.functional.softmax(logps, dim=1)
        # Running count of correctly identified classes
        ps = torch.exp(logps)
        _, predictions = ps.topk(1, dim=1)  # top predictions

        equals = predictions == labels.view(*predictions.shape)

        if len(predictions_all) == 0:
            predictions_all = ps.detach().cpu().numpy()
            label_all = labels.tolist()
            patient_all = list(patients)
            probs_all = probs.detach().cpu().numpy()
        else:
            try:
                predictions_all = np.vstack((predictions_all, ps.detach().cpu().numpy()))
                probs_all = np.vstack((probs_all, probs.detach().cpu().numpy()))
                label_all.extend(labels.tolist())
                patient_all.extend(list(patients))
            except:
                print(ps.detach().cpu().numpy().shape)

        all_hits = equals.view(equals.shape[0]).tolist()  # get all T/F indices
        all_corrects = labels[all_hits]

        hit_dict = {i: hit_dict[i] + all_corrects.tolist().count(i) for i in range(no_of_classes)}
        running_corrects += torch.sum(equals.type(torch.FloatTensor)).item()

    phase_acc = running_corrects / sum(total_dict.values())

    y_true = label_all.copy()
    y_pred = np.argmax(predictions_all, axis=1)
    print("accuray:", len(np.arange(len(y_true))[y_true == y_pred]) / len(y_true))
    metrics.confusion_matrix(y_true, y_pred)
    print(metrics.classification_report(y_true, y_pred, digits=3))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if no_of_classes == 2:
        fpr, tpr, _ = metrics.roc_curve(y_true, probs_all[:, 1])
        roc_auc[0] = metrics.auc(fpr, tpr)
        print("Binary AUC: {:.4f}".format(roc_auc[0]))
        roc_auc["micro"] = roc_auc[0]

        print("Micro Tile AUC: {:.4f}".format(roc_auc["micro"]))
    else:
        y_true = label_binarize(np.array(y_true), classes=range(no_of_classes))
        for i in range(no_of_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], probs_all[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            print("Class {} AUC: {:.4f}".format(i, roc_auc[i]))

        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), probs_all.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])



    if no_of_classes == 2:
        pa_auc = patient_aggregation(patient_all, probs_all, label_all, binary=True,SAVE_PATH=SAVE_CSV_PATH)
    else:
        pa_auc = patient_aggregation(patient_all, probs_all, label_all,SAVE_PATH=SAVE_CSV_PATH)

    print("\n".join("class: {}\t counts: {}".format(k, v) for k, v in total_dict.items()))
    print("\n".join("class: {}\t counts: {}".format(k, v) for k, v in hit_dict.items()))
    print('Acc: {:.4f} PA-AUC {:.4f}'.format(phase_acc, pa_auc))