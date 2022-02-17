from __future__ import print_function
from __future__ import division

import argparse
import glob
import sys, os
import math
import timm
import pandas as pd

import gc
import random
from datetime import datetime
import numpy as np
import time
import copy
from typing import Dict, Any

sys.path.insert(0, '../..')
import imageio
import h5py

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

from train_test import train_valid_model,test_best_model
from loader import create_train_loader,create_test_loader


def main(args):

    best_dir_path = args.savekey
    out_dim = args.outdim
    hdf5_path = args.path
    loss_func = args.loss
    ins_bal_flag = args.balance # conduct instance balance sampling or class-balance
    omiga = args.omiga
    name_prefix = args.namemarker
    val_metric = args.metric
    MODEL_PATH = args.modelpath
    TEST_STATUS = args.onlytest
    VALID_STATUS = args.revalid
    MODEL_CHOICE = args.model
    CSV_PATH = args.csv
    optim = args.optimizer
    epoch_numb = args.end_epoch
    batch_size = args.batch
    partition_seed = args.namemarker.split("-")[-1]

    lr = args.lr



    print("input args:::", args)

    device_lst = list(range(args.device))
    device_str = ",".join(str(e) for e in device_lst)
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str

    criterion = nn.CrossEntropyLoss()
    config = {"model_name": "ViT", 'opt': loss_func, "lr": lr, "batch_size": batch_size,
              "best_dir": best_dir_path, "report_dir": './reports/new',
              'beta': 0.9999, 'gamma': 1.0, 'omiga': omiga, 'no_of_classes': out_dim, "name": name_prefix,
              'metric': val_metric,'epochs':epoch_numb,'optimizer': optim,'h5_path':hdf5_path,'in_csv_path':CSV_PATH,'set_par_seed': partition_seed}

    if MODEL_CHOICE =='CNN':
        model = timm.create_model("densenet121", pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, out_dim)
        config['model_name'] = 'Dense121'

    elif MODEL_CHOICE =='SWIN':
        config['model_name'] = 'ViT-Swin'
        model = timm.create_model("swin_base_patch4_window7_224_in22k", pretrained=True)

        if "freeze" in name_prefix :
            for param in model.parameters():
                param.requires_grad = False
        model.head = nn.Linear(model.head.in_features, out_dim)

        for param in model.parameters():
            print(param.requires_grad)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=device_lst)


    if not TEST_STATUS: #train models

        if "tune" in name_prefix :
            model.load_state_dict(torch.load(MODEL_PATH))
            print("LOAD User-input Model Weights... Start tuning the model")
            print("Only compute the gradients for FC layer")
            if MODEL_CHOICE =='SWIN':
                for name, param in model.named_parameters():
                    if "head" not in name:
                        param.requires_grad = False

            print("Valid Params Update")
            if MODEL_CHOICE =='SWIN':
                for name, param in model.named_parameters():
                    print(name)
                    print(param.requires_grad)

            model.to(device)

        else:
            model.to(device)
            print("LOAD pre-trained ImageNet Model Weights")


        if optim =='sgd':
            optimizer = torch.optim.SGD([
                {'params': list(model.parameters())[:-1], 'lr': config["lr"], 'momentum': 0.9, 'weight_decay': 1e-4},
                {'params': list(model.parameters())[-1], 'lr': config["lr"], 'momentum': 0.9, 'weight_decay': 1e-4}
            ])


        elif optim =='adam':
            optimizer = torch.optim.Adam(model.parameters(), config["lr"], weight_decay=1e-4)

        print('start training process!')
        loaders = create_train_loader(config,ins_bal_flag)

        best_model = train_valid_model(net=model, configs=config, optimizer=optimizer,
                                       dataloaders=loaders, criterion=criterion, aug_times=aug_times, aug_class= False, const =const,a_device = device)

    else: # ONLY TEST A MODEL
        print("VALIDATION STATUS IS {}".format(VALID_STATUS))
        if VALID_STATUS:
            TEST_SET_NAME = 'val'
        else:
            TEST_SET_NAME = 'test'

        print('Assemble Test Data Loader')

        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(device)

        test_loader = create_test_loader(config,TEST_SET_NAME)
        test_best_model(net=model, testloader= test_loader, no_of_classes= args.outdim,a_device = device,SAVE_CSV_PATH=args.stats)

    return 0




parser = argparse.ArgumentParser(description='Configurations')

parser.add_argument('--savekey',
                    help='key for saving best model',
                    default='/Mu_bestmodel/4classWHO', type=str)

parser.add_argument('--onlytest', action='store_true', default=False, help='Only test the best model on testset')

parser.add_argument('--revalid', action='store_true', default=False, help='Re-test the best model on validation set')

parser.add_argument('--loss',
                    help='loss function: CE or CB',
                    default='CE', type=str)

parser.add_argument('--path',
                    help='path for hdf5 data',
                    default='/data/par1.hdf5', type=str)

parser.add_argument('--stats',
                    help='path for exporting best csv',
                    default='logs/output.csv', type=str)

parser.add_argument('--csv',
                    help='path for csv data',
                    default='/partition_csv/', type=str)

parser.add_argument('--outdim', type=int, help='how many classes to predict', default=3)

parser.add_argument('--namemarker',
                    help='string key for saving best model',
                    default='', type=str)

parser.add_argument('--omiga', type=float, help='temperature weight', default=1)

parser.add_argument('--modelpath',
                    help='path for pre-trained model data',
                    default='', type=str)

parser.add_argument('--model',help='CNN model or Vision Transformer',default='CNN', type=str)

parser.add_argument('--optimizer',help='Adam or SGD as optimm',default='sgd', type=str)

parser.add_argument('--augtimes', type=int, help='how many times the selected classes need to augment', default=0)

parser.add_argument('--metric', help='metric for selecting the best model on validation: AUC score or accuracy', default='AUC', type=str)

parser.add_argument('--balance', action='store_true', default=False, help='Default False is class-balance , set True to use instance-balance')

parser.add_argument('--batch', type=int, help='how large is a batch', default=128)

parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)

parser.add_argument('--end-epoch', type=int, help='max epoch', default=200)

parser.add_argument('--device', type=int, help='how many devices', default=4)


args = parser.parse_args()


if __name__ == "__main__":

    print("started!")
    import datetime

    now = datetime.datetime.now()
    print("Starting date and time : ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    print("Starting Execution Time:", now)
    results = main(args)
    print("finished!")
    print("end script")