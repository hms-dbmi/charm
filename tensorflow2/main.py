#!/usr/bin/env python
# coding: utf-8

# In[7]:


from utils import *
import h5py
import numpy as np
import pandas as pd
import math
import os
import argparse

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import label_binarize


def train_model_on_partition(hdf5_path, ds_name, model_4tune=None, best_idx=-1, configs=configs):
    target = configs['target']
    batch_size = configs['batch_size']
    
    # encode images into a feature space
    print("Embedding...")
    timer.on()
    h5file = h5py.File(hdf5_path, "r")
    #all_img_embs = encoder.predict(generator(h5file["img"], h5file[f"{target}Label"], batch_size=batch_size),
    #                               steps= math.ceil(h5file['img'].shape[0]/batch_size))
    
    all_img_embs = np.load('temp.npy')
    
    all_pts = h5file['pt_name'][()].astype(str)
    all_labels = h5file[f'{target}Label'][()].astype(int)
    
    # compute class weights of loss
    compute_class_weights(all_labels, configs)
    
    # one hot if dealing with multiple classes
    configs['no_class'] = all_labels.max()+1
    if configs['no_class']>2:
        all_labels = label_binarize(all_labels, classes=[0, 1, 2])
    timer.off()
    
    
    # train models on different patient partitions (for standard deviation)
    perf_dict = create_perf_dict()
    for par in range(5):
        
        # load patients' data based on partitions into a dictionary 
        print(f"Processing {par}th partition...")
        par_file_path = f"./data/patient_partition/{task_name}_{ds_name}_par{par}.csv"
        train_pt, val_pt, test_pt = get_ptList(par_file_path)
        data_dict = load_data(train_pt, val_pt, test_pt, all_img_embs, all_labels, target, all_pts)

        # settings for new models
        model_path = f"{res_dir}/{ds_name}_model_par{par}.h5"
        perf_dict['best_score'] = 0
        
        # hyper parameter optimization
        for count, lr_choice in enumerate([0.1, 0.01, 0.001] * 3):
            print(f'Hyperparameter optimization {count+1}/9, learning rate = {lr_choice}')
            if not model_4tune:
                model = create_tlie_head(lr_choice, configs['no_class'])
            else:
                best_model_path = f"{res_dir}/{model_4tune}_model_par{best_idx}.h5"
                model = load_model(best_model_path, compile=False)
                model.compile(optimizer=Adam(lr=lr_choice), loss='binary_crossentropy', metrics=['AUC', 'accuracy'])
            train_model(model, configs, data_dict, perf_dict, model_path)

        # Evaluation
        model = load_model(model_path)  # best head
        preds = model.predict(data_dict['test_img'])
        ptTruth, ptPreds, _ = tile_aggregation(preds, 
                                               data_dict[f'test_{target}Label'], 
                                               data_dict[f'test_pt_name'], configs['no_class'])
        par_auc, par_acc = evaluation(ptTruth, ptPreds, no_class=configs['no_class'])
        perf_dict["pt_auc"].append(par_auc)
        perf_dict["pt_acc"].append(par_acc)
        if 'TMB' in target:
            computer_tmb_prob(model, data_dict, par, configs)
    
    best_model_idx = report_perf(perf_dict, ds_name)
    return best_model_idx


# for parser

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--dstrain', 
                    help='name of the primary dataset used to train models', 
                    default="TCGA", 
                    type=str)
parser.add_argument('--dstest', 
                    help='independent dataset used to test models', 
                    default="", 
                    type=str)
parser.add_argument('--target', 
                    help='target to predict. could be cancer/GBM/IDH/molClass/TMB', 
                    default='cancer', 
                    type=str)
parser.add_argument('--subgroup', help='LGG or GBM if the data is stratified', default="", type=str)
parser.add_argument('--batch', help='batch size', default=128, type=int)
parser.add_argument('--device', help='GPU(s) to use', default=0, type=int)

args = parser.parse_args()


def main(args):
    configs=dict()
    configs['batch_size'] = args.batch
    configs['subgroup'] = args.subgroup
    ds_train = args.dstrain
    ds_test = args.dstest
    
    # Cause for TMB, we need to train the models in two separate steps (TBMClass and measured TMB)
    input_target = args.target
    target = 'TMBClass' if input_target=='TMB' else input_target
    configs['target'] = target
    
    task_map = {'cancer':"cancerVsBenign", 'GBM':"LGGvsGBM", "IDH":'IDHdetection',
               'molClass':"molClass", "TMBClass":'TMBRegression'}
    task_name = task_map[target] + f'_{subgroup}' if subgroup else task_map[target]
    res_dir = f"./model/{task_name}"

    train_hdf5_path = f"./data/{task_name}_{ds_train}.hdf5"
    test_hdf5_path = f"./data/{task_name}_{ds_test}.hdf5"

    print("input args:::", args)
    
    # set up GPU(s)
    device_lst = list(range(args.device))
    device_str = ",".join(str(e) for e in device_lst)
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str
    
    create_folder()
    encoder = build_encoder()
    timer = Timer()

    # use primary dataset to train models
    best_idx = train_model_on_partition(train_hdf5_path, ds_name=ds_train)
    
    # if any, use a portion of data in the independent dataset to tune above models before testing
    if target in ['GBM', 'IDH']:
        print("Tuning the models")
        _ = train_model_on_partition(test_hdf5_path, ds_name=ds_test, model_4tune=ds_train, best_idx=best_idx)
        
    return 0


if __name__ == "__main__":

    print("started!")
    results = main(args)
    print("finished!")

