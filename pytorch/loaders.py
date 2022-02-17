from __future__ import print_function
from __future__ import division

import argparse
import math
import pandas as pd

from datetime import datetime
import numpy as np
import time
import copy

from utils import HDF5Dataset_Meta
from typing import Dict, Any

sys.path.insert(0, '../..')
import imageio
import h5py
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler


def create_test_loader(config,TEST_SET_NAME,cur_job=None,cur_db=None):
    if 'CanBe' in config['name']:
        cur_job = 'cancerVsBenign'
        if 'ex' in config['name']:
            cur_db = 'TCGA'
        else:
            cur_db = 'BWH'

    elif 'IDH' in config['name']:
        if "LGG" in config['name']:
            subset = 'LGG'
        else:
            subset = 'GBM'

        cur_job = f'IDHdetection_{subset}'

        if "BP" in config['name']:
            cur_db = 'BP'
        else:
            cur_db = 'TCGA'
    elif 'LgHg' in config['name']:
        cur_job = 'LGGvsGBM'
        if "BP" in config['name']:
            cur_db = 'BP'
        else:
            cur_db = 'TCGA'

    elif 'TMB' in config['name']:
        if "LGG" in config['name']:
            subset = 'LGG'
        else:
            subset = 'GBM'
        cur_job = f'TMBRegression_{subset}'
        if "tune" in config['name']:
            cur_db = 'BP'
        else:
            cur_db = 'TCGA'

    elif 'MOLE' in config['name']:
        cur_job = 'molClass'
        if "BP" in config['name']:
            cur_db = 'BP'
        else:
            cur_db = 'TCGA'

    print('Current Job: {}, Current Data: {}'.format(cur_job, cur_db))
    test_loader = DataLoader(HDF5Dataset_Meta(path=config['hdf5_path'], set_name=TEST_SET_NAME, seed=config['set_par_seed'], job=cur_job,db=cur_db, csv_path=config['in_csv_path']),batch_size=config['batch_size'], shuffle=True,drop_last=False)

    return test_loader



def create_train_loader (config,ins_bal_flag,cur_job=None,cur_db=None):
    entire_pts = []

    if 'CanBe' in config['name']:

        cur_job = 'cancerVsBenign'
        cur_db = 'BWH'

    elif 'LgHg' in config['name']:

        cur_job = 'LGGvsGBM'
        if "BP" in config['name']:
            cur_db = 'BP'
            print('Fine-tune the TCGA model with B+P data')
        else:
            cur_db = 'TCGA'
            print('Low-High Grade TEST')

    elif 'IDH' in config['name']:
        if "LGG" in config['name']:
            subset = 'LGG'
        else:
            subset = 'GBM'

        cur_job = f'IDHdetection_{subset}'

        if "tune" in config['name']:
            cur_db = 'BP'
        else:
            cur_db = 'TCGA'

    elif 'TMB' in config['name']:
        if "LGG" in config['name']:
            subset = 'LGG'
        else:
            subset = 'GBM'

        cur_job = f'TMBRegression_{subset}'

        if "tune" in config['name']:
            cur_db = 'BP'
        else:
            cur_db = 'TCGA'


    elif 'MOLE' in config['name']:

        if '3' in config['name']:
            cur_job = 'molClass3'
        else:
            cur_job = 'molClass'

        if "TCGA" in config['name']:
            cur_db = 'TCGA'
        elif "BWH" in config['name']:
            cur_db = 'BWH'
            print('Fine-tune the TCGA model with BWH')
        elif "BP" in config['name']:
            cur_db = 'BP'
            print('Fine-tune the TCGA model with B+P data')
        else:
            cur_db = 'TCGA'
            print('Low-High Grade TEST')

        print('Current Job: {}, Current Data: {}'.format(cur_job, cur_db))

        A_CSV = config['in_csv_path'] + f"{cur_job}_{cur_db}_par{config['set_par_seed']}.csv"
        print(A_CSV)
        df = pd.read_csv(A_CSV)

        train_pt = df[df['par'] == 'train']['Patient ID']
        val_pt = df[df['par'] == 'val']['Patient ID']

        # read h5 data

        h5file = h5py.File(config['hdf5_path'], "r")
        if 'IDH' in config['name']:
            labels = h5file['IDHLabel'][()]
            entire_pts = h5file['pt_name'][()].astype(str)
        elif 'CanBe' in config['name']:
            labels = h5file['CancerLabel'][()]
            entire_pts = h5file['pt_name'][()].astype(str)
        elif 'LgHg' in config['name']:
            labels = h5file['GBMLabel'][()]
            entire_pts = h5file['pt_name'][()].astype(str)
        elif 'TMB' in config['name']:
            labels = h5file['TMBClassLabel'][()]
            entire_pts = h5file['pt_name'][()].astype(str)
        elif 'MGMT' in config['name']:
            labels = h5file['MGMTLabel'][()]
            entire_pts = h5file['pt_name'][()].astype(str)
        elif 'MOLE' in config['name']:
            labels = h5file['molClassLabel'][()].astype(int)
            entire_pts = h5file['pt_name'][()].astype(str)

        train_index_list = []
        val_index_list = []

        for pt in train_pt:
            idx = np.where(entire_pts == pt)[0]
            train_index_list.extend(idx)

        for pt in val_pt:
            idx = np.where(entire_pts == pt)[0]
            val_index_list.extend(idx)

        train_targets = labels[train_index_list].tolist()

    class_sample_counts = [train_targets.count(i) for i in range(out_dim)]

    ## instance balance
    class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = class_weights[train_targets]

    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    print('Current Job: {}, Current Data: {}'.format(cur_job, cur_db))
    if ins_bal_flag:
        train_loader = DataLoader(
            HDF5Dataset_Meta(path=config['hdf5_path'], set_name="train", seed=config['set_par_seed'], job=cur_job, db=cur_db,
                             csv_path=config['in_csv_path']), batch_size=config['batch_size'], shuffle=True, drop_last=False)
        config['balance'] = "instance"
    else:
        train_loader = DataLoader(
            HDF5Dataset_Meta(path=config['hdf5_path'], set_name="train", seed=config['set_par_seed'], job=cur_job, db=cur_db,
                             csv_path=config['in_csv_path']), batch_size=config['batch_size'], drop_last=False, sampler=sampler)
        config['balance'] = "class"

    print('Train loader done')
    val_loader = DataLoader(
        HDF5Dataset_Meta(path=config['hdf5_path'], set_name="val", seed=config['set_par_seed'], job=cur_job, db=cur_db,
                         csv_path=config['in_csv_path']), batch_size=config['batch_size'],
        drop_last=False, shuffle=True)

    print('Validation loader done')

    loaders = {"train": train_loader, "val": val_loader}

    return loaders