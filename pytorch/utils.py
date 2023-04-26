from __future__ import print_function
from __future__ import division

import numpy as np
from PIL import Image
import glob
import sys, os
from sklearn import metrics

from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets, models, transforms
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from scipy import interp

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.functional import kl_div, softmax, log_softmax
import matplotlib.pyplot as plt
import copy
from typing import Dict, Any

sys.path.insert(0, '../..')
import pandas as pd

import pandas as pd
import requests
import zipfile
import imageio
import h5py
from datetime import datetime

import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
import torchvision
from torchvision import datasets, models, transforms

from PIL import Image
from matplotlib import pylab as plt

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device = False):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    if 0 in samples_per_cls: # some classes have 0 samples in this batch
        samples_per_cls = [x if x != 0 else 1 for x in samples_per_cls]

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss


class Meta_Dataset:
    def __init__(self, task='CancerVsBenign',db = 'BWH',
                 all_path='', seed=0,
                 pt_lst_csv='',entire=False, set='train'):
        self.task = task
        self.db = db
        self.all_path = all_path
        self.pt_lst_csv = pt_lst_csv
        self.seed = seed
        self.entire = entire
        self.set_name = set

        self.get_pt_lst()

    def get_pt_lst(self):  # read patient lists from CSV
        CSV_PATH = self.pt_lst_csv + f"{self.task}_{self.db}_par{self.seed}.csv"
        print("Reading {} SET from CSV: {}".format(self.set_name,CSV_PATH))
        df = pd.read_csv(CSV_PATH)
        self.target_pt = df[df['par'] == self.set_name]['Patient ID']

    def get_data(self, pt_list):  # extract data from all-set using patient lists
        h5file = h5py.File(self.all_path, "r")
        entire_pts = h5file['pt_name'][()].astype(str)
        if self.task == "cancerVsBenign":
            if self.set_name != 'train' and self.db != 'BWH':
                labels = h5file['GBMLabel'][()]
            else:
                labels = h5file['CancerLabel'][()]
        elif self.task == 'LGGvsGBM':
            labels = h5file['GBMLabel'][()]
        elif self.task == 'IDHdetection_LGG' or self.task == 'IDHdetection_GBM':
            labels = h5file['IDHLabel'][()]
        elif self.task == 'TMBRegression_LGG' or self.task == 'TMBRegression_GBM':
            labels = h5file['TMBClassLabel'][()]
        elif self.task == 'MGMTdetection_LGG' or self.task == 'MGMTdetection_GBM':
            labels = h5file['MGMTLabel'][()]
        elif self.task == 'molClass':
            labels = h5file['molClassLabel'][()]
        elif self.task == 'molClass601' or self.task == 'molClass3':
            labels = h5file['molClassLabel'][()]


        index_list = []
        img_embs = h5file['img']

        # print(pt_list[:10])
        for pt in pt_list:
            idx = np.where(entire_pts == pt)[0]
            index_list.extend(idx)
        index_list = list(index_list)

        img = []
        for numb, a_idx in enumerate(index_list):
            img.append(img_embs[a_idx])

        label = labels[index_list]
        pt_code = entire_pts[index_list]
        img = np.array(img)

        return img, label, pt_code

    def create_Meta(self, sep=False):

        data = dict()
        print('reading data from separated h5 files by patients? {}'.format(sep))

        if sep:
            all_imgs = []
            all_pas = []
            all_labels = []

            for pt in self.target_pt:
                file = self.all_path + pt + '.h5'
                h5file = h5py.File(file, "r")
                all_imgs.append(np.array(h5file['img']))
                all_pas.extend([pt] * len(h5file['img']))
                all_labels.append(h5file['GBMLabel'][()])


            data[f'{self.set_name}_img'] = np.concatenate((all_imgs), axis=0)
            data[f'{self.set_name}_pt_name'] = np.array(all_pas)
            data[f'{self.set_name}_label'] = np.concatenate((all_labels), axis=0)

            print('Loading images done')

            if self.task == "cancerVsBenign" and self.set_name =='test' and self.db == 'TCGA':

                print('Testing TCGA as CanBe external set, assign all test labels to 1')

                data[f'{self.set_name}_label'] = np.ones(data[f'{self.set_name}_label'].shape)


        else:
            if not self.entire:
                print('Take a partition from the entire set')

                data[f'{self.set_name}_img'], data[f'{self.set_name}_label'], data[f'{self.set_name}_pt_name'] = self.get_data(self.target_pt)
            else:
                print('Take the entire set')
                h5file = h5py.File(self.all_path, "r")

                if self.task == "cancerVsBenign":
                    data['test_img'] = np.array(h5file['img'])
                    print('entire image reading done...')
                    data['test_pt_name'] = h5file['pt_name'][()].astype(str)

                    if self.set !='train':
                        data['test_label'] = [1]*len(h5file['pt_name'][()].astype(str))
                    else:
                        data['test_label'] = h5file['CancerLabel'][()]
                    print('entire data reading done...')

        return data

class HDF5Dataset_Meta(Dataset):

    def __init__(self, path, set_name,seed,job='cancerVsBenign',db = 'BWH',entire=False,csv_path=None,pat_sep = False):
        self.file_path = path
        self.dataset = None
        self.set = set_name
        self.seed = seed
        self.job = job
        self.db = db

        print('Reading csv from {}'.format(csv_path))
        if entire:
            self.data = Meta_Dataset(all_path=self.file_path, seed=self.seed, task=job, db=db,entire=True,pt_lst_csv=csv_path).create_Meta(sep=pat_sep)
        else:
            # self.data only contains specific set, e.g. train_set
            self.data = Meta_Dataset(all_path=self.file_path, seed=self.seed,task=job,db=db,set=self.set,pt_lst_csv=csv_path).create_Meta(sep=pat_sep)

        self.dataset_len = len(self.data[self.set + "_label"])

        self.train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        if self.dataset is None:
            self.imgs = self.data[self.set + "_img"]
            self.labels = self.data[self.set + "_label"]
            self.patients = self.data[self.set + "_pt_name"]
            cur_img = self.imgs[index]
            PIL_image = Image.fromarray(np.uint8(cur_img)).convert('RGB')

            if self.set == 'train':
                img = self.train_transform(PIL_image)
            else:
                img = self.val_transform(PIL_image)

            if self.job == 'cancerVsBenign' and self.db == 'TCGA':
                label = float(1)
            else:
                label = self.labels[index].astype('float32')
            patient = self.patients[index]
        return (img, label, patient)

    def __len__(self):

        return self.dataset_len

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)
    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device = False):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    if 0 in samples_per_cls: # some classes have 0 samples in this batch
        samples_per_cls = [x if x != 0 else 1 for x in samples_per_cls]

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss


def patient_aggregation(ll_pa,all_pred,all_lab,binary=False,best_pa_auc=0,SAVE_PATH=None):

    a = np.expand_dims(ll_pa, axis=1)
    b = np.expand_dims(all_lab, axis=1)

    record_auc = {}
    record_pred = {}

    if binary:
        all_results = pd.DataFrame(data=np.hstack((a, b, all_pred)),
                                   columns=["patient_id", "label", "logits_0", "logits_1"])
        patient_stats = {}
        fea_pa_agg = {}

        for p_name in ["p50"]:

            patient_stats[p_name] = {}
            fea_pa_agg[p_name] = {}
            record_auc[p_name] = {}
            record_pred[p_name] = {}

            for one_pa in all_results["patient_id"].unique():

                sub_df = all_results[all_results["patient_id"] == one_pa]

                # TODO: here we use different percentiles to check patient auc
                # 50%
                p50 = int(sub_df.shape[0] * 0.5)  # median index for extracting row

                patient_stats[p_name][one_pa] = {}
                fea_pa_agg[p_name][one_pa] = {}


                for sort_key in ["logits_0", "logits_1"]:

                    sorted = sub_df.sort_values(by=[sort_key])
                    pert =  [p50][["p50"].index(p_name)]
                    # record into stats dictionary

                    fea_pa_agg[p_name][one_pa][sort_key] = {'label': int(sub_df["label"].iloc[0]), 'index': sorted.iloc[pert].name}

                    patient_stats[p_name][one_pa][sort_key] = {"outputs": sorted.iloc[pert][2:],
                                                       "pred": np.argmax(sorted.iloc[pert][2:]),
                                                       'label': int(sub_df["label"].iloc[0])}

            # np.save('patient_stats_tcga4_bp.npy', patient_stats)
            for key in ["logits_0", "logits_1"]:
                out_auc,out_preds,out_probs,out_paname,out_labels = key_aggreg_print_metric(patient_stats[p_name], key, binary=binary,percentile = p_name)
                record_auc[p_name][key] = out_auc
                record_pred[p_name][key] = [out_preds,out_probs,out_paname,out_labels]

        res = {key: max(val.values()) for key, val in record_auc.items()}
        max_value = max(res.values())
        max_pert = max(res, key=res.get)
        max_pa = record_auc[max_pert]
        max_key = max(max_pa, key=max_pa.get)

        save_df = pd.DataFrame()
        save_df["patient_id"] = record_pred[max_pert][max_key][2].tolist()
        # save_df["pred"] = record_pred[max_pert][max_key][0].tolist()
        save_df["prob1"] = record_pred[max_pert][max_key][1][:, 1].tolist()
        save_df["label"] = record_pred[max_pert][max_key][3].tolist()

        save_df.to_csv(SAVE_PATH)

        print("BEST Patient AUC SCORE: {} ".format(max_value))

    else:
        all_results = pd.DataFrame(data=np.hstack((a, b, all_pred)),
                                   columns=["patient_id", "label", "logits_0", "logits_1", "logits_2"])

        patient_stats = {}
        fea_pa_agg = {}

        for p_name in ["p50"]:

            patient_stats[p_name] = {}
            fea_pa_agg[p_name] = {}
            record_auc[p_name] = {}
            record_pred[p_name] = {}

            for one_pa in all_results["patient_id"].unique():

                sub_df = all_results[all_results["patient_id"] == one_pa]
                # 50%
                p50 = int(sub_df.shape[0] * 0.5)  # median index for extracting row

                patient_stats[p_name][one_pa] = {}
                fea_pa_agg[p_name][one_pa] = {}

                for sort_key in ["logits_0", "logits_1", "logits_2"]:

                    sorted = sub_df.sort_values(by=[sort_key])
                    pert = [p50][["p50"].index(p_name)]

                    fea_pa_agg[p_name][one_pa][sort_key] = {'label': int(sub_df["label"].iloc[0]), 'index': sorted.iloc[pert].name}

                    patient_stats[p_name][one_pa][sort_key] = {"outputs": sorted.iloc[pert][2:],
                                                       "pred": np.argmax(sorted.iloc[pert][2:]),
                                                       'label': int(sub_df["label"].iloc[0])}

            for key in ["logits_0", "logits_1", "logits_2"]:

                out_auc,out_preds,out_probs,out_paname,out_labels = key_aggreg_print_metric(patient_stats[p_name], key, binary=binary, percentile=p_name)
                record_auc[p_name][key] = out_auc
                record_pred[p_name][key] = [out_preds, out_probs, out_paname, out_labels]

        res = {key: max(val.values()) for key, val in record_auc.items()}
        max_value = max(res.values())
        max_pert = max(res, key=res.get)
        max_pa = record_auc[max_pert]
        max_key = max(max_pa, key=max_pa.get)

        save_df = pd.DataFrame()
        save_df["Patient ID"] = record_pred[max_pert][max_key][2].tolist()
        save_df["class0"] = record_pred[max_pert][max_key][1][:, 0].tolist()
        save_df["class1"] = record_pred[max_pert][max_key][1][:, 1].tolist()
        save_df["class2"] = record_pred[max_pert][max_key][1][:, 2].tolist()
        save_df["truth"] = record_pred[max_pert][max_key][3].tolist()

        save_df.to_csv(SAVE_PATH)

        print("BEST Patient AUC SCORE: {} ".format(max_value))



    return max_value






def key_aggreg_print_metric(patient_stats, key, binary = False, save_fig= False,percentile='p50',best_pa_auc=0):


    pa_preds = []
    pa_labels = []
    pa_outputs = []
    pa_names = []

    for patient in patient_stats.keys():
        pa_preds.append(patient_stats[patient][key]["pred"])
        pa_labels.append(patient_stats[patient][key]["label"])
        pa_outputs.append(patient_stats[patient][key]["outputs"])
        pa_names.append(patient)

    pa_preds = np.array(pa_preds)
    pa_labels = np.array(pa_labels)
    pa_outputs = np.array(pa_outputs).astype(float)
    pa_names = np.array(pa_names)



    # print patient names with true predictions
    print_TP_TN(pa_names,pa_preds,pa_labels)

    y_true = pa_labels

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Compute ROC curve and ROC area for each class
    y_true = pa_labels

    y_pred = np.argmax(pa_outputs, axis=1)


    PA_ACCU = len(np.arange(len(y_true))[y_true == y_pred]) / len(y_true)
    print("Patient Accuracy:", PA_ACCU)


    if binary:

        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        print('Use Key {} at P{}'.format(key, percentile))
        print("Patient Matric using {}".format(key), sep="\n")
        print("TN: {}, FP:{}: FN:{}, TP:{}".format(tn, fp, fn, tp))

        idx = int(key.split("_")[-1])
        fpr, tpr, thresholds = metrics.roc_curve(y_true, pa_outputs[:, 1])
        print(pa_outputs[:, 1])
        #fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        print("Patient AUC SCORE:", roc_auc)
        if roc_auc > best_pa_auc:
            # print class 0 for LGG or print class 1 probs for GBM
            a = np.bincount(y_true)
            target_c = np.argmax(a)
            # print("POI probs are:", sep="\n")

        if save_fig:
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")


            plt.savefig(datetime.now().strftime("%H:%M:%S"))



        return roc_auc,pa_preds,pa_outputs,pa_names,pa_labels

    else:

        y_oh = label_binarize(y_true, classes=[0, 1, 2])  # one-hot encoding
        n_classes = 3

        tn1, fp1, fn1, tp1,tn2, fp2, fn2, tp2,tn3, fp3, fn3, tp3 = metrics.multilabel_confusion_matrix(y_true, y_pred).ravel()
        print('Use Key {} at P{}'.format(key, percentile))
        print("Patient Metric using {}".format(key), sep="\n")
        print("TN1: {}, FP1:{}: FN1:{}, TP1:{}".format(tn1, fp1, fn1, tp1))
        print("TN2: {}, FP2:{}: FN2:{}, TP2:{}".format(tn2, fp2, fn2, tp2))
        print("TN3: {}, FP3:{}: FN3:{}, TP3:{}".format(tn3, fp3, fn3, tp3))

        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_oh[:, i], pa_outputs[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_oh.ravel(), pa_outputs.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])


        return roc_auc["micro"],pa_preds,pa_outputs,pa_names,pa_labels



def print_TP_TN (patient_lst, preds_lst, labels_lst):
    TN_lst = np.where(labels_lst == 0)
    TP_lst = np.where(labels_lst == 1)
    P_lst = np.where(preds_lst == labels_lst)[0]

    TNs = patient_lst[np.intersect1d(TN_lst,P_lst)]
    TPs = patient_lst[np.intersect1d(TP_lst, P_lst)]

    print("TN patients: {}".format(TNs))
    print("TP patients: {}".format(TPs))