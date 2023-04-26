#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

import time
class Timer:
    def __init__(self):
        self.start_time = time.time()
        print("Create a new timer")
    def on(self):
        self.start_time = time.time()
        print("Start the timer")
    def off(self):
        print("--- %s seconds ---" % (time.time() - self.start_time))

        
import os
def create_folder():
    print("Creating new folders")
    for f in ['./data/patient_partition', './data/TMB feature']:
        try: os.makedirs(f)
        except: pass
    
    for task in ['cancerVsBenign', 'LGGvsGBM', 'molClass']:
        try: os.makedirs(f'./model/{task}')
        except: pass
        
    for task in ['IDHdetection', 'TMBRegression']:
        for subgroup in ['LGG','GBM']:
            try: os.makedirs(f'./model/{task}_{subgroup}')
            except: pass
            

import pandas as pd
def get_ptList(par_file_path):
    df = pd.read_csv(par_file_path)
    train_pt = df[df['par']=='train']['Patient ID']
    val_pt = df[df['par']=='val']['Patient ID']
    test_pt = df[df['par']=='test']['Patient ID']
    return train_pt, val_pt, test_pt


from sklearn.utils import shuffle
def get_data_from_ptList(pt_list, all_img_embs, all_labels, all_pts, sub=0):
    '''
    Find patients' image embeddings, labels in the arrays based on the location of patients' names
    '''
    index_list = [] 
    for pt in pt_list:
        # Find the location of a patient's name
        idx = np.where(all_pts == pt)[0]
        index_list.extend(idx)
    
    # Extract corresponding features
    img = all_img_embs[index_list]
    label = all_labels[index_list]
    pt_code = all_pts[index_list]
    
    #shuffle
    if sub=='train':
        img, label, pt_code = shuffle(img, label, pt_code)
    return img, label, pt_code


def load_data(train_pt, val_pt, test_pt, all_img_embs, all_labels, target, all_pts):
    data = dict()
    for sub, pt_list in zip(["train", "val", "test"], [train_pt, val_pt, test_pt]):
        data[f'{sub}_img'], data[f'{sub}_{target}Label'], data[f'{sub}_pt_name'] =        get_data_from_ptList(pt_list, all_img_embs, all_labels, all_pts, sub)
    return data


from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
def build_encoder():
    data_shape = (224, 224, 3)
    base_model = EfficientNetB5(weights='imagenet', include_top=False, input_shape=data_shape)
    print("Now is using EfficientNetB5")

    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    enc = Model(base_model.input, x, name='encoder')
    return enc


def generator(img, labels, batch_size):
    datasetLength = labels.shape[0]
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < datasetLength:
            limit = min(batch_end, datasetLength)
            X = img[batch_start:limit]
            Y = labels[batch_start:limit]                
            yield (X, Y)
            batch_start += batch_size   
            batch_end += batch_size

            
def create_tlie_head(lr_choice, no_class=2):
    inputs = Input(shape=(2048,))
    
    if no_class==2:
        x = Dense(1024, activation='relu')(inputs) # buffer layer
        x = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        met = ['AUC', 'accuracy']
    else:
        x = Dense(16, activation='relu')(inputs) # buffer layer
        x = Dense(3, activation='softmax')(x)
        loss = 'categorical_crossentropy'
        met = ["categorical_crossentropy","AUC"]
        
    model = Model(inputs, x, name='decision_head')
    model.compile(optimizer=Adam(lr=lr_choice), loss=loss, metrics=met)
    return model


def compute_class_weights(all_labels, configs):
    # note: use the number of patients might be better than the number of slides
    n2 = np.sum(all_labels==2)
    n1 = np.sum(all_labels==1)
    n0 = np.sum(all_labels==0)
    
    if n2==0:
        class_weights = {0: n1/(n1+n0), 1: n0/(n1+n0)}
    elif 'TMB' in configs['target']:
        class_weights = {0: 0.33, 1: 0.33, 2: 0.33}
    else:
        n0, n1, n2 = 89, 86, 17
        n_all = n0+n1+n2
        class_weights = {0: n_all/n0, 1: n_all/n1, 2: n_all/n2}
    configs['class_weights'] = class_weights

            
def train_model(model, configs, data_dict, perf_dict, save_path):
    target = configs['target'] 
    for epochs in range(8):
        result = model.fit(data_dict[f"train_img"], data_dict[f"train_{target}Label"],
                           batch_size=configs['batch_size'],
                           epochs=1,
                           verbose=1,
                           validation_data=(data_dict["val_img"], data_dict[f"val_{target}Label"]),
                           class_weight=configs['class_weights'],
                           shuffle=True) 
        
        # Early stopping
        if ((result.history['auc'][0] - result.history['val_auc'][0])> 0.2) & (epochs>2):
            break
        
        score = result.history['val_auc'][0]
        if score > perf_dict['best_score']:
            print("Save new model!")
            perf_dict['best_score'] = score
            model.save(save_path)


def tile_aggregation(tile_preds, tile_labels, pt_names, no_class=2):
    if no_class==2:
        df = pd.DataFrame({"Patient ID":pt_names,
                           "preds":tile_preds.ravel(),
                           "truth":tile_labels
                          })
        df = df.groupby("Patient ID").median().reset_index(drop=False)

        ptTruth = df['truth']
        ptPreds = df['preds']
        return ptTruth, ptPreds, df
    else:
        agg = dict()
        for name, tile in zip(['truth', 'preds'], [tile_labels, tile_preds]):
            df = pd.DataFrame({"Patient ID":pt_names,
                               "class0":tile[:,0],
                               "class1":tile[:,1],
                               "class2":tile[:,2]})  
            df = df.groupby('Patient ID').quantile(0.5).reset_index(drop=False)
            agg[name] = np.array(df.drop(["Patient ID"], axis=1))

        ptTruth, ptPreds = agg['truth'], agg['preds']
        return ptTruth, ptPreds, df


def create_perf_dict():
    dic = dict()
    for name in ['pt_auc', 'pt_acc', 'best_score']:
        dic[name] = []
    return dic


def report_perf(perf_dict, ds_name):
    auc_vec = np.array(perf_dict['pt_auc'])
    acc_vec = np.array(perf_dict['pt_acc'])
    print(f"On {ds_name}")
    print(f"AUC: {auc_vec.mean().round(2)} ± {round(np.std(auc_vec),3)}")
    print(f"acc: {acc_vec.mean().round(2)} ± {round(np.std(acc_vec),3)}")
    return auc_vec.argmax()
    
    
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
def evaluation(y_true, y_pred, no_class=2, auc_type='micro', show=True):
    if no_class==2:
        try:
            auc_score = round(roc_auc_score(y_true, y_pred),2)
        except:
            auc_score = float("nan")
        y_dec = np.array([1 if i>=0.5 else 0 for i in y_pred])
        acc_score = round(accuracy_score(y_true, y_dec), 2)
    else:
      # y_true: int, shape = (n, 3); 
      # y_pred: float, shape = (n, 3)

      # acc_score
        acc_score = round(accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1)), 2)

      # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(no_class):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

      # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

      # Compute macro-average ROC curve and ROC area
      # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(no_class)]))

      # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(no_class):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

      # Finally average it and compute AUC
        mean_tpr /= no_class
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        auc_score = roc_auc["micro"] if auc_type=='micro' else roc_auc["macro"] 
        
    if show:
        print('patient level AUC:', auc_score)
        print("patient level acc:", acc_score)
    return auc_score, acc_score

    
def computer_tmb_prob(model, data_dict, par, configs):
    df_tmb = []
    subgroup = configs['subgroup']
    for sub in ['train','val','test']:
        preds = model.predict(data_dict[f'{sub}_img'])
        _, _, df = tile_aggregation(preds, 
                                    data_dict[f'{sub}_TMBClassLabel'], 
                                    data_dict[f'{sub}_pt_name'], configs['no_class'])
        df['par'] = sub
        df_tmb.append(df)
    df_final = pd.concat(df_tmb).reset_index(drop=True)
    df_final.to_csv(f'./data/TMB feature/TMB_{subgroup}_feature_par{par}.csv', index=False)
   
