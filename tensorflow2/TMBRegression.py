#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
def train_test_svm(X_train, X_test, y_train, y_test):
    sc_X = StandardScaler().fit(X_train)
    sc_y = StandardScaler().fit(y_train[:,np.newaxis])

    X_train_scl = sc_X.transform(X_train)
    X_test_scl = sc_X.transform(X_test)
    y_train_scl = sc_y.transform(y_train[:,np.newaxis])

    reg = SVR(kernel = 'linear')
    reg.fit(X_train_scl, y_train_scl.ravel())
    y_pred_scl = reg.predict(X_test_scl)
    y_preds = sc_y.inverse_transform(y_pred_scl)
    
    rmse, r2, pear_r, spear_r = reg_evaluation(y_test, y_preds)
    return rmse, r2, pear_r, spear_r


from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr 
def reg_evaluation(truth, preds):
    truth, preds = truth.squeeze(), preds.squeeze()
    rmse = np.sqrt(mean_squared_error(truth, preds)).round(3)
    r2 = r2_score(truth, preds).round(3)
    pear_r = pearsonr(truth, preds)
    spear_r = spearmanr(truth, preds)
    return rmse, r2, pear_r[0], spear_r[0]

def get_regression_data(feat_csv_path, df_meta):
    try:
        df_img_feat = pd.read_csv(feat_csv_path)
    except:
        print("Please do TMB classification first and get feature vectors before doing regression")
    df_final = df_img_feat.merge(df_meta, how='left', on='Patient ID')
    df_final = df_final[['Patient ID', 'par', 'class0', 'class1', 'class2', 
                         "Diagnosis Age", "Sex", 'measured TMB']]
    df_final = df_final.dropna()

    df_train = df_final[df_final['par']!='test']
    X_train = np.array(df_train[['class0', 'class1', 'class2', "Diagnosis Age", "Sex"]])
    y_train = np.array(df_train['measured TMB'])

    df_test = df_final[df_final['par']=='test']
    X_test = np.array(df_test[['class0', 'class1', 'class2', "Diagnosis Age", "Sex"]])
    y_test = np.array(df_test['measured TMB'])
    return X_train, X_test, y_train, y_test


parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--subgroup', 
                    help='subgroup for TMB regression', 
                    default="LGG", 
                    type=str)

parser.add_argument('--csv', 
                    help='csv path for acquiring metadata', 
                    default="./data/TMB feature/TMBRegression_metadata.csv", 
                    type=str)

args = parser.parse_args()


def main(args):
    subgroup     = args.subgroup
    metadata_path = args.csv

    df_meta = pd.read_csv(metadata_path)
    spear_r_arr = []
    for par in range(5):
        feat_csv_path = f"./data/TMB feature/TMB_{subgroup}_feature_par{par}.csv"
        X_train, X_test, y_train, y_test = get_regression_data(feat_csv_path, df_meta)
        _, _, _, spear_r = train_test_svm(X_train, X_test, y_train, y_test)
        spear_r_arr.append(spear_r)
    spear_r_arr = np.array(spear_r_arr)

    print('Spearman r:',spear_r_arr.mean().round(2),'±',np.std(spear_r_arr).round(2))
    return 0

if __name__ == "__main__":

    print("started!")
    results = main(args)
    print("finished!")

