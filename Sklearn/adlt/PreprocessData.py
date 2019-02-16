#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:01:57 2018

@author: macgx
"""
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.model_selection import train_test_split
import numpy as np
import arff
import os
#%%
current_script_path = os.path.dirname(os.path.abspath(__file__)).split('/')
# os.path.dirname does not include file name
current_script_path.pop()
current_script_path[-1] = 'UCI_data_arff'
datapath = '/'.join(current_script_path)


def preprocess(dataname):
    if dataname == 'adult':
        with open('{}/adult_train.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[14] = data[14].astype('int64')
        
        # with open('{}/adult_test.arff'.format(datapath), 'r') as f2:
        #     data_test = arff.load(f2)
        #     data_test = pd.DataFrame(data_test['data'])
        #     data_test[14] = data_test[14].astype('int64')
        
        # data = data.append(data_test, ignore_index=True)
        X = data[data.columns[:14]]
        y = data[data.columns[14]]
        return X, y
    
    elif dataname == 'pima':
        with open('{}/pima.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[8] = data[8].astype('int64')

            X = data[data.columns[:8]]
            y = data[data.columns[8]]

        return X, y
    
    elif dataname == 'bank':
        with open('{}/bank.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[16] = data[16].astype('int64')
               
        X = data[data.columns[:16]]
        y = data[data.columns[16]]
        return X, y  
    
    elif dataname == 'car':
        # Characteristics: has missing value
        with open('{}/car.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[6] = data[6].astype('int64')
        
        X = data[data.columns[:6]]
        y = data[data.columns[6]]
        return X, y
    
    elif dataname == 'ches':
        with open('{}/chess-krvk.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[6] = data[6].astype('int64')
        
        X = data[data.columns[:6]]
        y = data[data.columns[6]] 
        return X, y       
    
    elif dataname == 'letter':
        with open('{}/letter.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[16] = data[16].astype('int64')
        
        X = data[data.columns[:16]]
        y = data[data.columns[16]]
        return X, y 
    
    elif dataname == 'magic':
        with open('{}/magic.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[10] = data[10].astype('int64')
        
        X = data[data.columns[:10]]
        y = data[data.columns[10]]
        return X, y 
    
    elif dataname == 'musk2':
        with open('{}/musk-2.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[166] = data[166].astype('int64')
        
        X = data[data.columns[:166]]
        y = data[data.columns[166]]
        return X, y
    
    elif dataname == 'page_blocks':
        with open('{}/page-blocks.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[10] = data[10].astype('int64')
        
        X = data[data.columns[:10]]
        y = data[data.columns[10]]
        return X, y

    elif dataname == 'semeion':
        with open('{}/semeion.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[256] = data[256].astype('int64')
        
        X = data[data.columns[:256]]
        y = data[data.columns[256]]       
        return X, y
    
    elif dataname == 'spam':
        with open('{}/spambase.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[57] = data[57].astype('int64')
        
        X = data[data.columns[:57]]
        y = data[data.columns[57]]    
        return X, y
    
    elif dataname == 'german_credit':
        with open('{}/statlog-german-credit.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[24] = data[24].astype('int64')
        
        X = data[data.columns[:24]]
        y = data[data.columns[24]]
        return X, y
    
    elif dataname == 'image':
        with open('{}/statlog-image.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[18] = data[18].astype('int64')
        
        X = data[data.columns[:18]]
        y = data[data.columns[18]]
        return X, y
    
    elif dataname == 'shuttle':
        #Approximately 80% of the data belongs to class 1. 
        #Therefore the default accuracy is about 80%. 
        #The aim here is to obtain an accuracy of 99 - 99.9%. 
        with open('{}/statlog-shuttle_train.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[9] = data[9].astype('int64')
        
        X = data[data.columns[:9]]
        y = data[data.columns[9]]     
        return X, y
    
    elif dataname == 'steel':
        with open('{}/steel-plates.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[27] = data[27].astype('int64')
        
        X = data[data.columns[:27]]
        y = data[data.columns[27]]  
        return X, y
    
    elif dataname == 'titanic':
        with open('{}/titanic.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[3] = data[3].astype('int64')
        
        X = data[data.columns[:3]]
        y = data[data.columns[3]] 
        return X, y
    
    elif dataname == 'thyroid':
        with open('{}/thyroid_train.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[21] = data[21].astype('int64')
        
        X = data[data.columns[:21]]
        y = data[data.columns[21]]       
        return X, y
    
    elif dataname == 'wine':
        with open('{}/wine-quality-red.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            ## convert str to int for target variable, manually modify target column index
            data[11] = data[11].astype('int64')
        
        X = data[data.columns[:11]]
        y = data[data.columns[11]] 
        return X, y

    elif dataname == '46':
        data = pd.read_csv('{}/splice_46.csv'.format(datapath), header=0, sep=',')
        data.drop(['Instance_name'], axis=1, inplace=True)
    
        ### my preprocessing strategy: convert categorical variables to [-1, 1]
        ### and standardize all attributes including previous converted-categorical 
    #    attribute_encoder = {'R': -1, 'S': -5 / 7, 'D': -3 / 7, 'N': 0, 'A': 1 / 7, 'G': 3 / 7, 'T': 5 / 7, 'C': 1}
    #    target_encoder = {'N': 0, 'EI': 1, 'IE': 2}
    #    for col in data.columns[:-1]:
    #        data[col] = data[col].map(attribute_encoder)      
    #    data['Class'] = data['Class'].map(target_encoder)
    #    # standardize all attributes
    #    data[data.columns[:-1]] = scale(data[data.columns[:-1]])
    
        ### paper author's strategy: convert categorical variables to integer
        ### no standardization
        attribute_encoder = {'R': 1, 'S': 2, 'D': 3, 'N': 4, 'A': 5, 'G': 6, 'T': 7, 'C': 8}
        target_encoder = {'N': 0, 'EI': 1, 'IE': 2}
        for col in data.columns[:-1]:
            data[col] = LabelEncoder().fit_transform(data[col])
            
        data['Class'] = data['Class'].map(target_encoder)
        
        #TODO: another strategy: data[col] = data[col].map(attribute_encoder)    
        X = data[data.columns[:-1]]
        y = data['Class']
        return X, y

    # TODO: URL ID 184 is kropt, which is same with UCI chess-krvk
    # URL ID 180 but datafile 184 is covertype.txt
    elif dataname == '184':
        ## OpenML dataset 184 is same as UCI chess-krvk
        ## paper author's strategy: no standardization
        with open('{}/dataset_188_kropt.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            
        attribute_encoder = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8}
        for col in data.columns[:6]:
            if col in [1,3,5]: ## convert str to int 
                data[col] = data[col].astype('int64')
            else:
                data[col] = data[col].map(attribute_encoder)    
        y = LabelEncoder().fit_transform(data[6])
        X = data[data.columns[:6]]        
        return X, y

    elif dataname == '389':
        with open('{}/fbis.wc.arff'.format(datapath), 'r') as f1:
            data = arff.load(f1)
            data = pd.DataFrame(data['data'])
            y = LabelEncoder().fit_transform(data[data.columns[-1]])
            X = data[data.columns[:-1]]
#            X = scale(data[data.columns[:-1]])
        return X, y

    elif dataname == '772':
        # Characteristics: no missing value
        data = pd.read_csv('{}/quake_772.csv'.format(datapath), header=0, sep=',')
#        X = scale(data[data.columns[:-1]])
        X = data[data.columns[:-1]]
        y = LabelEncoder().fit_transform(data[data.columns[-1]])
        return X, y

    elif dataname == '917':
        data = pd.read_csv('{}/fri_c1_1000_25_917.csv'.format(datapath), header=0, sep=',')
#        X = scale(data[data.columns[:-1]])
        X = data[data.columns[:-1]]
        y = LabelEncoder().fit_transform(data[data.columns[-1]])
        return X, y
    # TODO: not sure it's 1049
    elif dataname == '1049':
        # Characteristics: all attributes are numeric, binary class
        data = pd.read_csv('{}/pc4_1049.csv'.format(datapath), header=0, sep=',')
        y = data['c'].map({True: 1, False: 0})
#        X = scale(data[data.columns[:-1]])
        X = data[data.columns[:-1]]
        return X, y


# %%
if __name__ == '__main__':
    datapath = '/Users/macgx/Documents/fall2018/test3_mipego/UCI_data_arff'  
      
    #Approximately 80% of the data belongs to class 1. 
    #Therefore the default accuracy is about 80%. 
    #The aim here is to obtain an accuracy of 99 - 99.9%. 
    with open('{}/statlog-shuttle_train.arff'.format(datapath), 'r') as f1:
        data = arff.load(f1)
        data = pd.DataFrame(data['data'])
        ## convert str to int for target variable, manually modify target column index
        data[9] = data[9].astype('int64')
    
    X = data[data.columns[:9]]
    y = data[data.columns[9]] 
    # %%
    # =============================================================================
    # test modeling X, y
    # =============================================================================
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from collections import OrderedDict

    #    datasets_UCI = ['adult', 'pima', 'bank', 'car', 'ches', 'letter', 'magic',
    #                    'musk2', 'page_blocks', 'semeion', 'spam', 'german_credit',
    #                    'image', 'shuttle', 'steel', 'titanic', 'thyroid', 'wine',
    #                    '46','184','389', '772', '917', '1049']

    # TODO: run it

    datasets_UCI = ['772']
    for dataname in datasets_UCI:
        X, y = preprocess(dataname=dataname)

        error = OrderedDict()

        clf_knn = KNeighborsRegressor()
        accuracy_knn = cross_val_score(clf_knn, X, y, cv=5)
        error['knn'] = 1 - np.mean(accuracy_knn)

        clf_dt = DecisionTreeClassifier()
        accuracy_dt = cross_val_score(clf_dt, X, y, cv=5)
        error['dt'] = 1 - np.mean(accuracy_dt)

        clf_rf = RandomForestClassifier()
        accuracy_rf = cross_val_score(clf_rf, X, y, cv=5)
        error['rf'] = 1 - np.mean(accuracy_rf)

        clf_svm = SVC()  # default kernel =’rbf’
        accuracy_svm = cross_val_score(clf_svm, X, y, cv=5)
        error['svm'] = 1 - np.mean(accuracy_svm)

        clf_linsvm = LinearSVC()
        accuracy_linsvm = cross_val_score(clf_linsvm, X, y, cv=5)
        error['linsvm'] = 1 - np.mean(accuracy_linsvm)

        clf_adab = AdaBoostClassifier()
        accuracy_adab = cross_val_score(clf_adab, X, y, cv=5)
        error['adab'] = 1 - np.mean(accuracy_adab)

        clf_qda = QuadraticDiscriminantAnalysis()
        accuracy_qda = cross_val_score(clf_qda, X, y, cv=5)
        error['qda'] = 1 - np.mean(accuracy_qda)

        with open('../default_clf_perf.txt', 'a') as f2:
            f2.write('{},{} \n'.format(dataname, error.values()))


