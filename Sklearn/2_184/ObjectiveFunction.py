#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:09:24 2018

@author: macgx
"""

import os
import csv
import pandas as pd
import numpy as np
import zipfile
from abc import ABCMeta, abstractmethod

from mipego import mipego
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

#preprocess
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype, is_numeric_dtype
# model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# model assessment
from sklearn.model_selection import cross_val_score

from PreprocessData import preprocess


dirs = ['pim', 'ches', 'car', 'ltr', 'mgic', 'msk',
'p-blk', 'adlt', 'bnk','sem', 'spam', 's-gc',
's-im', 's-sh', 's-pl', 'tita', 'thy', 'wine',
'1_46', '2_184', '3_389', '4_772', '5_917', '6_1049']

datas = ['pima', 'ches', 'car', 'letter', 'magic', 'musk2',
'page_blocks', 'adult', 'bank','semeion', 'spam', 'german_credit',
'image', 'shuttle', 'steel', 'titanic', 'thyroid', 'wine',
'46', '184', '389', '772', '917', '1049']

dir_mapping = dict(zip(dirs, datas))

parent_path = os.path.dirname(os.path.abspath(__file__)).split('/')
## parse and extract parent directory names, to determine dataset name
dataname = dir_mapping[parent_path[-1]]

X, y = preprocess(dataname=dataname)

#TODO: what if model performance metric is ROC, f1 score

def rf_obj_func(cfg): # give cfg list in mipego
    clf = RandomForestClassifier(**cfg)#
    accuracy = cross_val_score(clf, X, y, cv=5)
    error = 1 - np.mean(accuracy)
    return error

def dt_obj_func(cfg):
    # ['max_depth_dt', 'min_samples_split_dt', 'min_samples_leaf_dt']
    # rename key
    cfg['max_depth'] = cfg.pop('max_depth_dt')
    cfg['min_samples_split'] = cfg.pop('min_samples_split_dt')
    cfg['min_samples_leaf'] = cfg.pop('min_samples_leaf_dt')
    clf = DecisionTreeClassifier(**cfg)
    accuracy = cross_val_score(clf, X, y, cv=5)
    error = 1 - np.mean(accuracy)
    return error

def svm_obj_func(cfg):
    cfg['C'] = 1 if cfg['C']  == 0 else cfg['C']
    cfg['gamma'] = 1 if cfg['gamma'] == 0 else cfg['gamma']

    cfg['C'] = 2 ** cfg['C']
    cfg['gamma'] = 2 ** cfg['gamma']

    clf = SVC(**cfg) # default kernel =’rbf’
    accuracy = cross_val_score(clf, X, y, cv=5)
    error = 1 - np.mean(accuracy)
    return error

def linsvm_obj_func(cfg):
    cfg['C_lin'] = 1 if cfg['C_lin'] == 0 else cfg['C_lin']
    cfg['C'] = cfg.pop('C_lin')
    cfg['C'] = 2 ** cfg['C']
    clf = LinearSVC(**cfg)
    accuracy = cross_val_score(clf, X, y, cv=5)
    error = 1 - np.mean(accuracy)
    return error

def knn_obj_func(cfg):
    clf = KNeighborsRegressor(**cfg)
    accuracy = cross_val_score(clf, X, y, cv=5)
    error = 1 - np.mean(accuracy)
    return error

def adab_obj_func(cfg):
    cfg['n_estimators'] = cfg.pop('n_estimators_adab')
    clf = AdaBoostClassifier(**cfg)
    accuracy = cross_val_score(clf, X, y, cv=5)
    error = 1 - np.mean(accuracy)
    return error

def qda_obj_func(cfg):
    cfg['reg_param'] = 1 if cfg['reg_param'] == 0 else cfg['reg_param']
    cfg['reg_param'] = 2 ** cfg['reg_param']
    clf = QuadraticDiscriminantAnalysis(**cfg)
    #TODO: https://stackoverflow.com/questions/39782243/how-to-use-cross-val-score-with-random-state
    #s = cross_val_score(clf, X, y, scoring='roc_auc', cv=cv), customize scoring
    accuracy = cross_val_score(clf, X, y, cv=5)
    error = 1 - np.mean(accuracy)
    return error
        




