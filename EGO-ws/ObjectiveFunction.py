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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# model assessment
from sklearn.model_selection import cross_val_score

from PreprocessData import PreprocessData

class Classifiers():
    def __init__(self, dataname):
        self.dataname = dataname

    def get_x_y(self):
        ppd = PreprocessData(dataname=self.dataname)
        # self.X, self.y = pd.test_preprocess()
        self.X, self.y = ppd.preprocess()
        print('finish get_x_y ')
        
    def combine_all_func_to_dict(self):

        def rf_obj_func(cfg): # give cfg list in mipego
            clf = RandomForestClassifier(**cfg)#
            accuracy = cross_val_score(clf, self.X, self.y, cv=5)
            error = 1 - np.mean(accuracy)
            return error

        def dt_obj_func(cfg):
            # ['max_depth_dt', 'min_samples_split_dt', 'min_samples_leaf_dt']
            # rename key
            cfg['max_depth'] = cfg.pop('max_depth_dt')
            cfg['min_samples_split'] = cfg.pop('min_samples_split_dt')
            cfg['min_samples_leaf'] = cfg.pop('min_samples_leaf_dt')
            clf = DecisionTreeClassifier(**cfg)
            accuracy = cross_val_score(clf, self.X, self.y, cv=5)
            error = 1 - np.mean(accuracy)
            return error

        def svm_obj_func(cfg):
            cfg['C'] = 1 if cfg['C']  == 0 else cfg['C']
            cfg['gamma'] = 1 if cfg['gamma'] == 0 else cfg['gamma']

            cfg['C'] = 2 ** cfg['C']
            cfg['gamma'] = 2 ** cfg['gamma']

            clf = SVC(**cfg) # default kernel =’rbf’
            accuracy = cross_val_score(clf, self.X, self.y, cv=5)
            error = 1 - np.mean(accuracy)
            return error

        def linsvm_obj_func(cfg):
            cfg['C_lin'] = 1 if cfg['C_lin'] == 0 else cfg['C_lin']
            cfg['C'] = cfg.pop('C_lin')
            cfg['C'] = 2 ** cfg['C']
            clf = LinearSVC(**cfg)
            accuracy = cross_val_score(clf, self.X, self.y, cv=5)
            error = 1 - np.mean(accuracy)
            return error

        def knn_obj_func(cfg):
            clf = KNeighborsClassifier(**cfg)
            accuracy = cross_val_score(clf, self.X, self.y, cv=5)
            error = 1 - np.mean(accuracy)
            return error

        def adab_obj_func(cfg):
            cfg['n_estimators'] = cfg.pop('n_estimators_adab')
            clf = AdaBoostClassifier(**cfg)
            accuracy = cross_val_score(clf, self.X, self.y, cv=5)
            error = 1 - np.mean(accuracy)
            return error

        def qda_obj_func(cfg):
            cfg['reg_param'] = 1 if cfg['reg_param'] == 0 else cfg['reg_param']
            cfg['reg_param'] = 2 ** cfg['reg_param']
            clf = QuadraticDiscriminantAnalysis(**cfg)
            #TODO: https://stackoverflow.com/questions/39782243/how-to-use-cross-val-score-with-random-state
            #s = cross_val_score(clf, self.X, self.y, scoring='roc_auc', cv=cv), customize scoring
            accuracy = cross_val_score(clf, self.X, self.y, cv=5)
            error = 1 - np.mean(accuracy)
            return error

        obj_func = {'knn': knn_obj_func, 'svm':svm_obj_func, 'linsvm':linsvm_obj_func, 'dt': dt_obj_func,
                    'rf': rf_obj_func, 'adab': adab_obj_func, 'qda': qda_obj_func}
        return obj_func


        




