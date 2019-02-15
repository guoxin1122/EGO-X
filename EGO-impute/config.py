#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:22:51 2018

@author: macgx
"""

import os
import numpy as np
import sys
from collections import OrderedDict
import datetime
import csv
import logging
from math import log2

from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace
from mipego import mipego
from mipego.Surrogate import RandomForest
import copy

from ObjectiveFunction import Classifiers


datanames_24 = ['pim', 'ches', 'car', 'ltr', 'mgic', 'msk',
                'p-blk', 'adlt', 'bnk','sem', 'spam', 's-gc',
                's-im', 's-sh', 's-pl', 'tita', 'thy', 'wine',
                '1_46', '2_184', '3_389', '4_772', '5_917', '6_1049']

if len(sys.argv) == 2:
    print(sys.argv[1])
    dataname = sys.argv[1]
    assert dataname in datanames_24


# logging.basicConfig(filename='debug.log', filemode='w', level=logging.INFO)
current_path = os.path.dirname(os.path.abspath(__file__))
print('current_path=', current_path)

alg_choice = NominalSpace(['knn', 'svm', 'linsvm', 'dt', 'rf', 'adab', 'qda'], 'alg')
knn_search_space = OrdinalSpace([1, 30], 'n_neighbors')
svm_search_space = ContinuousSpace([[log2(1e-5), log2(1e5)], [log2(1e-5), log2(1e5)]],
                                   ['C', 'gamma'])
linsvm_search_space = ContinuousSpace([log2(1e-5), log2(1e5)], 'C_lin')
dt_search_space = OrdinalSpace([[1, 10], [2,100],[2,100]],
                               ['max_depth_dt', 'min_samples_split_dt', 'min_samples_leaf_dt'])
rf_search_space = OrdinalSpace([[1, 30], [1, 10], [2,100], [2,100]], 
                                ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'])
adab_search_space = OrdinalSpace([1, 30], 'n_estimators_adab')
qda_search_space = ContinuousSpace([log2(1e-3), log2(1e3)], 'reg_param')

search_space = alg_choice * knn_search_space * svm_search_space * linsvm_search_space * dt_search_space \
               * rf_search_space * adab_search_space * qda_search_space

clsf = Classifiers(dataname)
clsf.get_x_y()
obj_func = clsf.combine_all_func_to_dict() 

# obj_func = {'knn': knn_obj_func, 'svm':svm_obj_func, 'linsvm':linsvm_obj_func, 'dt': dt_obj_func,
#             'rf': rf_obj_func, 'adab': adab_obj_func, 'qda': qda_obj_func}

# subspace = {'knn':knn_search_space, 'rf': rf_search_space, 'dt':dt_search_space, 'adab':adab_search_space,
#             'qda': qda_search_space}

subdim = {'knn': len(knn_search_space), 'svm':len(svm_search_space), 'linsvm':len(linsvm_search_space),
          'dt':len(dt_search_space), 'rf':len(rf_search_space),
          'adab':len(adab_search_space), 'qda': len(qda_search_space)}

# var_class = [knn, svm,svm, lin, dt,dt,dt, rf,rf,rf,rf, adab, qda]
var_class = ['alg_choice'] + ['knn'] * subdim['knn'] + ['svm'] * subdim['svm'] \
            + ['linsvm'] * subdim['linsvm'] + ['dt'] * subdim['dt'] + ['rf'] * subdim['rf'] \
            + ['adab'] * subdim['adab']+ ['qda'] * subdim['qda']
var_class = np.array(var_class)
default_x = np.array(['rf', 15, 1, 1, 1, 5, 49, 49, 15, 5, 49, 49, 15, 1]) #13 dim without qda

#TODO: run different settings, e.g. n_init_sample=1, n_job=4 (MGFI)
histF_10reps = []
incumbent_10reps = []
runtime_10reps = []

for rep in range(10):
    tic = datetime.datetime.now()
    model = RandomForest(levels=search_space.levels)
    opt = mipego(search_space, var_class, default_x, obj_func, model, minimize=True, noisy=False,
                    max_eval=200, max_iter=None, n_init_sample=5, n_point=5, n_job=1,
                    random_seed=100+rep,
                    infill='MGFI', t0=2, tf=1e-1, schedule='exp', max_infill_eval=None,
                    n_restart=None, wait_iter=3, optimizer='MIES',
                    log_file=None, data_file=None, verbose=False)

    incumbent, stop_dict = opt.run()
    histF = opt.hist_f
    histX = opt.hist_x
    hist_incumbent = opt.hist_incumbent

    histF_10reps.append(histF)
    incumbent_10reps.append(incumbent)

    toc = datetime.datetime.now()
    runtime_10reps.append(str(toc - tic))

with open("{}/outcome/{}/histF_10reps.txt".format(current_path, dataname), "w") as f30:
    writer = csv.writer(f30)
    writer.writerows(histF_10reps)

with open("{}/outcome/{}/runtime_10reps.txt".format(current_path, dataname), "w") as f31:
    for item in runtime_10reps:
        f31.write(str(item) + '\n')

with open("{}/outcome/{}/incumbent_10reps.txt".format(current_path, dataname), "w") as f32:
    for incumb in incumbent_10reps:
        for element in incumb:
            f32.write(str(element) + ',')
        f32.write('\n')

print('finish meta3 on {} '.format(dataname))
print('mipego stop_dict = ', stop_dict)


