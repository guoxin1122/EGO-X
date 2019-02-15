#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:03:34 2018

@author: macgx
"""

import numpy as np
from tabulate import tabulate
import os
import pandas as pd



#TODO: output table_24 to latex
# https://stackoverflow.com/questions/14380371/export-a-latex-table-from-pandas-dataframe

dirs = os.listdir('.')
dirs.remove('.DS_Store')
#dirs.remove( 'note.numbers')
dirs.remove( 'analyse.py')
dirs.remove('.idea')
dirs.remove('default_clf_perf.txt')
dirs.remove('results_table')

dirs.remove('adlt')


#dirs = ['1pima', '2ches', '3car', '4letter', '5magic', '6musk2', '7page_blocks']
results = {}
for dirname in dirs:  
    with open('./{}/outcome/histF_10reps.txt'.format(dirname), 'r') as f:
        f_mean_10reps = []
        for line in f.readlines():
            line = line.strip()
            line = line.split(',')
            line = [eval(item) for item in line]
            f_mean_10reps.append(line[-1])
        results[dirname] = np.mean(np.array(f_mean_10reps))
                    
f_ = {}
for key, value in results.items():
    f_[key] = [value * 100]

f_['adlt'] = 99
f_['s-sh'] = 99


df_impute = pd.DataFrame.from_dict(f_)

df_impute.index = ['meta2']
df_impute = df_impute.round(2)

df_impute.to_csv('./results_table/results.csv', sep='\t')











