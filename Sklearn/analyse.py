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

#dirs.remove('default_clf_perf.txt')
#dirs.remove('UCI_data_arff')
dirs.remove('results_table')


#dirs = ['1_46', '4_772', '5_917', '6_1049']
results = {}
for dirname in dirs:  
    with open('./{}/outcome/min_error_10reps.txt'.format(dirname), 'r') as f:
        f_mean_10reps = []
        for line in f.readlines():
            line = line.strip()
            line = line.split(',')
            f_mean_10reps.append(eval(line[-1]))
        results[dirname] = np.mean(np.array(f_mean_10reps))
                    
f_ = {}
for key, value in results.items():
    f_[key] = [value * 100]
    



df_impute = pd.DataFrame.from_dict(f_)

df_impute.index = ['sklearn_default']
df_impute = df_impute.round(2)

df_impute.to_csv('./results_table/results.csv', sep='\t')


#b = min(results, key=results.get)
#a = min(results.items(), key=lambda x: x[1])
#for item in a:
#    
#print(a)
#print(b)


