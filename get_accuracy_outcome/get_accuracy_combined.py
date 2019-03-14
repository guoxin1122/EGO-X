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

from scipy.stats import wilcoxon
from decimal import Decimal


#TODO: output table_24 to latex
# https://stackoverflow.com/questions/14380371/export-a-latex-table-from-pandas-dataframe


def get_24_accuracies(approach):
    path_prefix = '/Users/macgx/Documents/fall2018/test3_mipego'# /EGO-baseline
    
    problem_names = ['1_46', '2_184', '3_389', '4_772', '5_917', '6_1049','adlt', 'bnk',
                     'car', 'ches','ltr', 'mgic', 'msk', 'p-blk', 'pim',  's-gc', 
                     's-im', 's-pl', 's-sh', 'sem', 'spam', 'thy', 'tita', 'wine'] 

    if approach == 'EGO-baseline':
        histF_file = 'incumbent_f_10reps.txt'
        f_avg_for = {}
        for problem_name in problem_names:  
            with open('/{}/{}/outcome/{}/{}'.format(path_prefix, approach, problem_name, histF_file), 'r') as f:
                f_10reps = []
                for line in f.readlines():
                    line = line.strip()
                    line = line.split(',')
                    f_10reps.append(eval(line[-1]))
                f_avg_for[problem_name] = np.mean(np.array(f_10reps))        
        
    elif approach == 'Sklearn':
        histF_file = 'min_error_10reps.txt'
        f_avg_for = {}
        for problem_name in problem_names:  
            with open('/{}/{}/{}/outcome/{}'.format(path_prefix, approach, problem_name, histF_file), 'r') as f:
                f_10reps = []
                for line in f.readlines():
                    line = line.strip()
                    line = line.split(',')
                    f_10reps.append(eval(line[-1]))
                f_avg_for[problem_name] = np.mean(np.array(f_10reps))       
            
    else:
        histF_file = 'histF_10reps.txt'
        f_avg_for = {}
        for problem_name in problem_names:  
            with open('/{}/{}/outcome/{}/{}'.format(path_prefix, approach, problem_name, histF_file), 'r') as f:
                f_10reps = []
                for line in f.readlines():
                    line = line.strip()
                    line = line.split(',')
                    f_10reps.append(eval(line[-1]))
                f_avg_for[problem_name] = np.mean(np.array(f_10reps))
    
    # transform 0.1234 to 12.34                   
    f_avg_for_transformed = {}
    for key, value in f_avg_for.items():
        f_avg_for_transformed[key] = [value * 100]
#    f_avg_for_transformed['adlt'] = 99
    
    # for one approach, generate 24 accuracies
    f_avg_for_all = pd.DataFrame.from_dict(f_avg_for_transformed)
    # give approach name for the row
    f_avg_for_all.index = [approach]
    f_avg_for_all = f_avg_for_all.round(2)
    return f_avg_for_all


f_avg_all_approaches = []
approaches = ['EGO-ws', 'EGO-ss', 'EGO-impute', 'SMAC', 'EGO-baseline', 'Sklearn']

for approach in approaches:
    f_avg_for_all = get_24_accuracies(approach)
    f_avg_all_approaches.append(f_avg_for_all)

accuracy_table = pd.concat(f_avg_all_approaches)

#accuracy_table.to_csv('./get_accuracy_outcome/results.csv', sep='\t')
## http://tec.citius.usc.es/stac/
#TODO: add rank column
accuracy_table.T.to_csv('./combined/accuracy_T_stac.csv', sep=',')

#%%
## convert combinedAccuracyTable to make it suitable for STAC
#with open('/Users/macgx/Documents/fall2018/test3_mipego/get_accuracy_outcome/combinedAccuracyTable', 'r') as f2:
dc = pd.read_csv('/Users/macgx/Documents/fall2018/test3_mipego/get_accuracy_outcome/combATable.csv', 
                 sep=';')
dc = dc.set_index(dc.columns[0])
dc.T.to_csv('./get_accuracy_outcome/combined/combATable_stac.csv', sep=',')

#%% plot CD, already get avrank from stac web platform

import Orange
import matplotlib.pyplot as plt

with open('/Users/macgx/Documents/fall2018/test3_mipego/get_accuracy_outcome/RankCombined.csv', 'r') as f2:
    avranks = []
    algorithm_names = []
    for line in f2.readlines()[1:]: #skip head row
        line = line.strip()
        line = line.split(',')
        print(line)
        avranks.append(eval(line[0]))
        algorithm_names.append(line[1])

#tested on 24 problem (datasets) including adlt
num_problems = 14
cd = Orange.evaluation.compute_CD(avranks, num_problems) 
Orange.evaluation.graph_ranks(avranks, algorithm_names, cd=cd, width=6, textspace=1.5)
plt.savefig('./combined/CDplot')
plt.show()

## create rank column and add it in the end of the accuracy table
rank_dict = dict(zip(algorithm_names, avranks))
rank_column = pd.DataFrame.from_dict(rank_dict, orient='index', columns=['Rank'])
rank_column = rank_column.round(2)

# here we use combined accuracy_table
accuracy_table_combined = pd.read_csv('./combATable.csv', sep=';')
accuracy_table_combined = accuracy_table_combined.set_index(accuracy_table_combined.columns[0])
accuracy_with_rank = pd.concat([accuracy_table_combined, rank_column], axis=1)
accuracy_with_rank.to_csv('./combined/accuracy_with_rank.csv', sep=',')

#%% generate underline --> significant difference from the best on one problem


path_prefix = '/Users/macgx/Documents/fall2018/test3_mipego'# /pall_meta2'

approaches =  ['EGO-ws', 'EGO-ss', 'EGO-impute', 'SMAC', 'EGO-baseline', 'Sklearn']


def get_performance_data_of_one_problem(problem_name):
    data_1_problem = {}
    for approach in approaches:
    
        if approach == 'EGO-baseline':
            with open('/{}/{}/outcome/{}/incumbent_f_10reps.txt'.format(path_prefix, approach, problem_name), 'r') as f1:
                f_10reps = []
                for line in f1.readlines():
                    line = line.strip()
                    line = line.split(',')
                    f_10reps.append(eval(line[-1]))
                data_1_problem[approach] = f_10reps
        
        elif approach == 'Sklearn':
            with open('/{}/{}/{}/outcome/min_error_10reps.txt'.format(path_prefix, approach, problem_name), 'r') as f1:
                f_10reps = []
                for line in f1.readlines():
                    line = line.strip()
                    line = line.split(',')
                    f_10reps.append(eval(line[-1]))
                data_1_problem[approach] = f_10reps
            
        else:
            with open('/{}/{}/outcome/{}/histF_10reps.txt'.format(path_prefix, approach, problem_name), 'r') as f1:
                f_10reps = []
                for line in f1.readlines():
                    line = line.strip()
                    line = line.split(',')
                    f_10reps.append(eval(line[-1]))
                data_1_problem[approach] = f_10reps    

    return data_1_problem #type: dict 


## create underline dataframe for one problem          
def get_underline_1_problem(problem_name, data_1_problem):
    df_underline = pd.DataFrame(index=accuracy_table.index, columns=[problem_name])
        
    best_approach_for_1_problem = accuracy_table[problem_name].idxmin()
    
    for approach in approaches:
        if approach == best_approach_for_1_problem:
            df_underline.loc[approach, problem_name] = 'best'
        else:
            t, p = wilcoxon(x=data_1_problem[best_approach_for_1_problem], 
                            y=data_1_problem[approach])  
            if p < 0.05 and p >=0:
                df_underline.loc[approach, problem_name] = 1
            elif p >= 0.05:
                df_underline.loc[approach, problem_name] = 0
    return df_underline # type: dataframe, shape: one column            

    
problem_names = ['1_46', '2_184', '3_389', '4_772', '5_917', '6_1049','adlt', 'bnk',
                     'car', 'ches','ltr', 'mgic', 'msk', 'p-blk', 'pim',  's-gc', 
                     's-im', 's-pl', 's-sh', 'sem', 'spam', 'thy', 'tita', 'wine'] 
underline_24 = []  
data_24 = {}  
for problem_name in problem_names:
    data_1_problem = get_performance_data_of_one_problem(problem_name)
    df_underline_one_column = get_underline_1_problem(problem_name, data_1_problem)
    underline_24.append(df_underline_one_column)
    data_24[problem_name] = data_1_problem
df_underline_24 = pd.concat(underline_24, axis=1)   
df_underline_24.to_csv('./underline_indicator.csv', sep='\t') 
    
    
#%% do another pairwise wilcoxon test based upon the accuracy table
## Decimal module, to make round() function working properly
getcontext()
getcontext().prec = 7

approaches = accuracy_table_combined.index
#approaches =  ['EGO-ws', 'EGO-ss', 'EGO-impute', 'SMAC', 'EGO-baseline', 'Sklearn']

wilcoxon_table = pd.DataFrame(index=approaches,columns=approaches, dtype='float')
bold_indicator_table = pd.DataFrame(index=approaches,columns=approaches, dtype='float')

for row_i in range(accuracy_table_combined.shape[0]):
    
    for row_j in range(accuracy_table_combined.shape[0]): #accuracy_table.shape = (6, 24)
        if row_j == row_i:
            pass
        else:
            t, p = wilcoxon(x=accuracy_table_combined.iloc[row_i], 
                            y=accuracy_table_combined.iloc[row_j])
            
            #maybe the element in the table is t value
            p_ = Decimal(p)
            wilcoxon_table.iloc[row_i, row_j] = round(p_, 2)
            
            ##create bold indicator table --> significant difference
            if p < 0.05 and p >= 0:
                bold_indicator_table.iloc[row_i, row_j] = 1
            elif p >= 0.05:
                bold_indicator_table.iloc[row_i, row_j] = 0
            ## parentheses indicator can be draw by hand


wilcoxon_table.to_csv('./combined/wilcoxon_table.csv', sep=',') 
 













    
    