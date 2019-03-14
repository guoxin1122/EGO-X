#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:09:06 2018

@author: macgx
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

def get_best_baseline_clf(path_prefix, baseline, problem_name):
    classifier = {}
    for clf in ['knn', 'svm', 'linsvm', 'dt', 'rf', 'adab', 'qda']:
        hist_clf_10reps = []
        with open('{}/{}/{}/outcome/{}/histF_10reps.txt'.format(path_prefix, baseline, problem_name, clf), 'r') as f1:
            for line in f1.readlines():
                line = line.strip()
                line = line.split(',')
                line = [eval(i) for i in line]
                hist_clf_10reps.append(line[-1])   
        clf_mean = np.mean(np.array(hist_clf_10reps))
        classifier[clf] = clf_mean
    best_clf = min(classifier.items(), key=lambda x:x[1])
    return best_clf[0]



def get_test_results(problem_name):

    sklearn_default = 'condition_sklearn_default'
    impute = 'condition_impute'
    meta2 = 'condition_meta2'
    meta3 = 'condition_meta3'
    smac = 'condition_smac'
    baseline = 'cond_multi_resp_log2'
    
    pd_error = {}
    path_prefix = '/Users/macgx/Documents/fall2018/test3_mipego'
    
    
    best_baseline_clf = get_best_baseline_clf(path_prefix, baseline, problem_name)
    with open('{}/{}/{}/outcome/{}/histF_10reps.txt'.format(path_prefix, baseline, problem_name, best_baseline_clf), 'r') as f1:
        error_10reps = []
        for line in f1.readlines():
            line = line.strip()
            line = line.split(',')
            error_10reps.append(eval(line[-1]))
        
        pd_error[baseline] = error_10reps
    
    
    
    with open('{}/{}/{}/outcome/min_error_10reps.txt'.format(path_prefix, sklearn_default, problem_name), 'r') as f1:
        error_10reps = []
        for line in f1.readlines():
            line = line.strip()
            line = line.split(',')
            error_10reps.append(eval(line[-1]))
        
        pd_error[sklearn_default] = error_10reps
        
        
    with open('{}/{}/{}/outcome/histF_10reps.txt'.format(path_prefix, impute, problem_name), 'r') as f2:
        error_10reps = []
        for line in f2.readlines():
            line = line.strip()
            line = line.split(',')
            error_10reps.append(eval(line[-1]))
        
        pd_error[impute] = error_10reps
        
    with open('{}/{}/{}/outcome/histF_10reps.txt'.format(path_prefix, meta2, problem_name), 'r') as f3:
        error_10reps = []
        for line in f3.readlines():
            line = line.strip()
            line = line.split(',')
            error_10reps.append(eval(line[-1]))
        
        pd_error[meta2] = error_10reps    
        
    with open('{}/{}/{}/outcome/histF_10reps.txt'.format(path_prefix, smac, problem_name), 'r') as f5:
        error_10reps = []
        for line in f5.readlines():
            line = line.strip()
            line = line.split(',')
            error_10reps.append(eval(line[-1]))
        
        pd_error[smac] = error_10reps
        
    with open('{}/{}/{}/outcome/histF_10reps.txt'.format(path_prefix, meta3, problem_name), 'r') as f4:
        error_10reps = []
        for line in f4.readlines():
            line = line.strip()
            line = line.split(',')
            error_10reps.append(eval(line[-1]))
        
        pd_error[meta3] = error_10reps
        
    df_pd_error = pd.DataFrame.from_dict(pd_error)   
    df_pd_error_mean = df_pd_error.mean(axis=0)
    best_method_name = df_pd_error_mean.idxmin()
    
    df_pd_error_mean_str = {}
    df_pd_error_mean_str_numeric = {}
    for x_name in df_pd_error.columns:
        if x_name == best_method_name:
            ##('####### best method: {}'.format(best_method_name))
            value = round(df_pd_error_mean[x_name] * 100,  2)
            df_pd_error_mean_str[x_name] = '{:.2f}'.format(value) + '*'
            df_pd_error_mean_str_numeric[x_name] = df_pd_error_mean[x_name]
            continue
        else:
            t, p = wilcoxon(x=df_pd_error[x_name], y=df_pd_error[best_method_name])
            ##('###### {}'.format(x_name))
            ##('t={}. p={}'.format(t,p))
            
            if p >= 0.05:
                ##('x, y come from the same distribution, that is, x and y are not significant different')
                value = round(df_pd_error_mean[x_name] * 100,  2)
                df_pd_error_mean_str[x_name] = '{:.2f}'.format(value)
                df_pd_error_mean_str_numeric[x_name] = df_pd_error_mean[x_name]
            
            elif p >= 0 and p < 0.05:
                ##('x, y are significant different')
                value = round(df_pd_error_mean[x_name] * 100,  2)
                df_pd_error_mean_str[x_name] = '{:.2f}'.format(value) + '-'
                df_pd_error_mean_str_numeric[x_name] = df_pd_error_mean[x_name]
    
    for key, value in df_pd_error_mean_str.items():
        df_pd_error_mean_str[key] = [value]
     
    for key, value in df_pd_error_mean_str_numeric.items():
        df_pd_error_mean_str_numeric[key] = [value]
        
    d_str_df = pd.DataFrame.from_dict(df_pd_error_mean_str)
    d_str_df = d_str_df.T
    
    d_str_df_numeric = pd.DataFrame.from_dict(df_pd_error_mean_str_numeric)
    d_str_df_numeric = d_str_df_numeric.T
    
    d_str_df.columns = [problem_name]
    d_str_df_numeric.columns = [problem_name]

    return d_str_df, d_str_df_numeric


problem_names = ['pim', 'ches', 'car', 'ltr', 'mgic', 'msk', 
'p-blk', 'adlt', 'bnk','sem', 'spam', 's-gc', 
's-im', 's-sh', 's-pl', 'tita', 'thy', 'wine', 
'1_46', '2_184', '3_389', '4_772', '5_917', '6_1049']


part_problems = ['3_389', '4_772', '5_917', '6_1049', 'bnk','car', 'ches',
 'ltr', 'msk', 'p-blk',  'pim', 's-gc', 
's-im', 's-pl', 'sem', 'spam', 'thy','tita',  'wine' ] #'1_46',


all_results, all_results_numeric = get_test_results(problem_name='1_46')

for problem_name in part_problems:
    one_column, one_column_numeric = get_test_results(problem_name)
    all_results = pd.concat([all_results, one_column], axis=1)
    all_results_numeric = pd.concat([all_results_numeric, one_column_numeric], axis=1)
    
    

all_results_numeric_T = all_results_numeric.T
best_approach = 'condition_meta3'

######## pairwise wilcoxon signed-rank tests
index = range(6)
columns = all_results_numeric_T.columns
df_pairwise = pd.DataFrame(index=index, columns=columns)
num_cols = len(all_results_numeric_T.columns)
for base_i in range(num_cols):
    for comp_j in range(base_i + 1, num_cols):
        base_compare = all_results_numeric_T[all_results_numeric_T.columns[base_i]]
        comparer = all_results_numeric_T[all_results_numeric_T.columns[comp_j]]
        t, p = wilcoxon(x=base_compare, y=comparer)
        df_pairwise.iloc[base_i, comp_j] = p
for i in range(num_cols):
    for j in range(num_cols):
        if i < j:
            df_pairwise.iloc[j, i] = df_pairwise.iloc[i, j]
df_pairwise.index = df_pairwise.columns


#sns.set()
#ax = sns.heatmap(df_pairwise)#, annot=True, cmap="YlGnBu"
#cm = sns.light_palette("green", as_cmap=True)
#s = df_pairwise.style.background_gradient(cmap=cm)
#
df_pairwise.fillna(1)
plt.matshow(df_pairwise)

            
#for approach in all_results_numeric_T.columns:
#    if approach == best_approach:
#        continue
#    else:
#        t, p = wilcoxon(x=all_results_numeric_T[best_approach], y=all_results_numeric_T[approach])
#        print('###### {}'.format(approach))
#        print('t={}. p={}'.format(t,p))

######### Nemenyi post-hoc test
#all_results_melt = all_results_numeric_T.melt(var_name='groups', value_name='values')
#a = sp.posthoc_conover(all_results_melt, val_col='values', group_col='groups', p_adjust = 'fdr_bh')
#
#pc = sp.posthoc_conover(all_results_melt, val_col='values', group_col='groups')
#heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
#sp.sign_plot(pc, **heatmap_args)
#
#
#pc = sp.posthoc_conover(all_results_melt, val_col='values', group_col='groups')
## Format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
#cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
#heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
#sp.sign_plot(pc, **heatmap_args)
  