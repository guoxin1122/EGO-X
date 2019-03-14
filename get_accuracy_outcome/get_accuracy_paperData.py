#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 08:41:41 2019

@author: macgx
"""
import pandas as pd
from scipy.stats import wilcoxon
from decimal import Decimal


#path_prefix = 'Users/macgx/Documents/fall2018/test3_mipego/get_accuracy_outcome'
#
#method_names = ['EGO-ws', 'EGO-ss', 'EGO-impute', 'EGO-baseline', 'Sklearn','RS',
#                'Spearmint', 'SMAC', 'CMA', 'GP-matern-noimpute', 'GP-matern', 
#           'GP-cond', 'GP-laplace', 'GP-matern-ls', 'GP-cond-ls', 'GP-laplace-ls']





#integer_index = list(range(1, 25))
#probelmDataset = dict(zip(problem_names, integer_index))
#
#problem_name = '1_46'
##def get_performance_data_of_one_problem_PaperData(problem_name:
#def get_data_of_one_method_one_problem(method_name, problem_name):
#    data = {}
#    with open('/{}/paperData/{}.txt'.format(path_prefix, method_name), 'r') as f1:
#        for line in f1.readlines()[probelmDataset[problem_name] : probelmDataset[problem_name] + 1]:
#            line = line.strip()
#            line = line.split(' ')
#            line_numeric = [eval(item) for item in line]
#            data[method_name] = line_numeric
#    return data
#
### get data_1_problem 
#data_update = get_data_of_one_method_one_problem('RS', problem_name)        
#for method_name in method_names:
#    data = get_data_of_one_method_one_problem(method_name, problem_name)        
#    data_update = {**data, **data_update}
        
        
        
        
        
        
        
        
#%% generate underline --> significant difference from the best on one problem


path_prefix = '/Users/macgx/Documents/fall2018/test3_mipego'

approaches_EGO =  ['EGO-ws', 'EGO-ss', 'EGO-impute',  'EGO-baseline', 'Sklearn']#'SMAC',
approaches_GP = ['Spearmint', 'SMAC', 'CMA', 'GP-matern-noimpute', 'GP-matern', 'RS',
                 'GP-cond', 'GP-laplace', 'GP-matern-ls', 'GP-cond-ls', 'GP-laplace-ls']


def get_performance_data_of_one_problem(problem_name):
    data_1_problem = {}
    for approach in approaches_EGO:
    
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
                
    for approach in approaches_GP:
        
        with open('{}/GP-X/{}/outcome/{}/histF_10reps.txt'.format(path_prefix, approach, problem_name), 'r') as f2:
            f_10reps_GP = []
            for line in f2.readlines():
                line = line.strip()
                f_10reps_GP.append(eval(line))
            data_1_problem[approach] = f_10reps_GP  

    return data_1_problem #type: dict 


data_1_problem = get_performance_data_of_one_problem('1_46')

accuracy_table_24datasets = pd.read_csv('./combined/results1_24datasets.csv', sep=';')
accuracy_table_24datasets = accuracy_table_24datasets.set_index(accuracy_table_24datasets.columns[0])

approaches = approaches_EGO + approaches_GP
#%% create underline dataframe for 14 problems

def get_underline_1_problem(problem_name, data_1_problem):
    df_underline = pd.DataFrame(index=accuracy_table_24datasets.index, columns=[problem_name])
        
    best_approach_for_1_problem = accuracy_table_24datasets[problem_name].idxmin()
    
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

    
#problem_names = ['1_46', '2_184', '3_389', '4_772', '5_917', '6_1049','adlt', 'bnk',
#                     'car', 'ches','ltr', 'mgic', 'msk', 'p-blk', 'pim',  's-gc', 
#                     's-im', 's-pl', 's-sh', 'sem', 'spam', 'thy', 'tita', 'wine'] 

problem_names = ['3_389', '4_772', 'adlt', 'bnk',
                 'ltr', 'mgic',  'p-blk', 'pim',  's-gc', 
                 's-im', 's-sh', 'sem', 'spam', 'tita']


underline_24 = []  
data_24 = {}  
for problem_name in problem_names:
    data_1_problem = get_performance_data_of_one_problem(problem_name)
    df_underline_one_column = get_underline_1_problem(problem_name, data_1_problem)
    underline_24.append(df_underline_one_column)
    data_24[problem_name] = data_1_problem
df_underline_24 = pd.concat(underline_24, axis=1)   
df_underline_24.to_csv('./combined/underline_indicator_14.csv', sep=',')   



#%% read RankCombined.csv to get rank-method dictionary
with open('/Users/macgx/Documents/fall2018/test3_mipego/get_accuracy_outcome/RankCombined.csv', 'r') as f3:
    avranks = []
    algorithm_names = []
    for line in f3.readlines()[1:]: #skip head row
        line = line.strip()
        line = line.split(',')
#        print(line)
        avranks.append(eval(line[0]))
        algorithm_names.append(line[1])
        
## create rank column and add it in the end of the accuracy table
rank_dict = dict(zip(algorithm_names, avranks))
rank_column = pd.DataFrame.from_dict(rank_dict, orient='index', columns=['Rank'])
rank_column = rank_column.round(2)


#%% plot CD, avrank is obtained from above

import Orange
import matplotlib.pyplot as plt
#tested on 24 problem (datasets) including adlt
num_problems = 14
cd = Orange.evaluation.compute_CD(avranks, num_problems) 
Orange.evaluation.graph_ranks(avranks, algorithm_names, cd=cd, width=6, textspace=1.5)
plt.savefig('./combined/CDplot')
plt.show()

#%% add rank to accuracy table

# here we use combined accuracy_table with 14 datasets
#accuracy_table_combined = pd.read_csv('./combATable.csv', sep=';')
accuracy_table_combined = pd.read_csv('./combined/results2_14datasets.csv', sep=';')

accuracy_table_combined = accuracy_table_combined.set_index(accuracy_table_combined.columns[0])
accuracy_with_rank = pd.concat([accuracy_table_combined, rank_column], axis=1)
accuracy_with_rank.to_csv('./combined/accuracy14_with_rank.csv', sep=',')      
        