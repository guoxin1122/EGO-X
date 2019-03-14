#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 13:29:04 2018

@author: macgx
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

sns.set()
sns.set_style("whitegrid")

# only show meta3 because it is the best among all approaches
EGOss = '/Users/macgx/Documents/fall2018/test3_mipego/EGO-ss'


clf_choices = ['knn', 'svm', 'linsvm', 'dt', 'rf', 'adab', 'qda']

clf_counter = {'knn':0, 'svm':0, 'linsvm':0, 'dt':0, 'rf':0, 'adab':0, 'qda':0}

part_problems = ['3_389', '4_772', 'adlt', 'bnk',
                'ltr', 'mgic',  'p-blk', 'pim',  's-gc', 
                 's-im',  's-sh', 'sem', 'spam', 'tita'] 
# '1_46', '2_184',  '5_917', '6_1049', 'car', 'ches','msk','s-pl','thy', , 'wine'

for dataname in part_problems:
    with open('{}/outcome/{}/incumbent_10reps.txt'.format(EGOss, dataname), 'r') as f1:
        for line in f1.readlines():
            line = line.strip()
            line = line.split(',')
            for clf in clf_choices:
                if clf == line[0]:
                    clf_counter[clf] += 1
                    
#clf_choices = ['adab', 'dt', 'knn', 'linsvm','qda',  'rf',  'svm' ]                    
#color_code = ['#90c060', '#787878', 'pink', '#9090d8', 'darkkhaki', 'orange', 'orangered']

#color_code_barplot = ['pink','orangered', '#9090d8','#787878','orange','#90c060','darkkhaki']
 
color_code_barplot = ['#ffb5b8','#e24a33', '#988ed5','#777777','#fbc15d','#8dba43','#b3a05b']
            
ratio = [item / (10*len(part_problems)) for item in list(clf_counter.values())]                     
                      
fig = plt.figure(1)
ax = fig.add_subplot(111)

ax.grid(zorder=0, linestyle='-')
ax.bar(list(clf_counter.keys()), ratio, color=color_code_barplot)
ax.set_ylabel('Ratio')
fig.show()
fig.savefig('./visualize_outcome/datasets14/barplot')


#%%
############# stacked barplot for meta3


def get_one_row(dataname):
    clf_counter_stacked = {'knn':0, 'svm':0, 'linsvm':0, 'dt':0, 'rf':0, 'adab':0, 'qda':0}
    with open('{}/outcome/{}/incumbent_10reps.txt'.format(EGOss, dataname), 'r') as f2:
        for line in f2.readlines():
            line = line.strip()
            line = line.split(',')
            for clf in clf_choices:
                if clf == line[0]:
                    clf_counter_stacked[clf] += 1
        clf_counter_stacked = {k:[v] for k, v in clf_counter_stacked.items()}
        one_row = pd.DataFrame.from_dict(clf_counter_stacked)
        one_row.index = [dataname]
        return one_row
    
               
all_rows = get_one_row(dataname='3_389')                
for dataname in part_problems:
    if dataname == '3_389':
        continue
    else:
        one_row = get_one_row(dataname)           
        all_rows = pd.concat([all_rows, one_row], axis=0)
    

#cmap = cm.get_cmap('Spectral')
#cmap = ['green','grey', 'pink', 'purple', 'peru', 'orange', 'orangered' ]   
#color_code = ['pink','orangered', '#9090d8','#787878','orange','#90c060','darkkhaki']
color_code = ['#ffb5b8','#e24a33', '#988ed5','#777777','#fbc15d','#8dba43','#b3a05b']

#color_code = ['#90c060', '#787878', 'pink', '#9090d8', 'darkkhaki', 'orange', 'orangered']
cmap = LinearSegmentedColormap.from_list("", color_code )  


              
fig1 = plt.figure(2)
ax1 = fig1.add_subplot(111)

all_rows.plot(ax=ax1, kind='bar', stacked=True, colormap=cmap)                

ax1.set_ylabel('Repetitions') 
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))              

fig1.show()
fig1.savefig('./visualize_outcome/datasets14/barplot_alldata')
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                