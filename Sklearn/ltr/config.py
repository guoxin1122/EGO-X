import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from collections import OrderedDict

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

current_path = os.path.dirname(os.path.abspath(__file__))
print('current_path=', current_path)

X, y = preprocess(dataname=dataname)

min_error_10reps = []
for rep in range(10):

    X, y = preprocess(dataname=dataname)

    error = OrderedDict()
    
    clf_knn = KNeighborsClassifier()
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

    min_error = min(error.items(), key=lambda x: x[1]) #min_error is a tuple
    min_error_10reps.append(min_error)
    

with open('{}/outcome/min_error_10reps.txt'.format(current_path), 'w') as f2:
    for item in min_error_10reps:
        f2.write('{},{} \n'.format(item[0], item[1]))
    # for item in min_error:
    #     f2.write(str(item) + ',' )
    # f2.write('\n')












