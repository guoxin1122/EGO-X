import numpy as np
import scipy as sp
import scipy.stats as st
import itertools as it
import pandas as pd

def friedman_test(*args):
    """
        Performs a Friedman ranking test.
        Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.
        
        Parameters
        ----------
        sample1, sample2, ... : array_like
            The sample measurements for each group.
            
        Returns
        -------
        F-value : float
            The computed F-value of the test.
        p-value : float
            The associated p-value from the F-distribution.
        rankings : array_like
            The ranking for each group.
        pivots : array_like
            The pivotal quantities for each group.
            
        References
        ----------
        M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
        D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
    """
    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    rankings = []
    for i in range(n):
        
        row = [col[i] for col in args]
        row_sort = sorted(row) #ascending, so the first one is smallest
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v)-1)/2. for v in row])

    rankings_avg = [sp.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r/sp.sqrt(k*(k+1)/(6.*n)) for r in rankings_avg]

    chi2 = ((12*n)/float((k*(k+1))))*((sp.sum(r**2 for r in rankings_avg))-((k*(k+1)**2)/float(4)))
    iman_davenport = ((n-1)*chi2)/float((n*(k-1)-chi2))
    #Iman and Davenport (1980) showed that Friedman’s χ2 is undesirably conservative 
    # and derived a better statistic, as above

    p_value = 1 - st.f.cdf(iman_davenport, k-1, (k-1)*(n-1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


dc = pd.read_csv('/Users/macgx/Documents/fall2018/test3_mipego/get_accuracy_outcome/combATable.csv', 
                 sep=';')

#dc = pd.read_csv('/Users/macgx/Documents/fall2018/test3_mipego/get_accuracy_outcome/results.csv', 
#                 sep=',')
#0.0007862984064577194
dc = dc.set_index(dc.columns[0])
dcT = dc.T
dcT.index = list(range(14))
#for colname in dcT.columns:
#    dcT[colname].values.tolist()
#dcT
dcT_dict = dcT.to_dict()
    
#dc_array = dc.T.values

statistic, p_value, rankings, ranking_cmp  = friedman_test(*dcT_dict.values())
rankings, names = map(list, zip(*sorted(zip(rankings, dcT_dict.keys()), key=lambda t: t[0])))
rankings_names = pd.DataFrame({'Rank':rankings, 'Algorithms':names})
rankings_names.to_csv('RankCombined.csv', sep=',', index=False)

ranks = {key: ranking_cmp[i] for i,key in enumerate(dcT_dict.keys())}

def friedman(alpha=0.05, post_hoc="bonferroni_dunn_test", control=None):
    values = clean_missing_values(request.json)
    statistic, p_value, rankings, ranking_cmp = npt.friedman_test(*values.values())
    rankings, names = map(list, zip(*sorted(zip(rankings, values.keys()), key=lambda t: t[0])))
    
    ranks = {key: ranking_cmp[i] for i,key in enumerate(values.keys())}
    if post_hoc.split('_')[-1] == "test":
        comparisons, z_values, _, adj_p_values = getattr(npt, post_hoc)(ranks, control)
    else:
        comparisons, z_values, _, adj_p_values = getattr(npt, post_hoc)(ranks)
    return statistic, p_value, rankings, names, comparisons, z_values, adj_p_values















