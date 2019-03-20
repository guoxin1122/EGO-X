# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
@email: wangronin@gmail.com

"""
from __future__ import division
from __future__ import print_function

import pdb
import dill, functools, itertools, copyreg, logging
import numpy as np
import types
import logging

import queue
import threading
import time
import copy

from joblib import Parallel, delayed
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import r2_score

from collections import OrderedDict
from collections import deque

from .InfillCriteria import EI, PI, MGFI
from .optimizer import mies
from .utils import proportional_selection

# TODO: remove the usage of pandas here change it to customized np.ndarray
# TODO: finalize the logging system
class Solution(np.ndarray):
    def __new__(cls, x, fitness=None, n_eval=0, index=None, var_name=None):
        obj = np.asarray(x, dtype='object').view(cls)
        obj.fitness = fitness
        obj.n_eval = n_eval
        obj.index = index
        obj.var_name = var_name
        # must return the newly created object
        return obj # obj is self
    
    def __array_finalize__(self, obj):
        if obj is None: return
        # Needed for array slicing
        self.fitness = getattr(obj, 'fitness', None)
        self.n_eval = getattr(obj, 'n_eval', None)
        self.index = getattr(obj, 'index', None)
        self.var_name = getattr(obj, 'var_name', None)
        
#        self.bound = getattr(obj, 'bound', None)
    
    def to_dict(self):
        if self.var_name is None: return
        return {k : self[i] for i, k in enumerate(self.var_name)}  

    def to_ordered_dict(self):
        if self.var_name is None: return
        x_dict = {k : self[i] for i, k in enumerate(self.var_name)}  
        return OrderedDict(sorted(x_dict.items()))

    def to_array(self):
        return np.array(self)
    
    def __str__(self):
        return str(self.to_array())

    
class mipego(object):
    """
    Generic Bayesian optimization algorithm
    """
    def __init__(self, search_space, var_class, obj_func, surrogate, subspace,
                 minimize=True, noisy=False, max_eval=None, max_iter=None, 
                 infill='EI', t0=2, tf=1e-1, schedule=None,
                 n_init_sample=None, n_point=1, n_job=1, backend='multiprocessing',
                 n_restart=None, max_infill_eval=None, wait_iter=3, optimizer='MIES', 
                 log_file=None, data_file=None, verbose=False, random_seed=None,
                 available_gpus=[], parallel_mies=False):
        """
        parameter
        ---------
            search_space : instance of SearchSpace type
            obj_func : callable,
                the objective function to optimize
            surrogate: surrogate model, currently support either GPR or random forest
            minimize : bool,
                minimize or maximize
            noisy : bool,
                is the objective stochastic or not?
            max_eval : int,
                maximal number of evaluations on the objective function
            max_iter : int,
                maximal iteration
            n_init_sample : int,
                the size of inital Design of Experiment (DoE),
                default: 20 * dim
            n_point : int,
                the number of candidate solutions proposed using infill-criteria,
                default : 1
            n_job : int,
                the number of jobs scheduled for parallelizing the evaluation. 
                Only Effective when n_point > 1 
            backend : str, 
                the parallelization backend, supporting: 'multiprocessing', 'MPI', 'SPARC'
            optimizer: str,
                the optimization algorithm for infill-criteria,
                supported options: 'MIES' (Mixed-Integer Evolution Strategy), 
                                   'BFGS' (quasi-Newtion for GPR)
            available_gpus: array:
                one dimensional array of GPU numbers to use for running on GPUs in parallel. Defaults to no gpus.

        """
        self.verbose = verbose
        self.log_file = log_file
        self.data_file = data_file
        self.subspace = subspace # is a dictionary
        self.parallel_mies = parallel_mies

        self._space = search_space # Note: complete search space including categorical
        self.obj_func = obj_func # Note: dictionary
        self.var_class = var_class
        self.alg_name = list(self._space.levels[0])#
        # self.alg_name = ['knn', 'svm', 'linsvm', 'dt', 'rf', 'adab', 'qda']

        self.subvar_names = {'knn': ['n_neighbors'], 'svm': ['C', 'gamma'], 'linsvm': ['C_lin'],
                             'dt': ['max_depth_dt', 'min_samples_split_dt', 'min_samples_leaf_dt'],
                             'rf': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
                             'adab': ['n_estimators_adab'], 'qda': ['reg_param']}
        self.subdata = {alg: [] for alg in self.alg_name}
        self.acquisition_func = {alg: 1 for alg in self.alg_name}
        self.subsurrogate = {}
        for alg in self.alg_name:
            self.subsurrogate[alg] = copy.deepcopy(surrogate)
        self.noisy = noisy
        self.surrogate = surrogate
        self.async_surrogates = {}       
        self.n_point = n_point
        self.n_jobs = min(self.n_point, n_job)
        self.available_gpus = available_gpus
        self._parallel_backend = backend
        self.infill = infill
        self.minimize = minimize
        self.dim = len(self._space) # number of variables in search space
        self._best = min if self.minimize else max
        self.var_names = self._space.var_name

        self.r_index = self._space.id_C       # index of continuous variable
        self.i_index = self._space.id_O       # index of integer variable
        self.d_index = self._space.id_N       # index of categorical variable

        self.param_type = self._space.var_type
        self.N_r = len(self.r_index)
        self.N_i = len(self.i_index)
        self.N_d = len(self.d_index)
       
        # parameter: objective evaluation
        # TODO: for noisy objective function, maybe increase the initial evaluations
        self.init_n_eval = 1      
        self.max_eval = int(max_eval) if max_eval else np.inf
        self.max_iter = int(max_iter) if max_iter else np.inf
        self.n_init_sample = self.dim * 20 if n_init_sample is None else int(n_init_sample)
        self.eval_hist = []
        self.eval_hist_id = []
        self.iter_count = 0
        self.eval_count = 0

        # setting up cooling schedule
        if self.infill == 'MGFI':
            self.t0 = t0 # 2
            self.tf = tf # 0.1
            self.t = t0 # 2
            self.schedule = schedule
            
            # TODO: find a nicer way to integrate this part
            # cooling down to 1e-1
            max_iter = self.max_eval - self.n_init_sample
            if self.schedule == 'exp':                         # exponential
                self.alpha = (self.tf / t0) ** (1. / max_iter) 
            elif self.schedule == 'linear':
                self.eta = (t0 - self.tf) / max_iter           # linear
            elif self.schedule == 'log':
                self.c = self.tf * np.log(max_iter + 1)        # logarithmic 
            elif self.schedule == 'self-adaptive':
                raise NotImplementedError

        # paramter: acquisition function optimziation
        mask = np.nonzero(self._space.C_mask | self._space.O_mask)[0] # np.nonzero return the indices of the elements that are non-zero.
                                                                        # mask is actually id_C and id_O
        self._bounds = np.array([self._space.bounds[i] for i in mask])             # bounds for continuous and integer variable
        # self._levels = list(self._space.levels.values())
        self._levels = np.array([self._space.bounds[i] for i in self._space.id_N]) # levels for discrete variable
        self._optimizer = optimizer
        # TODO: set this number smaller when using L-BFGS and larger for MIES
        self._max_eval = int(5e2 * self.dim) if max_infill_eval is None else max_infill_eval
        self._random_start = int(5 * self.dim) if n_restart is None else n_restart
        self._wait_iter = int(wait_iter)    # maximal restarts when optimal value does not change

        # Intensify: the number of potential configuations compared against the current best
        # self.mu = int(np.ceil(self.n_init_sample / 3))
        self.mu = 3
        
        # stop criteria
        self.stop_dict = {}
        self.hist_f = []
        self.hist_iter_count = []
        self.hist_x = []
        self.hist_incumbent = []
        ######### temporarily comment self._check_params() TODO:maybe use it in the right way
        # self._check_params()

        # set the random seed
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)
            
        self._get_logger(self.log_file)
        
        # allows for pickling the objective function 
        # TODO: checke arguments of copyreg.pickle(type, function)
        # can not find dill.pickles
        #TODO: uncomment the following
        # copyreg.pickle(self._eval_one, dill.pickles)
        # copyreg.pickle(self.obj_func, dill.pickles)
        
#        copyreg.pickle(types.MethodType, self._eval_one)
#        copyreg.pickle(types.MethodType, self.obj_func)

        # paralellize gpus
        self.init_gpus = True
        self.evaluation_queue = queue.Queue()

    
    def _get_logger(self, logfile):
        """
        When logfile is None, no records are written
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('- %(asctime)s [%(levelname)s] -- '
                                      '[- %(process)d - %(name)s] %(message)s')

        # create console handler and set level to warning
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # create file handler and set level to debug
        if logfile is not None:
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def _compare(self, f1, f2):
        """
        Test if perf1 is better than perf2
        """
        if self.minimize:
            return f1 < f2
        else:
            return f2 > f2
    
    def _remove_duplicate(self, data):
        """
        check for the duplicated solutions, as it is not allowed
        for noiseless objective functions
        """
        ans = []
        X = np.array([s.tolist() for s in self.data], dtype='object')
        for i, x in enumerate(data):
            CON = np.all(np.isclose(np.asarray(X[:, self.r_index], dtype='float'),
                                    np.asarray(x[self.r_index], dtype='float')), axis=1)
            INT = np.all(X[:, self.i_index] == x[self.i_index], axis=1)
            CAT = np.all(X[:, self.d_index] == x[self.d_index], axis=1)
            if not any(CON & INT & CAT):
                ans.append(x)
        return ans

    def _eval_gpu(self, x, gpu_no=0, runs=1):
        """
        evaluate one solution
        """
        # TODO: sometimes the obj_func take a dictionary as input...
        fitness_, n_eval = x.fitness, x.n_eval
        # try:
            # ans = [self.obj_func(x.tolist()) for i in range(runs)]
        # except:
        ans = [self.obj_func(x.to_dict(), gpu_no) for i in range(runs)]

        fitness = np.sum(ans)

        x.n_eval += runs
        x.fitness = fitness / runs if fitness_ is None else (fitness_ * n_eval + fitness) / x.n_eval

        self.eval_count += runs
        self.eval_hist += ans
        self.eval_hist_id += [x.index] * runs
        
        return x, runs, ans, [x.index] * runs

    def _extract_and_evaluate(self, x, alg):# x is Solution, alg is string
        mask = self.var_class == alg
        valid_id = np.nonzero(mask)[0]  # valid id for rf
        value = x[valid_id].to_array()
        vname = x[valid_id].var_name[valid_id]
        cfg = dict(zip(vname, value))
        x.fitness = self.obj_func[alg](cfg)

        return x  # x is an updated Solution

    def _eval_one(self, x, runs=1):
        """
        evaluate one solution
        """
        # TODO: sometimes the obj_func take a dictionary as input...
        fitness_, n_eval = x.fitness, x.n_eval
        # try:
        # ans = [self.obj_func(x.tolist()) for i in range(runs)]
        # except:

        ##### check x's categorical variable before evaluation
        for alg in self.alg_name:
            if x[0] == alg:
                x = self._extract_and_evaluate(x, alg)

        ans = x.fitness

        self.eval_count += runs
        self.eval_hist += ans
        self.eval_hist_id += [x.index] * runs
        self.hist_x.append(x)
        
        return x, runs, ans, [x.index] * runs

    def evaluate(self, data, runs=1): # Note: input argument is data, instead of self.data ! 
#####here data is candidate points generated from acquisition in one step, i.e. len(data)==n_points here
        """ Evaluate the candidate points and update evaluation info in the dataframe
        """
        if isinstance(data, Solution):
            self._eval_one(data)
        
        elif isinstance(data, list): # default: a list of n_point Solution(s)
            if self.n_jobs > 1:
                if self._parallel_backend == 'multiprocessing': # parallel execution using joblib
                    res = Parallel(n_jobs=self.n_jobs, verbose=False)(
                        delayed(self._eval_one, check_pickle=False)(x) for x in data)
                    # TODO: maybe the problem is about input arguments of _eval_one missing gpu_no
                    x, runs, hist, hist_id = zip(*res) # e.g. res = [ (9,8,[7],[6]), (5,4,[3],[2]) ]
                    self.eval_count += sum(runs)
                    self.eval_hist += list(itertools.chain(*hist)) # append elements in hist list to eval_hist list; hist=([7], [3])
                    self.eval_hist_id += list(itertools.chain(*hist_id)) # hist_id=([6], [2])
                    for i, k in enumerate(data):
                        data[i] = x[i].copy() # copy updated solution x to data, i.e. update candidate solutions. Note: len(x)==len(data)
                elif self._parallel_backend == 'MPI': # parallel execution using MPI
                    # TODO: to use InstanceRunner here
                    pass
                elif self._parallel_backend == 'Spark': # parallel execution using Spark
                    pass        
            else: # n_point==1, i.e. now data list contains only one solution
                for i, x in enumerate(data): # data = [sample1, sample2, sample3]
                    self._eval_one(x)

    def fit_and_assess(self, surrogate=None, data=None):
        while True:
            try:
                X = np.atleast_2d([s.tolist() for s in data])

                fitness = np.array([s.fitness for s in data])
                ## add if statement, for the case n_init_sample=1
                if len(fitness) == 1:
                    fitness_scaled = fitness
                else:
                    # normalization the response for numerical stability
                    # e.g., for MGF-based acquisition function
                    # through normalization, infill value become larger
                    _min, _max = np.min(fitness), np.max(fitness)
                    ## add if statement to deal with the case where all elements in array fitness are equal
                    if not _min == _max:
                        fitness_scaled = (fitness - _min) / (_max - _min)
                    else:
                        fitness_scaled = fitness

                # fit the surrogate model
                if (surrogate is None):
                    self.surrogate.fit(X, fitness_scaled)
                    self.is_update = True
                    fitness_hat = self.surrogate.predict(X)# this is used to compute r2 score
                else:
                    surrogate.fit(X, fitness_scaled)
                    self.is_update = True
                    fitness_hat = surrogate.predict(X)                
                
                r2 = r2_score(fitness_scaled, fitness_hat)
                break 
            except Exception as e:
                print("Error fitting model, waiting 5s and retrying...")
                print('X:', X)
                print('fitness:', fitness)
                print(e)
                time.sleep(5)
        # TODO: in case r2 is really poor, re-fit the model or transform the input? 
        # consider the performance metric transformation in SMAC
        return r2

    def select_candidate(self): # select_candidate(self, key)
        self.is_update = False  # self.arg_max_acquisition(key)
        X, infill_value = self.arg_max_acquisition(subsurrogate=self.subsurrogate) # return multiple candidates and values
        # X is not a Solution
        # e.g. n_point=4 acquisition functions, they are totally same,
        # then they are optimized by mies respectively and generate 4 X. (include duplicate X)
        
        if self.n_point > 1:
            X = [Solution(x, index=len(self.data) + i, var_name=self.var_names) for i, x in enumerate(X)]
        else:
            X = [Solution(X, index=len(self.data), var_name=self.var_names)]

        X = self._remove_duplicate(X)

        # if the number of new design sites obtained is less than required,
        # draw the remaining ones randomly
        if len(X) < self.n_point:
            if self.verbose:
                self.logger.warn("iteration {}: duplicated solution found "
                                    "by optimization! New points is taken from random "
                                    "design".format(self.iter_count))
            N = self.n_point - len(X)
            if N > 1:
                s = self._space.sampling(N=N, method='LHS')
            else:      # To generate a single sample, only uniform sampling is feasible
                s = self._space.sampling(N=1, method='uniform')
            X += [Solution(x, index=len(self.data) + i, var_name=self.var_names) for i, x in enumerate(s)]

        candidates_id = [x.index for x in X]
        # for noisy fitness: perform a proportional selection from the evaluated ones
        if self.noisy:
            id_, fitness = zip([(i, d.fitness) for i, d in enumerate(self.data) if i != self.incumbent_id]) # id_ in [0,len(data)-1]
            # should be ... zip(*[(i, d.fitness) ...
            __ = proportional_selection(fitness, self.mu, self.minimize, replacement=False) # __ in [0, len(data)-1]
            candidates_id.append(id_[__]) # candidates = candidates generated from arg_max_acquisiziton + proportional selected from data set

        # TODO: postpone the evaluate to intensify...
        # move evaluation and augmenting data out of this method

        # # evaluate candidates obtained from arg_max_acquisition
        # # and append them to data set
        # self.evaluate(X, runs=self.init_n_eval)
        # # after self.evaluate(X), X.fitness is updated
        # self.data += X

        return candidates_id, X # this return is used for intensify function

    def intensify(self, candidates_ids):
        """
        intensification procedure for noisy observations (from SMAC)
        """
        # TODO: verify the implementation here
        maxR = 20 # maximal number of the evaluations on the incumbent
        for i, ID in enumerate(candidates_ids):
            r, extra_run = 1, 1
            conf = self.data.loc[i] # should be loc[ID] ?  data don't have loc method !!!
            self.evaluate(conf, 1)
            print(conf.to_frame().T)

            if conf.n_eval > self.incumbent_id.n_eval:
                self.incumbent_id = self.evaluate(self.incumbent_id, 1)
                extra_run = 0

            while True:
                if self._compare(self.incumbent_id.perf, conf.perf): # _compare return f1 < f2
                    self.incumbent_id = self.evaluate(self.incumbent_id, 
                                                   min(extra_run, maxR - self.incumbent_id.n_eval))
                    print(self.incumbent_id.to_frame().T)
                    break
                if conf.n_eval > self.incumbent_id.n_eval:
                    self.incumbent_id = conf
                    if self.verbose:
                        print('[DEBUG] iteration %d -- new incumbent selected:' % self.iter_count)
                        print('[DEBUG] {}'.format(self.incumbent_id))
                        print('[DEBUG] with performance: {}'.format(self.incumbent_id.perf))
                        print()
                    break

                r = min(2 * r, self.incumbent_id.n_eval - conf.n_eval)
                self.data.loc[i] = self.evaluate(conf, r)
                print(self.conf.to_frame().T)
                extra_run += r

    def _initialize_2(self):
        self.data = []
        samples = self._space.sampling(self.n_init_sample)
        for k, sample in enumerate(samples): # sample is a list
            # evaluate each part of sample, to initialize corresponding subdataset and subsurrogate
            for alg in self.alg_name:
                valid_id = np.nonzero(self.var_class == alg)[0]
                s = [sample[i] for i in valid_id] # implement s = sample[valid_id]

                partOfSample = Solution(s, index=k, var_name=self.subvar_names[alg])
                # evaluate partOfSample
                value = partOfSample.to_array()
                vname = self.subvar_names[alg]
                cfg = dict(zip(vname, value))
                partOfSample.fitness = self.obj_func[alg](cfg)

                # augment subdata[alg]
                self.subdata[alg].append(partOfSample)

                #TODO: check if we need self.data. Yes, we need
                # let partOfSample.fitness be sample_.fitness, to initialize entire dataset
                if sample[0] == alg:
                    sample_ = Solution(sample, index=k, var_name=self.var_names)
                    sample_.fitness = partOfSample.fitness
                    self.data += [sample_]

        # initialize surrogate for each classifier
        for alg in self.alg_name:
            r2 = self.fit_and_assess(surrogate=self.subsurrogate[alg], data=self.subdata[alg])

        #TODO:deal with initial evaluation count
        self.eval_count += self.n_init_sample # * len(self.alg_name)
        # eval_count = 70 now



    def _initialize(self):
        """Generate the initial data set (DOE) and construct the surrogate model
        """
        if self.verbose:
            self.logger.info('selected surrogate model: {}'.format(self.surrogate.__class__)) 
            self.logger.info('building the initial design of experiemnts...')

                                       #knn, svm,svm, lin, dt,dt,dt, rf,rf,rf,rf, adab, qda
        samples = self._space.sampling(self.n_init_sample)
        self.data = [Solution(s, index=k, var_name=self.var_names) for k, s in enumerate(samples)]
        self.evaluate(self.data, runs=self.init_n_eval) # after running self.evaluate(self.data)
                                      # self.data is updated i.e. add information to fitness attribute

        r2 = self.fit_and_assess()


    def gpuworker(self, q, gpu_no):
        "GPU worker function "

        self.async_surrogates[gpu_no] = copy.deepcopy(self.surrogate);
        while True:
            if self.verbose:
                self.logger.info('GPU no. {} is waiting for task'.format(gpu_no))

            confs_ = q.get()

            time.sleep(gpu_no)

            if self.verbose:
                self.logger.info('Evaluating:')
                self.logger.info(confs_.to_dict())
            confs_ = self._eval_gpu(confs_, gpu_no)[0] #will write the result to confs_

            
            if self.data is None:
                self.data = [confs_]
            else: 
                self.data += [confs_]
            perf = np.array([s.fitness for s in self.data])
            #self.data.perf = pd.to_numeric(self.data.perf)
            #self.eval_count += 1
            self.incumbent_id = np.nonzero(perf == self._best(perf))[0][0]
            self.incumbent = self.data[self.incumbent_id]
            
            if self.verbose:
                self.logger.info("{} threads still running...".format(threading.active_count()))

            # model re-training
            self.iter_count += 1
            self.hist_f.append(self.incumbent.fitness)
            self.hist_iter_count.append(self.iter_count)
            if self.verbose:
                self.logger.info('iteration {} with current fitness {}, current incumbent is:'.format(self.iter_count, self.incumbent.fitness))
                self.logger.info(self.incumbent.to_dict())

            incumbent = self.incumbent
            #return self._get_var(incumbent)[0], incumbent.perf.values

            q.task_done()

            #print "GPU no. {} is waiting for task on thread {}".format(gpu_no, gpu_no)
            if not self.check_stop():
                
                self.logger.info('Data size is {}'.format(len(self.data)))
                if len(self.data) >= self.n_init_sample:
                    self.fit_and_assess(surrogate = self.async_surrogates[gpu_no])
                    while True:
                        try:
                            X, infill_value = self.arg_max_acquisition(surrogate = self.async_surrogates[gpu_no])
                            confs_ = Solution(X, index=len(self.data)+q.qsize(), var_name=self.var_names)
                            break
                        except Exception as e:
                            print(e)
                            print("Error selecting candidate, retrying in 60 seconds...")
                            time.sleep(60)
                    q.put(confs_)
                else:
                    samples = self._space.sampling(1)
                    confs_ = Solution(samples[0], index=len(self.data)+q.qsize(), var_name=self.var_names)
                    #confs_ = self._to_dataframe(self._space.sampling(1))
                    if (q.empty()):
                        q.put(confs_)
                
            else:
                break

        print('Finished thread {}'.format(gpu_no))

    def _evaluate_selected_candidate(self, x, alg):
        valid_id = np.nonzero(self.var_class == alg)[0]
        partOfx = x[valid_id]  # x[valid_id] is Solution
        # to get dictionary cfg
        value = x[valid_id].to_array()
        vname = x[valid_id].var_name[valid_id] #TODO: check if x.var_name[valid_id] works
        cfg = dict(zip(vname, value))
        partOfx.fitness = self.obj_func[alg](cfg)

        # augment subdata[alg], subdata is only used to fit associated surrogate
        self.subdata[alg] += [partOfx]

        # let the fitness of parOfx be x.fitness
        x.fitness = partOfx.fitness

        # record evaluation history according to x's information
        runs = 1
        self.eval_count += runs
        self.eval_hist += x.fitness
        self.eval_hist_id += [x.index] * runs
        self.hist_x.append(x)

        return x.fitness


    def step(self):
        if not hasattr(self, 'data'):
            self._initialize_2()
            ########### store best fitness of initial samples to self.hist_f as its first element.######
            fitness = np.array([s.fitness for s in self.data]) # self.data now consists of evaluated initial samples and evaluated points selected by acquisition function
            self.hist_f.append(self._best(fitness))

        ids, X = self.select_candidate() # self.select_candidate(subsurrogate=self.subsurrogate)

        # evaluate candidate X and fit corresponding surrogate
        x = X[0] # x is Solution
        for alg in self.alg_name:
            if x[0] == alg:
                x_fitness = self._evaluate_selected_candidate(x, alg=alg) # do not need x_fitness
                # fit corresponding surrogate
                self.fit_and_assess(surrogate=self.subsurrogate[alg], data=self.subdata[alg])
        self.data += [x]

        # record incumbent
        fitness = np.array([s.fitness for s in self.data])  # self.data now consists of evaluated initial samples and evaluated points selected by acquisition function
        self.incumbent_id = np.nonzero(fitness == self._best(fitness))[0][0]
        self.incumbent = self.data[self.incumbent_id]

        # update iteration information
        self.iter_count += 1
        self.hist_f.append(self.incumbent.fitness)
        self.hist_iter_count.append(self.iter_count)
        self.hist_incumbent.append(self.incumbent)

        # print the incumbent in current iteration
        if self.verbose:
            self.logger.info('iteration {}, current incumbent is:'.format(self.iter_count))
            self.logger.info(self.incumbent.to_dict())

        return self.incumbent, self.incumbent.fitness


    def run(self):
        if (len(self.available_gpus) > 0):

            if self.n_jobs > len(self.available_gpus):
                print("Not enough GPUs available for n_jobs")
                return 1

            self.n_point = 1 #set n_point to 1 because we only do one evaluation at a time (async)
            # initialize
            if self.verbose:
                self.logger.info('selected surrogate model: {}'.format(self.surrogate.__class__)) 
                self.logger.info('building the initial design of experiemnts...')

            samples = self._space.sampling(self.n_init_sample)
            datasamples = [Solution(s, index=k, var_name=self.var_names) for k, s in enumerate(samples)]
            self.data = None


            for i in range(self.n_init_sample):
                self.evaluation_queue.put(datasamples[i])

            #self.evaluate(self.data, runs=self.init_n_eval)
            ## set the initial incumbent
            #fitness = np.array([s.fitness for s in self.data])
            #self.incumbent_id = np.nonzero(fitness == self._best(fitness))[0][0]
            #self.fit_and_assess()
            # #######################
            # new code... 
            #self.data = pd.DataFrame()
            #samples = self._space.sampling(self.n_init_sample)
            #initial_data_samples = self._to_dataframe(samples)
            # occupy queue with initial jobs
            #for i in range(self.n_jobs):
            #    self.evaluation_queue.put(initial_data_samples.iloc[i])

            thread_dict = {}
            # launch threads for all GPUs
            for i in range(self.n_jobs):
                t = threading.Thread(target=self.gpuworker, args=(self.evaluation_queue,
                                                                  self.available_gpus[i],))
                t.setDaemon = True
                thread_dict[i] = t
                t.start()

            # wait for queue to be empty and all threads to finish
            self.evaluation_queue.join()
            threads = [thread_dict[a] for a in thread_dict]
            for thread in threads:
                thread.join()

            print('\n\n All threads should now be done. Finishing program...\n\n')

            self.stop_dict['n_eval'] = self.eval_count
            self.stop_dict['n_iter'] = self.iter_count

            return self.incumbent, self.stop_dict

        else:

            while not self.check_stop():
                self.step()

            self.stop_dict['n_eval'] = self.eval_count
            self.stop_dict['n_iter'] = self.iter_count
            return self.incumbent, self.stop_dict

    def check_stop(self):
        # TODO: add more stop criteria
        # unify the design purpose of stop_dict
        if self.iter_count >= self.max_iter:
            self.stop_dict['max_iter'] = True

        if self.eval_count >= self.max_eval:
            self.stop_dict['max_eval'] = True

        return len(self.stop_dict)

    def _acquisition(self, plugin=None, dx=False, subsurrogate=None):
        # if plugin is None:
        #     # plugin = np.min(self.data.perf) if self.minimize else -np.max(self.data.perf)
        #     # Note that performance are normalized when building the surrogate
        #     plugin = 0 if self.minimize else -1
        # if surrogate is None:
        #     surrogate = self.surrogate;
        # if self.n_point > 1:  # multi-point method
        #     # create a portofolio of n infill-criteria by
        #     # instantiating n 't' values from the log-normal distribution
        #     # exploration and exploitation
        #     # TODO: perhaps also introduce cooling schedule for MGF
        #     # TODO: other method: niching, UCB, q-EI
        #     tt = np.exp(0.5 * np.random.randn())
        #     acquisition_func = MGFI(surrogate, plugin, minimize=self.minimize, t=tt)
        #     # acquisition_func is a callable object with input X and dx; __call__(self, X, dx=False)
        #     # consider acquisition_func as a function i.e. f_value=acquisition_func(X, dx)
        if self.n_point == 1: # sequential mode
            if self.infill == 'EI':
                # generate acquisition function from respective surrogate
                newplugin = None
                for alg in self.alg_name:
                    best_point = np.min(subsurrogate[alg].y)
                    if newplugin == None or best_point < newplugin:
                        newplugin = best_point
                for alg in self.alg_name:
                    self.acquisition_func[alg] = EI(subsurrogate[alg], newplugin, minimize=self.minimize)
                    self.acquisition_func[alg] = functools.partial(self.acquisition_func[alg], dx=dx)

            # if self.infill == 'EI':
            #     acquisition_func = EI(surrogate, plugin, minimize=self.minimize)
            # elif self.infill == 'PI':
            #     acquisition_func = PI(surrogate, plugin, minimize=self.minimize)
            # elif self.infill == 'MGFI':
            #     acquisition_func = MGFI(surrogate, plugin, minimize=self.minimize, t=self.t)
            #     self._annealling()
            # elif self.infill == 'UCB':
            #     raise NotImplementedError
        return self.acquisition_func # dictionary
        # return functools.partial(acquisition_func, dx=dx)
        
    def _annealling(self):
        if self.schedule == 'exp':  
             self.t *= self.alpha
        elif self.schedule == 'linear':
            self.t -= self.eta
        elif self.schedule == 'log':
            # TODO: verify this
            self.t = self.c / np.log(self.iter_count + 1 + 1)

## Notes: meta3 idea: each step, get 7 infill criteria values, get max one among them
    ## 7 infill criterian means 7 surrogates and 7 mies (also 7 pairs xopt_, fopt_)
    ## mies does not use multistart
    def _argmax_single_start(self, one_obj_func, alg): # acq_obj_func is not dictionary
        eval_budget = self._max_eval
        if self._optimizer == 'MIES':
            opt = mies(self.subspace[alg], one_obj_func, max_eval=eval_budget, minimize=False, verbose=False)
            xopt_, fopt_, stop_dict = opt.optimize()
            # convert Individual xopt_ to normal list
            xopt_ = [item for item in xopt_]
        return xopt_, fopt_

    def arg_max_acquisition(self, plugin=None, subsurrogate=None):
        """
        Global Optimization on the acquisition function 
        """
        if self.verbose:
            self.logger.info('acquisition function optimziation...')
        
        dx = True if self._optimizer == 'BFGS' else False
        # generate multiple acquisition functions using differenct surrogates, subsurrogate is dictionary
        obj_func = self._acquisition(plugin=plugin, dx=dx, subsurrogate=subsurrogate)
        # obj_func is dictionary

        ## add here
        if self.n_point == 1:
            if self.parallel_mies == False:
                multiple_xopt = []
                current_best_fopt = -99
                for alg in self.alg_name:
                    xopt, fopt= self._argmax_single_start(obj_func[alg], alg) # now xopt is a list
                    multiple_xopt = multiple_xopt + xopt
                    if fopt > current_best_fopt:
                        current_best_fopt = fopt
                        current_best_alg = alg

                multiple_xopt = [current_best_alg] + multiple_xopt

                candidates = multiple_xopt
                values = current_best_fopt

            elif self.parallel_mies == True:
                res = Parallel(n_jobs=len(self.alg_name), verbose=self.verbose)(
                    delayed(self._argmax_single_start, check_pickle=False)(func, alg) for func in obj_func.values() for alg in self.alg_name)
                # func is actually the fitness function used in MIES, f=func(x), input vector output scalar
                #TODO: _pickle.PicklingError: Could not pickle the task to send it to the workers.
                multiple_xopt, multiple_fopt = list(zip(*res))
                # max_fopt = np.max(multiple_fopt)
                # index = multiple_fopt.index(max_fopt)
                # max_xopt = multiple_xopt(index)

        elif self.n_point > 1:
            pass

        # if self.n_point == 1:
        #     # acquisition_func is EI(), PI(), MGFI() or UCB()
        #     # candidates, values refer to one point indeed
        #     candidates, values = self._argmax_multistart(obj_func)
        # else:
        #     res = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
        #         delayed(self._argmax_multistart, check_pickle=False)(func) for func in obj_func)
        #     # func is actually the fitness function used in MIES, f=func(x), input vector output scalar
        #     candidates, values = list(zip(*res))
            
        return candidates, values
                 # X, infill_value, in meta2, only one value indeed

    def _argmax_multistart(self, obj_func): # use mies to find optimum of fitness function obj_func
        # keep the list of optima in each restart for future usage
        xopt, fopt = [], []  
        eval_budget = self._max_eval # evaluation budget of MIES or BFGS; default value is 500 * self.dim, also can be set manually through max_infill_eval when instantiating mipego object   
        best = -np.inf
        wait_count = 0

        for iteration in range(self._random_start): # self._random_start = int(5 * self.dim) if n_restart is None else n_restart

            x0 = self._space.sampling(N=1, method='uniform')[0]
            
            # TODO: add IPOP-CMA-ES here for testing
            # TODO: when the surrogate is GP, implement a GA-BFGS hybrid algorithm
            if self._optimizer == 'BFGS':
                if self.N_d + self.N_i != 0:
                    raise ValueError('BFGS is not supported with mixed variable types.')
                # TODO: find out why: somehow this local lambda function can be pickled...
                # for minimization
                func = lambda x: tuple(map(lambda x: -1. * x, obj_func(x)))
                xopt_, fopt_, stop_dict = fmin_l_bfgs_b(func, x0, pgtol=1e-8, # why self._bounds contains continuous and integer instead of only continuous?   
                                                        factr=1e6, bounds=self._bounds, # self._bounds contains continuous and integer variable     
                                                        maxfun=eval_budget)
                xopt_ = xopt_.flatten().tolist()
                fopt_ = -np.asscalar(fopt_)
                
                if stop_dict["warnflag"] != 0 and self.verbose:
                    self.logger.warn("L-BFGS-B terminated abnormally with the "
                                     " state: %s" % stop_dict)
                                
            elif self._optimizer == 'MIES':  # obj_func is dictionary
                opt = mies(self._space, obj_func, max_eval=eval_budget, minimize=False, verbose=False)
                xopt_, fopt_, stop_dict = opt.optimize()

            if fopt_ > best:
                best = fopt_
                wait_count = 0
                if self.verbose:
                    self.logger.info('restart : {} - funcalls : {} - Fopt : {}'.format(iteration + 1, 
                        stop_dict['funcalls'], fopt_))
            else:
                wait_count += 1

            eval_budget -= stop_dict['funcalls']
            xopt.append(xopt_)
            fopt.append(fopt_)
            
            if eval_budget <= 0 or wait_count >= self._wait_iter:
                break
        # maximization: sort the optima in descending order
        idx = np.argsort(fopt)[::-1] # slice array a[start:end:step], so a[::-1] means all items in the array, reversed, i.e. from big to small
        return xopt[idx[0]], fopt[idx[0]]

    def _check_params(self):
        assert hasattr(self.obj_func, '__call__')

        if np.isinf(self.max_eval) and np.isinf(self.max_iter):
            raise ValueError('max_eval and max_iter cannot be both infinite')
                    
#if __name__ == '__main__':
#    data_file = 'test_datafile.log'
#    for i in range(5):
#        
#        incumbent = data[i].to_dict() 
#        df_incumbent = pd.DataFrame().append(incumbent, ignore_index=True)
#        #df = pd.DataFrame([incumbent], columns=incumbent.keys()) 
#        
#        df_incumbent.to_csv(data_file, header=False, index=False, mode='a')
#        
#        df = pd.DataFrame()
#        df = pd.DataFrame().append(incumbent, ignore_index=True)
#        df.to_csv(datafile, header=True, index=True, mode='w')
#        
#        df.to_csv(datafile, header=False, index=True, mode='a')
#        
#                
#        # initialize a dataframe
#        self.incumbent_df = pd.DataFrame([incumbent], columns=incumbent.keys())
#        # write incumbent file for the first time
#        self.incumbent_df.to_csv(self.data_file, header=True, index=False, mode='w')  
    








