
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, roc_auc_score, f1_score, recall_score, precision_score, mean_squared_error
from sklearn.linear_model import LinearRegression, BayesianRidge, LassoLars, SGDRegressor, SGDOneClassSVM, PassiveAggressiveRegressor, PassiveAggressiveClassifier, TweedieRegressor,MultiTaskElasticNet,HuberRegressor, QuantileRegressor, TheilSenRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, CompoundKernel, RBF, Sum, Matern, Exponentiation, PairwiseKernel
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from catboost import CatBoostRegressor ,CatBoostClassifier
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
from glob import glob
import pandas as pd
import numpy as np
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from ..utils import KFold_Sampler, bi_tempered_logistic_loss
from .weight_ensemble import *
from tqdm import tqdm
from .model_instance import *
from .ml_model import *

from torch import nn
import torch.optim as optim
warnings.filterwarnings('ignore')
import copy
from .basic_model import *