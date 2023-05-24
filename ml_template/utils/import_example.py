## Deep leanring
import torch.nn as nn
from  torch.utils.data import Dataset,DataLoader
import torch
import torchvision

# Data agumentation
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Machine learning
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier,CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from imblearn.over_sampling import KMeansSMOTE,SMOTE,SVMSMOTE
from imblearn.under_sampling import EditedNearestNeighbours

# Regression model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, roc_auc_score, f1_score, recall_score, precision_score, mean_squared_error
from sklearn.linear_model import LinearRegression, BayesianRidge, LassoLars, SGDRegressor, PassiveAggressiveRegressor, TweedieRegressor,HuberRegressor, QuantileRegressor, TheilSenRegressor
from sklearn.svm import SVR, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neural_network import MLPRegressor


import pandas as pd
import numpy as np
import cv2
from time import sleep
from glob import glob
import os,sys
import warnings
warnings.filterwarnings("ignore")

# figure
import matplotlib.pyplot as plt
import seaborn as sns

# progress
from tqdm.notebook import tqdm

# sys.path.append('../input/coolshan-coding-utils')
from confusion_matrix_pretty_print import pp_matrix_from_data
from logger import Logger
from model_instance import Model_Instance
from loss_family import bi_tempered_binary_logistic_loss
from utils import setSeed, move_to,init_weights