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