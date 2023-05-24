
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
from tqdm import tqdm
from .model_instance import *
from torch import nn
import torch.optim as optim
warnings.filterwarnings('ignore')
import copy
from .basic_model import  MLModels, Ensemble_Model


class Mean_Ensemble_Model(Ensemble_Model):
  def __init__(self, model_dict):
    super().__init__(model_dict)

  def ensemble_func(self, model_preds):
    return np.mean(model_preds,axis=1)


class Vote_Ensemble_Model(Ensemble_Model):
  def __init__(self, model_dict):
    super().__init__(model_dict)

  def ensemble_func(self, model_preds):
    return [ Counter(pred).most_common(1)[0][0] for pred  in model_preds]


class Stack_Ensemble_Model(Ensemble_Model):
  """
  overwrite: ensemble_func and fit
  """
  def __init__(self, model_dict,stack_model = SVC(C=0.1,probability=True),stack_training_split=0.3) -> None:
    self.stack_model = copy.deepcopy(stack_model)
    self.stack_training_split = stack_training_split
    super().__init__(model_dict)

  def stack_input_transform(self,model_preds):
    return model_preds

  def fit_stack_model(self,data,label):
    model_preds, model_dict_preds=self.model_predict_fn(data)
    model_preds = self.stack_input_transform(model_preds)
    self.stack_model.fit(model_preds,label)

  def fit(self, data, label):
    split_dict={0.1:(10,1),
                0.2:(5,1),
                0.15:(6,1),
                0.25:(4,1),
                0.3:(3,1),
                0.5:(2,1),}

    if self.stack_training_split not in split_dict.keys():
      model_data, model_label, stack_model_data, stack_model_label = KFold_Sampler(data,label,n_splits=100).get_multi_fold_data(n_fold=int(self.stack_training_split*100))
    else:
      model_data, model_label, stack_model_data, stack_model_label = KFold_Sampler(data,label,n_splits=split_dict[self.stack_training_split][0]).get_multi_fold_data(n_fold=split_dict[self.stack_training_split][1])

    _model = copy.deepcopy(MLModels(self.model_dict))
    _model.fit(model_data,model_label)

    if self.proba_mode:
      model_preds, model_dict_preds=_model._model_predicts_proba(stack_model_data)
    else:
      model_preds, model_dict_preds=_model._model_predicts(stack_model_data)

    model_preds = self.stack_input_transform(model_preds)
    self.stack_model.fit(model_preds,stack_model_label)
    print(self.stack_model.weights)
    super().fit(data,label)


  def ensemble_func(self,model_preds):
    model_preds = self.stack_input_transform(model_preds)
    return self.stack_model.predict(model_preds)

  def predict_proba(self, data):
    model_preds, model_dict_preds=self.model_predict_fn(data)
    model_preds = self.stack_input_transform(model_preds)
    return self.stack_model.predict_proba(model_preds)


class Stack_Ensemble_Proba_Model(Stack_Ensemble_Model):
  def __init__(self, model_dict,stack_model = SVC(C=0.1,probability=True),stack_training_split=0.1) -> None:
    super().__init__(model_dict, stack_model, stack_training_split)
    self.set_proba()

  def stack_input_transform(self,model_preds):
    return model_preds.reshape((model_preds.shape[0],-1))


class Mean_Ensemble_Proba_Model(Ensemble_Model):
  def __init__(self, model_dict):
    super().__init__(model_dict)
    super().set_proba()

  def ensemble_func(self,model_preds):
    return self._proba(model_preds).argmax(axis=-1)

class Weighted_Model_Instance(Model_Instance):

  def __init__(self,
                model,
                optimizer=None,
                scheduler=None,
                scheduler_epoch=False,
                loss_function=None,
                evaluation_metrics=lambda x,y : {},
                clip_grad=None,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                amp=False,
                accum_iter=1,
                model_weight_init=None,
                l1_norm=0,
                model_reg=0,
                classes_reg=0):
      super().__init__(model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scheduler_epoch=scheduler_epoch,
                loss_function=loss_function,
                evaluation_metrics=evaluation_metrics,
                clip_grad=clip_grad,
                device=device,
                amp=amp,
                accum_iter=accum_iter,
                model_weight_init=model_weight_init)
      self.model_reg=model_reg
      self.classes_reg=classes_reg
      self.l1_norm=l1_norm

  def get_loss(self,pred,label):
      loss_return = self.loss_function(pred,label)
      l1_regularization = torch.norm(self.model.weights, 1)/(self.model.num_classes*self.model.num_model)
      # l2_regularization = torch.norm(self.model.weights, 2)/self.model.weights.sum()
      model_regularization = torch.norm(self.model.weights.sum(dim=-1), 2)/self.model.weights.sum()
      classes_regularization = torch.norm(self.model.weights.sum(dim=-2), 2)/self.model.weights.sum()

      loss_return +=  self.model_reg*model_regularization +self.classes_reg*classes_regularization+self.l1_norm*l1_regularization
      if not isinstance(loss_return,tuple):
          loss_return = (loss_return,{'loss':loss_return.detach().to(torch.device('cpu'))})
      else:
          loss,loss_dict=loss_return

          if 'loss' not in loss_dict.keys():
              loss_dict['loss'] = loss
          loss_dict = {k:loss_dict[k].detach().to(torch.device('cpu').item()) for k in loss_dict.keys()}
          loss_return=(loss,loss_dict)
      return loss_return


class Weighted_Model(nn.Module):
  def __init__(self,num_model,num_classes) -> None:
    super(Weighted_Model, self).__init__()
    self.num_model=num_model
    self.num_classes=num_classes
    self.weights = nn.Parameter(torch.ones(num_model,num_classes)*0.5)
    self.active_fn = nn.ReLU()
    self.softmax = nn.Softmax(dim=-1)
    if num_classes >1:
      self.pred_fn = nn.Softmax(dim=-1)
    else:
      self.pred_fn = nn.Identity()


  def forward(self,data):
    self.weights.data=self.active_fn(self.weights.data)
    x = nn.Softmax(dim=-2)(self.weights).reshape(-1)*data#M*C
    x = x.view(-1,self.num_model,self.num_classes)
    x = x.sum(axis=1)
    x = self.pred_fn(x)
    return x

# class Fully_Weighted_Model(nn.Module):
#   def __init__(self,num_model,num_classes) -> None:
#     super(Weighted_Model, self).__init__()
#     self.num_model=num_model
#     self.num_classes=num_classes
#     self.weights = nn.Linear(num_model*num_classes,num_classes)
#     self.active_fn = nn.ReLU()
#     self.softmax = nn.Softmax(dim=-1)
#     if num_classes >1:
#       self.pred_fn = nn.Softmax(dim=-1)
#     else:
#       self.pred_fn = nn.Sigmoid()


#   def forward(self,data):
#     # self.weights.data=self.active_fn(self.weights.data)
#     x = self.weights(data)
#     x = self.pred_fn(x)
#     return x

class ML_Weighted_Model():
  def __init__(self,num_model, num_classes, lr=2e-3, epoch=None,model_reg=0,classes_reg=0,l1_norm=0) -> None:
    def loss_cls_fn(pred,label):
      one_hot_label = torch.nn.functional.one_hot(label,self.num_classes).to(torch.float32)
      return nn.BCELoss()(pred,one_hot_label) + bi_tempered_logistic_loss(pred,one_hot_label,0.8,1.2)
    def loss_reg_fn(pred,label):
      return nn.MSELoss()(pred,label.to(torch.float32))

    self.model = Weighted_Model(num_model,num_classes)
    self.epoch=(num_classes*num_model)*9 if epoch == None else epoch
    self.lr = lr
    self.num_classes = num_classes
    self.num_model = num_model
    self.load_stacking=False
    criterion = loss_reg_fn if num_classes ==1 else loss_cls_fn
    optimizer = optim.AdamW(self.model.parameters(),lr=self.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epoch)
    self.model_instance = Weighted_Model_Instance(model=self.model,
                                    optimizer=optimizer,
                                    loss_function=criterion,
                                    scheduler=scheduler,
                                    scheduler_epoch=False,
                                    device='cpu',
                                    clip_grad=1,
                                    amp=False,
                                    model_reg=model_reg,
                                    classes_reg=classes_reg,
                                    l1_norm=l1_norm)

  def load_weights(self,weights):
      self.load_stacking=True
      self.model.weights = nn.Parameter(weights)

  def predict(self,data):
    data = torch.tensor(data)
    pred = self.model_instance.inference(data)
    if self.num_classes >1:
      pred = torch.argmax(pred,axis=-1)
    return pred.numpy()

  def fit(self,data,label):
    if self.load_stacking:
      return
    data = torch.tensor(data)
    label = torch.tensor(label)
    for i in range(self.epoch):
      pred, (loss,eval) = self.model_instance.run_model(data,label,update=True)
    self.model_instance.inference(data)

  def predict_proba(self,data):
    data = torch.tensor(data)
    pred = self.model_instance.inference(data)
    return pred.numpy()

  @property
  def weights(self):
    return self.model_instance.model.weights#nn.Softmax(dim=-2)(self.model_instance.model.weights)

def get_stacking_df(cv_models):
  assert  isinstance(cv_models[0].stack_model, ML_Weighted_Model), 'only support ML_Weighted_Model'
  stacking_df=pd.DataFrame()
  for cv in range(len(cv_models)):
    weights = cv_models[cv].stack_model.weights.data.detach().numpy()
    for cls in range(weights.shape[-1]):
      cv_stacking_df=pd.DataFrame()
      cv_stacking_df['model'] = list(cv_models[0].model_list.keys())
      cv_stacking_df['cv']=cv
      cv_stacking_df['class']=cls
      cv_stacking_df['weights']= weights[:,cls]
      stacking_df = pd.concat([stacking_df,cv_stacking_df])
  stacking_df['mean_weights'] = stacking_df[stacking_df.columns[2:]].mean(axis=1)
  stacking_df = stacking_df.set_index(['model'],drop=True).loc[stacking_df.groupby(by=['model']).mean().sort_values(['mean_weights'],ascending=False).index].reset_index(drop=False)
  return stacking_df

def plot_cv_stacking_importance(stacking_df,columns_name):
  model_names=list(stacking_df.model.unique())
  fig, ax = plt.subplots(2,1,figsize=(8,12))
  sns.barplot(data=stacking_df, y='model',x='mean_weights',hue_order=model_names,ax=ax[0])
  stacking_df['class'] = stacking_df['class'].astype(str)
  ax[0].set_title('Model importance - mean')
  stacking_cls_df=stacking_df.groupby(by=['model','class'])[['weights']].mean().reset_index(drop=False).sort_values(['class','weights'],ascending = [True, False])
  sns.barplot(data=stacking_df, y='class',x='weights',hue='model',hue_order=model_names,ax=ax[1])
  ax[1].set_title('Model importance - class')
  plt.show()

  print(stacking_df.groupby(by=['model']).weights.mean().sort_values(ascending=False))
  print(stacking_cls_df.set_index('class'))

def plot_feature_importance(model,columns_name):
  importance = model.feature_importances_

  #Create arrays from feature importance and feature names
  feature_importance = np.array(importance)
  feature_names = np.array(columns_name)

  #Create a DataFrame using a Dictionary
  data={'feature_names':feature_names,'feature_importance':feature_importance}
  fi_df = pd.DataFrame(data)

  #Sort the DataFrame in order decreasing feature importance
  fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

  #Define size of bar plot
  plt.figure(figsize=(10,8))
  #Plot Searborn bar chart
  sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
  #Add chart labels

  plt.xlabel('FEATURE IMPORTANCE')
  plt.ylabel('FEATURE NAMES')
  plt.show()

  return  fi_df

def regression_model():

  model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                learning_rate=0.05, n_estimators=720,
                                max_bin = 55, bagging_fraction = 0.8,
                                bagging_freq = 5, feature_fraction = 0.2319,
                                feature_fraction_seed=9, bagging_seed=9,
                                min_data_in_leaf =6, min_sum_hessian_in_leaf = 11,
                                is_unbalance=True)

  kernel =  WhiteKernel(noise_level=3.0) + DotProduct() * RBF() * Matern() * RBF(length_scale=2.0)  #+Exponentiation(kernel=RBF(length_scale=2.0),exponent=2)
  gpr = GaussianProcessRegressor(kernel=kernel)
  reg = make_pipeline(StandardScaler(),
                        SGDRegressor(max_iter=500, tol=5e-4))

  model_dict = {'RF_3':RandomForestRegressor(n_estimators=310,max_depth=3),
              'RF_depth_None':RandomForestRegressor(n_estimators=310),
              'XGB_31_3':XGBRegressor(n_estimators=31,max_depth=3),
              'XGB_310_3':XGBRegressor(n_estimators=31,max_depth=3),
              'XGB_31':XGBRegressor(n_estimators=31),
              'XGB_310':XGBRegressor(n_estimators=310),
              'Bayesian':BayesianRidge(),
              #'GP_Reg':gpr,
              'Huber_Reg':HuberRegressor(),
              'SVM':SVR(),
              'SVM_lin':SVR(kernel='linear'),
              'SVM_poly':SVR(kernel='poly'),
              'SVM_0.2':SVR(C=0.2),
              'SVM_0.2_lin':SVR(C=0.2,kernel='linear'),
              'SVM_0.2_poly':SVR(C=0.2,kernel='poly'),
              'SVM_5':SVR(C=5),
              'SVM_5_lin':SVR(C=5,kernel='linear'),
              'SVM_5_poly':SVR(C=5,kernel='poly'),
              'LR':LinearRegression(),
              'KR':KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
              'KNN_Reg':KNeighborsRegressor(),
              'LGB_Reg':model_lgb,
              'MLP_Reg':MLPRegressor(activation = "relu", alpha = 0.1, hidden_layer_sizes = (5,5),
                            learning_rate = "constant", max_iter = 3000, random_state = 1000),
              'SGD_Reg':reg,
              }
  stack_model_dict= {
                'SVM_Linear_Reg':SVR(kernel='linear'),
                'Bayesian':BayesianRidge(),
                'MLP':MLPRegressor(activation = "relu", alpha = 0.1, hidden_layer_sizes = (5,),
                              learning_rate = "constant", max_iter = 1000, random_state = 1000),
                'Huber_Reg':HuberRegressor(),
                'LR':LinearRegression(),
                'XGB_Reg':XGBRegressor(n_estimators=11,max_depth=2),
                'RF_Reg':RandomForestRegressor(n_estimators=31,max_depth=2),
                }
  return model_dict

def classification_model():

  model_lgb = lgb.LGBMClassifier(objective='regression',num_leaves=5,
                                learning_rate=0.05, n_estimators=720,
                                max_bin = 55, bagging_fraction = 0.8,
                                bagging_freq = 5, feature_fraction = 0.2319,
                                feature_fraction_seed=9, bagging_seed=9,
                                min_data_in_leaf =6, min_sum_hessian_in_leaf = 11,
                                is_unbalance=True)

  kernel =  WhiteKernel(noise_level=3.0) + DotProduct() * RBF() * Matern() * RBF(length_scale=2.0)  #+Exponentiation(kernel=RBF(length_scale=2.0),exponent=2)
  gpr = GaussianProcessClassifier(kernel=kernel)
  reg = make_pipeline(StandardScaler(),
                        SGDOneClassSVM(max_iter=500, tol=5e-4))

  model_dict = {
              'RF_3':RandomForestClassifier(n_estimators=310,max_depth=3),
              'RF_depth_None':RandomForestClassifier(n_estimators=310),
              'XGB_31_3':XGBClassifier(n_estimators=31,max_depth=3),
              'XGB_310_3':XGBClassifier(n_estimators=31,max_depth=3),
              'XGB_31':XGBClassifier(n_estimators=31),
              'XGB_310':XGBClassifier(n_estimators=310),
              'Bayesian':BayesianRidge(),
              'GP_Cls':gpr,
              'RC_Cls':RidgeClassifier(),
              'SVM':SVC(probability=True),
              'SVM_lin':SVC(kernel='linear',probability=True),
              'SVM_poly':SVC(kernel='poly',probability=True),
              'SVM_0.2':SVC(C=0.2,probability=True),
              'SVM_0.2_lin':SVC(C=0.2,kernel='linear',probability=True),
              'SVM_0.2_poly':SVC(C=0.2,kernel='poly',probability=True),
              'SVM_5':SVC(C=5,probability=True),
              'SVM_5_lin':SVC(C=5,kernel='linear',probability=True),
              'SVM_5_poly':SVC(C=5,kernel='poly',probability=True),
              'KR':KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
              'KNN_Cls':KNeighborsClassifier(),
              'LGB_Cls':model_lgb,
              'MLP_Cls':MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (5,5),
                            learning_rate = "constant", max_iter = 3000, random_state = 1000),
              'SGD_Cls':reg,
              'QDA':QuadraticDiscriminantAnalysis(),
              }
  stack_model_dict= {
                'SVM_Linear_Reg':SVC(kernel='linear'),
                'MLP':MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (5,),
                              learning_rate = "constant", max_iter = 1000, random_state = 1000),
                'XGB_Reg':XGBClassifier(n_estimators=11,max_depth=2),
                'RF_Reg':RandomForestClassifier(n_estimators=31,max_depth=2),
                'QDA':QuadraticDiscriminantAnalysis(),
                }
  return model_dict



''' DEMO
reg_model = get_reg_ensemble_model()
stack_model_dict= {
              'SVM_Linear_Reg':SVR(kernel='linear'),
              'Bayesian':BayesianRidge(),
              'Huber_Reg':HuberRegressor(),
              'LR':LinearRegression(),
              }

evaluation_fn = calculate_metrics(['mse'])
model = Stack_Ensemble_Model(model_dict,stack_model=Mean_Ensemble_Model(stack_model_dict))
cv_models, cv_df = model.cross_validation_evaluate(data, label, evaluation_fn, fold=6, verbose=False)
cv_ensemble_model = Mean_Ensemble_Model(cv_models)

'''
