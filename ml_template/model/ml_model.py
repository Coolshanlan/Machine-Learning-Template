
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
    if not (isinstance(self.stack_model,ML_Weighted_Model) and self.stack_model.load_stacking ):

      if self.stack_training_split not in split_dict.keys():
        model_data, model_label, stack_model_data, stack_model_label = KFold_Sampler(data,label,n_splits=100).get_multi_fold(n_fold=int(self.stack_training_split*100))
      else:
        model_data, model_label, stack_model_data, stack_model_label = KFold_Sampler(data,label,n_splits=split_dict[self.stack_training_split][0]).get_multi_fold(n_fold=split_dict[self.stack_training_split][1])

      _model = copy.deepcopy(MLModels(self.model_dict))
      _model.fit(model_data,model_label)

      if self.proba_mode:
        model_preds, model_dict_preds=_model._model_predicts_proba(stack_model_data)
      else:
        model_preds, model_dict_preds=_model._model_predicts(stack_model_data)
      # evaluation_fn = calculate_metrics(['recall','precision','acc','f1score'])
      print('===== Model pre-training evaluation =====')
      _model.evaluate(stack_model_data,stack_model_label,evaluation_fn=mean_absolute_error)
      model_preds = self.stack_input_transform(model_preds)
      self.stack_model.fit(model_preds,stack_model_label)
      if isinstance(self.stack_model,ML_Weighted_Model):
        for midx,model_name in enumerate(list(_model.model_dict.keys())):
          print(model_name,':',np.array(self.stack_model.weights[midx,:]))


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
    self.weights = nn.Parameter(torch.ones(num_model,num_classes))
    self.active_fn = nn.ReLU()
    self.softmax = nn.Softmax(dim=-1)
    # if num_classes >1:
    #   self.pred_fn = nn.Softmax(dim=-1)
    # else:
    #   self.pred_fn = nn.Identity()


  def forward(self,data):
    self.weights.data=self.active_fn(self.weights.data)
    # x = nn.Softmax(dim=-2)(self.weights).reshape(-1)*data#M*C
    x = (self.weights/self.weights.sum(dim=-2)).reshape(-1)*data
    x = x.view(-1,self.num_model,self.num_classes)
    x = x.sum(axis=1)
    # x = self.pred_fn(x)
    return x

class Fully_Weighted_Model(nn.Module):
  def __init__(self,num_model,num_classes) -> None:
    super(Weighted_Model, self).__init__()
    self.num_model=num_model
    self.num_classes=num_classes
    self.weights = nn.Linear(num_model*num_classes,num_classes)
    self.active_fn = nn.ReLU()
    self.softmax = nn.Softmax(dim=-1)
    if num_classes >1:
      self.pred_fn = nn.Softmax(dim=-1)
    else:
      self.pred_fn = nn.Sigmoid()


  def forward(self,data):
    # self.weights.data=self.active_fn(self.weights.data)
    x = self.weights(data)
    x = self.pred_fn(x)
    return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class ML_Weighted_Model():
  def __init__(self,num_model, num_classes, lr=2e-3, epoch=None,model_reg=0,classes_reg=0,l1_norm=0) -> None:
    def loss_cls_fn(pred,label):
      label = label.to(device)
      ce_loss = nn.CrossEntropyLoss()(pred,label)

      # one_hot_label = torch.nn.functional.one_hot(label,self.num_classes).to(torch.float32)
      return ce_loss #FocalLoss()(pred,one_hot_label)+ce_loss + bi_tempered_logistic_loss(pred,one_hot_label,0.8,1.2)
    def loss_reg_fn(pred,label):
      label = label.to(device)
      return nn.MSELoss()(pred,label.to(torch.float32))

    self.model = Weighted_Model(num_model,num_classes)
    self.load_stacking=False
    self.epoch=(num_classes*num_model)*9 if epoch == None else epoch
    # self.epoch=1 if epoch == None else epoch

    self.lr = lr
    self.num_classes = num_classes
    self.num_model = num_model
    criterion = loss_reg_fn if num_classes ==1 else loss_cls_fn
    optimizer = optim.AdamW(self.model.parameters(),lr=self.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epoch)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model_instance = Weighted_Model_Instance(model=self.model,
                                    optimizer=optimizer,
                                    loss_function=criterion,
                                    scheduler=scheduler,
                                    scheduler_epoch=False,
                                    device=device,
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
    else:
      pred = pred.reshape(-1)
    return pred.cpu().detach().numpy()

  def fit(self,data,label):
    if self.load_stacking:
      return
    data = torch.tensor(data)
    label = torch.tensor(label)
    for i in range(self.epoch):
      # for _data, _label in zip(data,label):
        # _data = torch.tensor(_data).view(1,-1)
        # _label = torch.tensor(_label).view(-1)
      pred, (loss,eval) = self.model_instance.run_model(data,label,update=True)
    self.model_instance.inference(data)

  def predict_proba(self,data):
    data = torch.tensor(data)
    pred = self.model_instance.inference(data)
    return nn.Softmax(dim=-1)(pred).cpu().detach().numpy()

  @property
  def weights(self):
    # print(self.model_instance.model.weights.cpu().detach())
    # return torch.round(nn.Softmax(dim=-2)(self.model_instance.model.weights.cpu().detach()),decimals=3)#nn.Softmax(dim=-2)(self.model_instance.model.weights)
    w = self.model_instance.model.weights.cpu().detach()
    return torch.round(w/w.sum(dim=-2),decimals=3)#nn.Softmax(dim=-2)(self.model_instance.model.weights)



def weighted_stacking_analysis(cv_models,
                              feature_columns,
                              fig1_size=(8,15),
                              fig2_size=(30,30),
                              fig3_size=(8,20)):

  if not isinstance(cv_models,list):
    cv_models = [cv_models]

  return_dict={}

  assert  isinstance(cv_models[0].stack_model, ML_Weighted_Model), 'only support ML_Weighted_Model'
  #==================== Model Weights ========================
  stacking_df=pd.DataFrame()
  for cv in range(len(cv_models)):
    weights = cv_models[cv].stack_model.weights.data.detach().numpy()
    for cls in range(weights.shape[-1]):
      cv_stacking_df=pd.DataFrame()
      cv_stacking_df['model'] = list(cv_models[0].model_dict.keys())
      cv_stacking_df['cv']=cv
      cv_stacking_df['class']=cls
      cv_stacking_df['weights']= weights[:,cls]
      stacking_df = pd.concat([stacking_df,cv_stacking_df])
  stacking_df['mean_weights'] = stacking_df.groupby('model').weights.transform(lambda x: x.mean())
  stacking_df = stacking_df.set_index(['model'],drop=True).loc[stacking_df.groupby(by=['model']).mean().sort_values(['mean_weights'],ascending=False).index].reset_index(drop=False)
  return_dict['stacking_weights_df']=stacking_df

  pd.set_option('display.max_rows', 500)
  #============== Model Classes Importance Plot=================
  if not os.path.exists('./figure'):
    os.mkdir('./figure')
  model_names=list(stacking_df.model.unique())
  fig, ax = plt.subplots(2,1,figsize=fig1_size)
  sns.barplot(data=stacking_df, y='model',x='mean_weights',hue_order=model_names,ax=ax[0])
  stacking_df['class'] = stacking_df['class'].astype(str)
  ax[0].set_title('Model importance - mean')

  sns.barplot(data=stacking_df, y='class',x='weights',hue='model',hue_order=model_names,ax=ax[1])
  ax[1].set_title('Model importance - class')
  plt.savefig('figure/Model_classes_importance.png')
  plt.show()

  stacking_cls_df=stacking_df.groupby(by=['model','class']).weights.mean().reset_index(drop=False).sort_values(['class','weights'],ascending = [True, False]).set_index(['class','model'])
  stacking_mean_df = stacking_df.groupby(by=['model']).weights.mean().sort_values(ascending=False)
  print('=========== Model Classes Importance ============')
  print(stacking_mean_df)
  print(stacking_cls_df)
  return_dict['model_importance_mean_df']=stacking_mean_df
  return_dict['model_class_importance_mean_df']=stacking_cls_df

  #============== Feature Classes Importance Plot=================
  feature_importance_df = pd.DataFrame()
  for model_name , model in cv_models[0].model_dict.items():
    if 'feature_importances_' not in cv_models[0].model_dict[model_name].__dir__():
      continue
    feature_model_importance_df=pd.DataFrame()
    feature_model_importance_df['feature_name']=feature_columns
    feature_model_importance_df['model']=model_name
    importance_list=np.array([0.0]*len(feature_columns))
    for cv in range(len(cv_models)):
      importance = cv_models[cv].model_dict[model_name].feature_importances_
      importance = importance/sum(importance)
      importance_list+=np.array(importance)
    importance = importance_list/len(cv_models)
    for cls in range(stacking_cls_df.reset_index(drop=False)['class'].nunique()):
      feature_model_importance_df['importance'] = importance * stacking_cls_df.loc[str(cls)].loc[model_name]['weights']
      feature_model_importance_df['class'] = str(cls)
      feature_importance_df = pd.concat([feature_importance_df,feature_model_importance_df])
  feature_importance_df['mean_importance'] = feature_importance_df.groupby('feature_name').importance.transform(lambda x: x.mean())
  return_dict['feature_importance_df']=feature_importance_df

  print('=========== Feature Importance ============')

  fig, ax = plt.subplots(1,2,figsize=fig2_size)

  display_feature_model_df=feature_importance_df.groupby(by=['feature_name','model']).importance.mean().reset_index(drop=False)
  display_feature_model_df['mean_importance'] = display_feature_model_df.groupby('feature_name').importance.transform(lambda x: x.mean())
  display_feature_model_df = display_feature_model_df.sort_values('mean_importance',ascending=False)
  if display_feature_model_df.model.nunique()<=10:
    colors = sns.color_palette('tab10',display_feature_model_df.model.nunique())
  else:
    colors = sns.color_palette('tab20',display_feature_model_df.model.nunique())
  stack_list=None
  n_model=display_feature_model_df.model.nunique()
  model_list = display_feature_model_df.groupby('model').importance.mean().sort_values(ascending=False).index

  for midx,model in enumerate(model_list):
    if stack_list is None :
      ax[0].barh(display_feature_model_df.loc[display_feature_model_df.model == model,'feature_name'],\
        display_feature_model_df.loc[display_feature_model_df.model == model,'importance']/n_model,\
          color=colors[midx])
      stack_list=display_feature_model_df.loc[display_feature_model_df.model == model,'importance']/n_model
    else:
      ax[0].barh(display_feature_model_df.loc[display_feature_model_df.model == model,'feature_name'],\
        display_feature_model_df.loc[display_feature_model_df.model == model,'importance']/n_model,\
          left=stack_list,color=colors[midx])
      stack_list +=  display_feature_model_df.loc[display_feature_model_df.model == model,'importance'].values/n_model
  ax[0].legend(model_list)

  display_feature_model_df=display_feature_model_df.groupby(['feature_name']).importance.mean().reset_index(drop=False).sort_values(['importance'],ascending=False)
  # sns.lineplot(data=display_feature_model_df, y='feature_name',x='importance',color='#3caea3',linewidth=3,ax=ax[0])
  ax[0].set_title('Feature Importance each model ')
  ax[0].invert_yaxis()

    #==================plot2
  display_feature_importance_df = feature_importance_df.groupby(by=['feature_name','class']).importance.mean().reset_index(drop=False)
  display_feature_importance_df['mean_importance'] = display_feature_importance_df.groupby('feature_name').importance.transform(lambda x: x.mean())
  display_feature_importance_df = display_feature_importance_df.sort_values('mean_importance',ascending=False)

  if display_feature_importance_df['class'].nunique()<=10:
    colors = sns.color_palette('tab10',display_feature_importance_df['class'].nunique())
  else:
    colors = sns.color_palette('tab20',display_feature_importance_df['class'].nunique())
  stack_list=None
  n_class=display_feature_importance_df['class'].sort_values().unique()

  for cidx,classname in enumerate(n_class):
    if stack_list is None :
      ax[1].barh(display_feature_importance_df.loc[display_feature_importance_df['class'] == classname,'feature_name'],\
        display_feature_importance_df.loc[display_feature_importance_df['class'] == classname,'importance'],\
          color=colors[cidx])
      stack_list=display_feature_importance_df.loc[display_feature_importance_df['class'] == classname,'importance']
    else:
      ax[1].barh(display_feature_importance_df.loc[display_feature_importance_df['class'] == classname,'feature_name'],\
        display_feature_importance_df.loc[display_feature_importance_df['class'] == classname,'importance'],\
          left=stack_list,color=colors[cidx])
      stack_list +=  display_feature_importance_df.loc[display_feature_importance_df['class'] == classname,'importance'].values
  ax[1].legend(n_class)
  ax[1].invert_yaxis()

  # sns.barplot(data=display_feature_importance_df, y='feature_name',x='importance',hue='class',hue_order=list(feature_importance_df['class'].unique()),ax=ax[1])
  # _display_feature_importance_df=display_feature_importance_df.groupby(['feature_name']).importance.mean().reset_index(drop=False).sort_values(['importance'],ascending=False)
  # sns.lineplot(data=_display_feature_importance_df.reset_index(drop=False).sort_values(['importance'],ascending=False), y='feature_name',x='importance',color='#3caea3',linewidth=3,ax=ax[1])

  ax[1].set_title('Feature importance each classes')
  plt.savefig('figure/Feature_importance_each_classes_and_model.png')
  plt.show()

  fig, ax = plt.subplots(1,1,figsize=fig3_size)
  filter_display_feature_importance_df=display_feature_importance_df.sort_values(['class','importance'],ascending=[True,False]).groupby(by=['class']).head(10).set_index(['class','feature_name'])
  filter_display_feature_importance_df['mean_importance'] = filter_display_feature_importance_df.groupby('feature_name').importance.transform(lambda x: x.mean())
  filter_display_feature_importance_df = filter_display_feature_importance_df.sort_values('mean_importance',ascending=False)
  sns.barplot(data=filter_display_feature_importance_df.reset_index(drop=False), y='feature_name',x='importance',hue='class',hue_order=list(feature_importance_df['class'].unique()))
  ax.set_title('Each class top 10 features')
  plt.savefig('figure/Each_class_top_10_features.png')
  plt.show()
  print(filter_display_feature_importance_df.reset_index(drop=False).sort_values(['class','importance'],ascending=[True,False]))

  return return_dict

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
