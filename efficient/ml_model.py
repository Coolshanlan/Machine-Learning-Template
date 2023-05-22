
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
from .utils import KFold_Sampler
from tqdm import tqdm
from .model_instance import *
from torch import nn
import torch.optim as optim
warnings.filterwarnings('ignore')
import copy
## TODO
"""
weighted sum and NN weighted sum
"""



def eval_dict_to_dataframe(eval_dict):
    record_df=pd.DataFrame()
    for model_name, eval_metrics in eval_dict.items():
      if not isinstance(eval_metrics, dict):
        eval_metrics = {'eval_metric': eval_metrics}
      row_df = pd.DataFrame(eval_metrics,index=[0])
      row_df['model'] = model_name
      record_df = pd.concat([record_df,row_df])
    record_df=record_df.sort_values(by=record_df.columns[:-1].to_list())
    record_df= record_df.reset_index(drop=True)
    record_df = record_df[['model']+record_df.columns[:-1].to_list()]
    return record_df

class MLModels():
  """
  predict output:
  (ensemble_pred, model_pred_dict)

  model_predicts output:
  (model_predicts(n_sample, num_models), model_pred_dict)

  evaluation_fn:
  (ensemble_pred, model_dict_preds, eval_dict)

  ensemble_func(model_preds):
  model_preds shape (N_sample, num_model)


  Predict (predict):
  predict -> model_predicts(get all model predict output)
  -> transform_dict_predict -> ensemble_func


  Eval (evaluation_fn):
  predict -> evaluation_func
  return ensemble_predict, model_dict_predict, eval_dict
  """
  def __init__(self,models) -> None:
    models = copy.deepcopy(models)
    self.cv_models=None
    if isinstance(models,list):
      self.model_dict={}
      for midx, model in enumerate(models):
        self.model_dict[f'model_{midx+1}']= model
    elif not isinstance(models,dict):
      self.model_dict={}
      self.model_dict[f'model_{1}']= models
    else:
      self.model_dict = models

  def fit(self,data,label):
    pbar = tqdm(self.model_dict.items(), total=self.num_models, leave=False, bar_format='{desc:<30}\t{percentage:2.0f}%|{bar:10}{r_bar}')
    for model_name, model in pbar:
      pbar.set_description(f'{model_name} Training...')
      model.fit(data,label)

  def predicts(self,data):
    return self.model_predicts(data)

  def predicts_proba(self,data):
    return self.model_predicts_proba(data)

  def _model_predicts(self,data):
    model_dict_preds={}
    for model_name, model in self.model_dict.items():
      pred = model.predict(data)
      model_dict_preds[model_name]=pred

    model_preds = self.transform_dict_preds(model_dict_preds)
    return model_preds, model_dict_preds

  def model_predicts(self,data):
    return self._model_predicts(data)

  def _model_predicts_proba(self,data):
    model_dict_preds={}
    for model_name, model in self.model_dict.items():
      pred = model.predict_proba(data)
      model_dict_preds[model_name]=pred
    model_preds = self.transform_dict_preds_proba(model_dict_preds)
    return model_preds, model_dict_preds

  def model_predicts_proba(self,data):
    return self._model_predicts_proba(data)

  def transform_dict_preds(self,preds):
    return np.array(list(preds.values())).T #(num_data, num_model)

  def transform_dict_preds_proba(self,preds):
    outcome=np.hstack(np.array(list(preds.values())))# (num_data, num_model*classes)
    outcome=outcome.reshape((-1,self.num_models,np.array(list(preds.values())).shape[-1]))
    return outcome #(num_data, num_model, classes)

  def evaluate(self,data, label, evaluation_fn, verbose=True):
    _, model_dict_preds = self.model_predicts(data)

    eval_dict={}

    for model_name, pred in model_dict_preds.items():
      eval_metric = evaluation_fn(pred,label)
      eval_dict[model_name]=eval_metric

    eval_df = eval_dict_to_dataframe(eval_dict)
    if verbose:
      print(eval_df)

    return {'model_predict':model_dict_preds,
            'eval_df':eval_df}


  def cross_validation_evaluate(self,data, label, evaluation_fn, n_splits=5,n_repeats=1, verbose=True):
    '''
    evaluation_fn
    '''

    record_df = pd.DataFrame()
    cv_models = []
    kfc = KFold_Sampler(data, label, n_splits=n_splits, n_repeats=n_repeats)
    eval_columns=None

    for i,(cv_training_data, cv_training_label, cv_validation_data, cv_validation_label) in enumerate(kfc.splits()):
      model = copy.deepcopy(self)

      model.fit(cv_training_data,cv_training_label)

      eval_dict = model.evaluate(cv_validation_data, cv_validation_label ,evaluation_fn,verbose=False)
      eval_df = eval_dict['eval_df']

      if eval_columns is None:
        eval_columns=eval_df.columns[1:].to_list()

      if verbose:
        print(f'\n\n====== CV:{i} ======')
        print(eval_df)

      eval_df['fold'] = i
      record_df = pd.concat([record_df,eval_df])
      cv_models.append(model)

    record_df = record_df.reset_index(drop=True)
    print(f'\n====== CV Mean ======')
    print(record_df.groupby(['model']).mean().drop(columns=['fold']).sort_values(eval_columns).reset_index(drop=False))
    self.cv_models=cv_models
    return cv_models, record_df

  def __len__(self):
    return len(self.model_dict.keys())

  @property
  def num_models(self):
    return len(list(self.model_dict.keys()))


class Ensemble_Model(MLModels):
  def __init__(self, models, ensemble_fn=None) -> None:
    super().__init__(models)

    self.model_predict_fn=super()._model_predicts
    self.proba_mode=False
    if ensemble_fn:
      self.ensemble_func=ensemble_fn

  def set_proba(self):
    self.proba_mode=True
    self.model_predict_fn = super()._model_predicts_proba
    self.remove_no_prob_model()

  def remove_no_prob_model(self):
    model_dict_tmp = copy.deepcopy(self.model_dict)
    for model_name, model in self.model_dict.items():
      if 'predict_proba' not in model.__dir__():
        print(f"{model_name} don't have [predict_proba]")
        del model_dict_tmp[model_name]
    self.model_dict = model_dict_tmp

  def ensemble_func(self,model_pred):
    raise NotImplementedError

  def _proba(self,model_preds):
    return model_preds.mean(axis=-2)

  def predict(self,data):
    model_preds, model_dict_pred=self.model_predicts(data)
    return model_preds

  def predict_proba(self,data):
    model_preds, model_dict_pred=self.model_predicts_proba(data)
    return model_preds

  def model_predicts(self,data):
    model_preds, model_dict_preds=self.model_predict_fn(data)
    if self.proba_mode:
      for model, pred in model_dict_preds.items():
        model_dict_preds[model] = pred.argmax(axis=-1)
    model_dict_preds['Ensemble Model']= self.ensemble_func(model_preds)
    return model_dict_preds['Ensemble Model'], model_dict_preds

  def model_predicts_proba(self,data):
    model_preds, model_dict_preds=super()._model_predicts_proba(data)
    model_dict_preds['Proba Model'] = self._proba(model_preds)
    return model_dict_preds['Proba Model'], model_dict_preds


class Stack_Ensemble_Model(Ensemble_Model):
  """
  overwrite: ensemble_func and fit
  """
  def __init__(self, model_dict,stack_model = SVC(C=0.1,probability=True),stack_training_split=0.1) -> None:
    self.stack_model = copy.deepcopy(stack_model)
    self.stack_training_split = stack_training_split
    super().__init__(model_dict)

  def stack_input_transform(self,model_preds):
    return model_preds

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
    super().fit(model_data,model_label)

    model_preds, model_dict_preds=self.model_predict_fn(stack_model_data)
    model_preds = self.stack_input_transform(model_preds)
    self.stack_model.fit(model_preds,stack_model_label)

  def ensemble_func(self,model_preds):
    model_preds = self.stack_input_transform(model_preds)
    return self.stack_model.predict(model_preds)

  def predict_proba(self, data):
    model_preds, model_dict_preds=self.model_predict_fn(data)
    model_preds = self.stack_input_transform(model_preds)
    return self.stack_model.predict_proba(model_preds)


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
                model_weight_init=None):
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

  def get_loss(self,pred,label):
      loss_return = self.loss_function(pred,label)
      #引入調和平均數，讓model weight 一起縮放
      # l1_regularization = 1 * torch.norm(torch.prod(self.model.weights.data,dim=-1)/torch.sum(self.model.weights.data,dim=-1), 1)
      l1_regularization = 1 * torch.norm(self.model.weights.data, 1)
      loss_return += l1_regularization
      if not isinstance(loss_return,tuple):
          loss_return = (loss_return,{'loss':loss_return.detach().to(torch.device('cpu'))})
      else:
          loss,loss_dict=loss_return

          # display in console
          if 'loss' not in loss_dict.keys():
              loss_dict['loss'] = loss
          loss_dict = {k:loss_dict[k].detach().to(torch.device('cpu').item()) for k in loss_dict.keys()}
          loss_return=(loss,loss_dict)
      return loss_return



class Weighted_Model(nn.Module):
  def __init__(self,num_model,num_classes,init_mode='rand',init_value=0.1) -> None:
    super(Weighted_Model, self).__init__()
    self.init_mode=init_mode
    self.num_model=num_model
    self.num_classes=num_classes
    self.init_value=init_value
    if self.init_mode =='rand':
      self.weights = nn.Parameter(torch.abs(torch.rand(num_model,num_classes))+init_value)
    else:
      self.weights = nn.Parameter(torch.abs(torch.ones(num_model,num_classes)))
    self.active_fn = nn.ReLU()
    self.softmax = nn.Softmax(dim=-1)
    if num_classes >1:
      self.pred_fn = nn.Softmax(dim=-1)
    else:
      self.pred_fn = nn.Sigmoid()

  def forward(self,data):
    self.weights.data=self.active_fn(self.weights.view(-1).data).reshape(self.num_model,self.num_classes)
    x = self.weights.view(-1)*data#M*C
    x = self.active_fn(x)
    x = x.view(-1,self.num_model,self.num_classes)
    x = x.sum(axis=1)
    x = torch.div(x,torch.sum(self.weights.T,dim=-1))
    x = self.pred_fn(x)
    return x

class ML_Weighted_Model():
  def __init__(self,num_model, num_classes, lr=1e-3, epoch=500,init_mode='rand',init_value=0.1) -> None:
    self.model = Weighted_Model(num_model,num_classes,init_mode=init_mode,init_value=init_value)
    self.epoch=epoch
    self.lr = lr
    self.num_classes = num_classes
    self.num_model = num_model
    self.init_value=init_value
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(self.model.parameters(),lr=self.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epoch)
    self.model_instance = Weighted_Model_Instance(model=self.model,
                                    optimizer=optimizer,
                                    loss_function=criterion,
                                    scheduler=scheduler,
                                    scheduler_epoch=False,
                                    device='cpu',
                                    #clip_grad=1,
                                    amp=False,
                                    )#model_weight_init='normal')

  def predict(self,data):
    data = torch.tensor(data)
    pred = self.model_instance.inference(data)
    pred = torch.argmax(pred,axis=-1)
    return pred.numpy()

  def fit(self,data,label):
    data = torch.tensor(data)
    label = torch.tensor(label)
    for i in range(self.epoch):
      pred, (loss,eval) = self.model_instance.run_model(data,label,update=True)

  def predict_proba(self,data):
    data = torch.tensor(data)
    pred = self.model_instance.inference(data)
    return pred.numpy()

  @property
  def weights(self):
    return self.model_instance.model.weights





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
