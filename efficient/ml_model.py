
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
from .utils import K_Fold_Creator
warnings.filterwarnings('ignore')
import copy


class Ensemble_Model():
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
  def __init__(self,models, ensemble_fn=None) -> None:

    if isinstance(models,list):
      self.model_list={}
      for midx, model in enumerate(models):
        self.model_list[f'model_{midx+1}']= model
    else:
      self.model_list = models

    if ensemble_fn:
      self.ensemble_func= ensemble_fn

  @property
  def num_models(self):
    return len(list(self.model_list.keys()))

  def ensemble_func(self,model_preds):
    raise ImportError('Not implement ensemble funct')

  def fit(self,data,label):
    for model_name, model in self.model_list.items():
      model.fit(data,label)


  def model_predicts(self,data):
    model_dict_preds={}
    for model_name, model in self.model_list.items():
      pred = model.predict(data)
      model_dict_preds[model_name]=pred
    model_preds = self.transform_dict_preds(model_dict_preds)
    return model_preds, model_dict_preds

  def model_predicts_proba(self,data):
    model_dict_preds={}
    for model_name, model in self.model_list.items():
      pred = model.predict_proba(data)
      model_dict_preds[model_name]=pred
    model_preds = self.transform_dict_preds_proba(model_dict_preds)
    return model_preds, model_dict_preds

  def predict(self,data):
    model_preds, _=self.model_predicts(data)
    return self.ensemble_func(model_preds)

  def predict_proba(self,data):
    model_preds, _=self.model_predicts_proba(data)
    return self.ensemble_func(model_preds)

  def predict_detail(self,data):
    model_preds, model_dict_preds=self.model_predicts(data)
    return self.ensemble_func(model_preds), model_dict_preds

  def predict_proba_detail(self,data):
    model_preds, model_dict_preds=self.model_predicts(data)
    return self.ensemble_func(model_preds), model_dict_preds

  def transform_dict_preds(self,preds):
    return np.array(list(preds.values())).T

  def transform_dict_preds_proba(self,preds):
    return np.hstack(np.array(list(preds.values())))

  def __len__(self):
    return len(self.model_list.keys())

  def print_eval(self,eval_dict):
    for model_name, eval_metric in eval_dict.items():
      print('{}: {}'.format(model_name,eval_metric))

  def evaluate(self,data, label, evaluation_fn, verbose=True):
    ensemble_pred, model_dict_preds = self.predict_detail(data)

    eval_dict={}

    for model_name, pred in model_dict_preds.items():
      eval_metric = evaluation_fn(pred,label)
      eval_dict[model_name]=eval_metric

    eval_metric = evaluation_fn(ensemble_pred,label)
    eval_dict['Ensemble']=eval_metric

    if verbose:
      self.print_eval(eval_dict)

    return {'ensmeble_pred':ensemble_pred,
            'model_predict':model_dict_preds,
            'eval_outcome':eval_dict}

  def cross_validation_evaluate(self,data, label, evaluation_fn, splits=5,repeats=1, verbose=True):
    '''
    evaluation_fn
    '''

    def convert_to_dataframe(eval_dict):

      record_df=pd.DataFrame()

      for model_name, eval_metrics in eval_dict.items():
        if not isinstance(eval_metrics, dict):
          eval_metrics = {'eval_metric': eval_metrics}
        row_df = pd.DataFrame(eval_metrics,index=[0])
        row_df['model'] = model_name
        record_df = pd.concat([record_df,row_df])
      return record_df

    record_df = pd.DataFrame()
    cv_models = []
    kfc = K_Fold_Creator(data, label, splits=splits, repeats=repeats)
    eval_columns=None

    for i,(cv_training_data, cv_training_label, cv_validation_data, cv_validation_label) in enumerate(kfc.get_split()):

      model = copy.deepcopy(self)

      model.fit(cv_training_data,cv_training_label)

      ensemble_pred, model_dict_preds, eval_dict = model.evaluate(cv_validation_data, cv_validation_label ,evaluation_fn,verbose=False).values()
      row_df = convert_to_dataframe(eval_dict)

      if eval_columns is None:
        eval_columns=row_df.columns[:-1].to_list()
      if verbose:
        print(f'\n\n====== CV:{i} ======')
        print(row_df.sort_values(by=eval_columns))

      row_df['fold'] = i
      record_df = pd.concat([record_df,row_df])
      cv_models.append(model)

    record_df = record_df.reset_index(drop=True)
    print(f'\n====== CV Mean ======')
    print(record_df.groupby(['model']).mean().drop(columns=['fold']).sort_values(eval_columns))

    return cv_models, record_df

class Stack_Ensemble_Model(Ensemble_Model):
  """
  overwrite: ensemble_func and fit
  """
  def __init__(self, model_list,stack_model = LinearRegression(),stack_training_split=0.2) -> None:
    self.stack_model = stack_model
    self.stack_training_split = stack_training_split
    super().__init__(model_list)

  def fit(self, data, label):
    splits = int(1/self.stack_training_split)
    model_data, model_label, stack_model_data, stack_model_label = K_Fold_Creator(data,label,splits=splits).get_data(fold=0)
    super().fit(model_data,model_label)
    model_preds, model_dict_preds=self.model_predicts(stack_model_data)
    self.stack_model.fit(model_preds,stack_model_label)


  def ensemble_func(self,model_preds):
    return self.stack_model.predict(model_preds)

class Mean_Ensemble_Model(Ensemble_Model):
  def __init__(self, model_list):
    super().__init__(model_list)

  def ensemble_func(self, model_preds):
    return np.mean(model_preds,axis=1)

class Vote_Ensemble_Model(Ensemble_Model):
  def __init__(self, model_list):
    super().__init__(model_list)

  def ensemble_func(self, model_preds):
    return [ Counter(pred).most_common(1)[0][0] for pred  in model_preds]

class Ensemble_Proba_Model(Ensemble_Model):

  def predict(self, data):
    return self.predict_proba(data)

  def evaluate(self,data, label, evaluation_fn, verbose=True):
    ensemble_pred, model_dict_preds = self.model_predicts(data)

    eval_dict={}

    for model_name, pred in model_dict_preds.items():
      eval_metric = evaluation_fn(pred,label)
      eval_dict[model_name]=eval_metric

    ensemble_pred = self.predict_proba(data)
    eval_metric = evaluation_fn(ensemble_pred,label)
    eval_dict['Ensemble']=eval_metric

    if verbose:
      self.print_eval(eval_dict)

    return {'ensmeble_pred':ensemble_pred,
            'model_predict':model_dict_preds,
            'eval_outcome':eval_dict}

class Stack_Ensemble_Proba_Model(Ensemble_Proba_Model):
  """
  overwrite: ensemble_func and fit
  """
  def __init__(self, model_list,stack_model = LinearRegression(),stack_training_split=0.2) -> None:
    self.stack_model = stack_model
    self.stack_training_split = stack_training_split
    super().__init__(model_list)

  def fit(self, data, label):
    splits = int(1/self.stack_training_split)
    model_data, model_label, stack_model_data, stack_model_label = K_Fold_Creator(data,label,splits=splits).get_data(fold=0)
    super().fit(model_data,model_label)
    model_preds, model_dict_preds=self.model_predicts_proba(stack_model_data)
    self.stack_model.fit(model_preds,stack_model_label)

  def ensemble_func(self,model_preds):
    return self.stack_model.predict(model_preds)

class Mean_Ensemble_Proba_Model(Ensemble_Proba_Model):
  def __init__(self, model_list):
    super().__init__(model_list)

  def ensemble_func(self, model_preds):
    model_preds = model_preds.reshape((model_preds.shape[0],self.num_models,-1))
    return np.mean(model_preds,axis=-2).argmax(axis=1)



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

  model_list = {'RF_Reg':RandomForestRegressor(n_estimators=310,max_depth=3),
              'XGB_Reg':XGBRegressor(n_estimators=31,max_depth=2),\
              'Bayesian':BayesianRidge(),
              'GP_Reg':gpr,
              'Huber_Reg':HuberRegressor(),
              'SVM_Reg':SVR(),
              'LR':LinearRegression(),
              'KR':KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
              'KNN_Reg':KNeighborsRegressor(),
              'LGB_Reg':model_lgb,
              'MLP_Reg':MLPRegressor(activation = "relu", alpha = 0.1, hidden_layer_sizes = (5,5),
                            learning_rate = "constant", max_iter = 3000, random_state = 1000),
              'SGD_Reg':reg,
              }
  stack_model_list= {
                'SVM_Linear_Reg':SVR(kernel='linear'),
                'Bayesian':BayesianRidge(),
                'MLP':MLPRegressor(activation = "relu", alpha = 0.1, hidden_layer_sizes = (5,),
                              learning_rate = "constant", max_iter = 1000, random_state = 1000),
                'Huber_Reg':HuberRegressor(),
                'LR':LinearRegression(),
                'XGB_Reg':XGBRegressor(n_estimators=11,max_depth=2),
                'RF_Reg':RandomForestRegressor(n_estimators=31,max_depth=2),
                }
  return model_list

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

  model_list = {'RF_Cls':RandomForestClassifier(n_estimators=310,max_depth=3),
              'XGB_Cls':XGBClassifier(n_estimators=31,max_depth=2),\
              'Bayesian':BayesianRidge(),
              'GP_Cls':gpr,
              'RC_Cls':RidgeClassifier(),
              'SVM_Cls':SVC(),
              'SVM_Cls':SVC(kernel='linear'),
              'SVM_Cls':SVC(kernel='rbf'),
              'KR':KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
              'KNN_Cls':KNeighborsClassifier(),
              'LGB_Cls':model_lgb,
              'MLP_Cls':MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (5,5),
                            learning_rate = "constant", max_iter = 3000, random_state = 1000),
              'SGD_Cls':reg,
              'QDA':QuadraticDiscriminantAnalysis(),
              }
  stack_model_list= {
                'SVM_Linear_Reg':SVC(kernel='linear'),
                'MLP':MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (5,),
                              learning_rate = "constant", max_iter = 1000, random_state = 1000),
                'XGB_Reg':XGBClassifier(n_estimators=11,max_depth=2),
                'RF_Reg':RandomForestClassifier(n_estimators=31,max_depth=2),
                'QDA':QuadraticDiscriminantAnalysis(),
                }
  return model_list

def get_reg_ensemble_model():
  model_list = regression_model()
  return Ensemble_Model(model_list=model_list,\
                        ensemble_fn=lambda model_preds: \
                          np.mean(np.array(model_preds),axis=1)\
                        )


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


''' DEMO
reg_model = get_reg_ensemble_model()
stack_model_list= {
              'SVM_Linear_Reg':SVR(kernel='linear'),
              'Bayesian':BayesianRidge(),
              'Huber_Reg':HuberRegressor(),
              'LR':LinearRegression(),
              }

evaluation_fn = calculate_metrics(['mse'])
model = Stack_Ensemble_Model(model_list,stack_model=Mean_Ensemble_Model(stack_model_list))
cv_models, cv_df = model.cross_validation_evaluate(data, label, evaluation_fn, fold=6, verbose=False)
cv_ensemble_model = Mean_Ensemble_Model(cv_models)

'''