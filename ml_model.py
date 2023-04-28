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
import numpy as np

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
  """
  def __init__(self,model_list, ensemble_fn=None) -> None:

    if isinstance(model_list,list):
      self.model_list={}
      for midx, model in enumerate(model_list):
        self.model_list[f'model_{midx+1}']=model
    else:
      self.model_list = model_list

    if ensemble_fn:
      self.ensemble_func= ensemble_fn


  def ensemble_func(self,model_preds):
    return np.mean(np.array(model_preds),axis=1)

  def fit(self,feature,label):
    for model_name, model in self.model_list.items():
      model.fit(feature,label)


  def model_predicts(self,feature):
    model_dict_preds={}
    for model_name, model in self.model_list.items():
      pred = model.predict(feature)
      model_dict_preds[model_name]=pred
    model_preds = self.transform_dict_preds(model_dict_preds)
    return model_preds, model_dict_preds

  def predict(self,feature):
    model_preds, model_dict_preds=self.model_predicts(feature)
    return self.ensemble_func(model_preds), model_dict_preds

  def transform_dict_preds(self,preds):
    return np.array(list(preds.values())).T

  def __len__(self):
    return len(self.model_list.keys())

  def print_eval(self,eval_dict):
    for model_name, eval_metric in eval_dict.items():
      print('{}: {}'.format(model_name,eval_metric))

  def evaluation(self,feature, label, evaluation_fn, verbose=True):
    ensemble_pred, model_dict_preds = self.predict(feature)

    eval_dict={}

    for model_name, pred in model_dict_preds.items():
      eval_metric = evaluation_fn(pred,label)
      eval_dict[model_name]=eval_metric

    eval_metric = evaluation_fn(ensemble_pred,label)
    eval_dict['Ensemble']=eval_metric

    if verbose:
      self.print_eval(eval_dict)

    return ensemble_pred, model_dict_preds, eval_dict

class Stack_Ensemble_Model(Ensemble_Model):
  """
  overwrite: ensemble_func and fit
  """
  def __init__(self, model_list,stack_model = LinearRegression(),stack_training_split=0.2) -> None:
    self.stack_model = stack_model
    self.stack_training_split = stack_training_split
    super().__init__(model_list)

  def fit(self, feature, label):
    data_split_length=int(len(label)*self.stack_training_split)
    stack_model_feature, stack_model_label = feature[:data_split_length], label[:data_split_length]
    model_feature, model_label = feature[data_split_length:], label[data_split_length:]
    super().fit(model_feature,model_label)
    model_preds, model_dict_preds=self.model_predicts(stack_model_feature)
    self.stack_model.fit(model_preds,stack_model_label)

  def ensemble_func(self,model_preds):
    return self.stack_model.predict(model_preds)



def regression_model():
  kernel = DotProduct() + WhiteKernel()
  gpr = GaussianProcessRegressor(kernel=kernel)
  reg = make_pipeline(StandardScaler(),
                      SGDRegressor(max_iter=500, tol=5e-4))
  model_list = {'RF_Reg':RandomForestRegressor(n_estimators=300,max_depth=3),
              'XGB_Reg':XGBRegressor(n_estimators=31,max_depth=3),
              'SVM_Linear_Reg':SVR(kernel='linear'),
              'Bayesian':BayesianRidge(),
              'GP_Reg':gpr,
              'MLP':MLPRegressor(max_iter=250),
              'Huber_Reg':HuberRegressor(),
              'Quantile_Reg':QuantileRegressor(quantile=0.5),
              'TheilSe_Reg':TheilSenRegressor(),
              'LassoLars':LassoLars(alpha=.05),
              'SGD_Reg':reg,
              'T_Reg':TweedieRegressor(power=0),
              'SVM_Reg':SVR(),
              'PA_Reg':PassiveAggressiveRegressor(max_iter=10),
              'LR':LinearRegression(),
              }
  return model_list

def get_reg_ensemble_model():
  model_list = regression_model()
  return Ensemble_Model(model_list=model_list,\
                        ensemble_fn=lambda model_preds: \
                          np.mean(np.array(model_preds),axis=1)\
                        )
