import pandas as pd
from tqdm import  tqdm
import tqdm as tq
import copy
from ..utils import KFold_Sampler
import numpy as np
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

  def set_proba(self):
    self.proba_mode=True
    self.model_predict_fn = self._model_predicts_proba
    self.remove_no_prob_model()

  def remove_no_prob_model(self):
    model_dict_tmp = copy.deepcopy(self.model_dict)
    for model_name, model in self.model_dict.items():
      if 'predict_proba' not in model.__dir__():
        print(f"{model_name} don't have [predict_proba]")
        del model_dict_tmp[model_name]
    self.model_dict = model_dict_tmp

  def fit(self,data,label):
    with tqdm(range(len(self.model_dict.keys())), position=0,total=self.num_models, leave=False, bar_format='{desc:<30}\t{percentage:2.0f}%|{bar:10}{r_bar}') as pbar:
      for model_name, model in self.model_dict.items():
        pbar.set_description(f'{model_name} Training...')
        pbar.update()
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
