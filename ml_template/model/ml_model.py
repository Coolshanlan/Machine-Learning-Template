from .basic_import import *
from ..utils import KFold_Sampler
from .weight_ensemble import *
from .basic_model import *

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
      print('----- Model pre-training evaluation -----')
      _model.evaluate(stack_model_data,stack_model_label,evaluation_fn=mean_absolute_error)
      model_preds = self.stack_input_transform(model_preds)
      self.stack_model.fit(model_preds,stack_model_label)
      if isinstance(self.stack_model,ML_Weighted_Model):
        print('----- Stacking weights -----')
        max_length_name = max([len(model_name) for model_name in  list(_model.model_dict.keys())])
        for midx,model_name in enumerate(list(_model.model_dict.keys())):
          eval("print('{:<2} {:^"+str(max_length_name)+"} : {}'.format(str(midx),model_name,['%.3f'% v for v in np.array(self.stack_model.weights[midx,:])]))")

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


def regression_model():

  model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                learning_rate=0.05, n_estimators=720,
                                max_bin = 55, bagging_fraction = 0.8,
                                bagging_freq = 5, feature_fraction = 0.2319,
                                feature_fraction_seed=9, bagging_seed=9,
                                min_data_in_leaf =6, min_sum_hessian_in_leaf = 11,
                                is_unbalance=True,verbosity=-1)

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
              "Cat":CatBoostRegressor(
                          learning_rate=1e-2, 
                          loss_function='RMSE',verbose=0),
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
                                is_unbalance=True,verbosity=-1)

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
              "Cat_cls":CatBoostClassifier(),
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
