import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import random
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, roc_auc_score, f1_score, recall_score, precision_score

class Recorder(dict):
    '''

    '''
    def __call__(self,**kwargs):
        for k,v in kwargs.items():
            if k in self.keys():
                self[k].append(v)
            else:
                self[k]=[]
                self[k].append(v)

    def get_dict(self,concat=[]):
        return_dict={}
        for k in self.keys():
            if  k in concat:
                try:
                    return_dict[k] = np.concatenate(self[k],axis=0)
                except:
                    #print('model output shape are not constant')
                    return_dict[k] = self[k]
            else:
                return_dict[k] = self[k]
        return return_dict

    def get_avg(self,keys):
        return_dict={}
        for k in keys:
            return_dict[k]=np.mean(self[k]).astype(float)
        return return_dict

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    #print('initialize network with %s' % init_type)
    net.apply(init_func)

def setSeed(seed=31,tor=True,tensorf=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    pd.core.common.random_state(seed)
    # tensorflow seed setting
    if tensorf:
        import tensorflow.compat.v2 as tf
        tf.random.set_seed(seed)
        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1
        )
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)

    # pytorch seed setting
    if tor:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        #torch.backends.cudnn.deterministic = True

def move_to(obj,**kwargs):
    '''
    move data to gpu.
    support types of dict, object and list
    '''
    if torch.is_tensor(obj):
        return obj.to(**kwargs)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, **kwargs)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, **kwargs))
        return res


def calculate_metrics(metrics_name ):#List
    '''
    return a function that contain multiple
    function input:
        function(pred,label)
    function return:
        dict{metric1:value1, metric2:value2 ...}
    '''
    if not isinstance(metrics_name,list):
        metrics_name=[metrics_name]

    def preprocess_pred_format(pred):
        if len(pred.shape)==1:
            return pred>=0.5
        else:
            return pred.argmax(axis=1)

    metrics_functions={}
    metrics_functions['acc']=lambda pred,label: accuracy_score(label,preprocess_pred_format(pred))
    metrics_functions['f1_score']=lambda pred,label: f1_score(label,preprocess_pred_format(pred), average='macro')
    metrics_functions['f1score']=lambda pred,label: f1_score(label,preprocess_pred_format(pred), average='macro')
    metrics_functions['recall']=lambda pred,label: recall_score(label,preprocess_pred_format(pred), average='macro')
    metrics_functions['precision']=lambda pred,label: precision_score(label,preprocess_pred_format(pred), average='macro')
    metrics_functions['auroc']=lambda pred,label: roc_auc_score(label,pred if len(pred.shape) == 1 else pred[:,1])
    if not set(metrics_name).issubset(set(metrics_functions.keys())):
        raise Exception(f'{set(metrics_name) - set(metrics_functions.keys())} metrics not support')
    return lambda pred,label: {name:metrics_functions[name](pred,label)for name in metrics_name}
