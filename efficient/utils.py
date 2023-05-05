import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import random
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import RepeatedStratifiedKFold

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
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

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

def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


class KFold_Sampler():
    def __init__(self,data,label,n_splits,n_repeats=1) -> None:
        self.fold_index = []
        self.data = np.array(data)
        self.label = np.array(label)
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
        self._get_fold_index()

    def _get_fold_index(self):
        for (train_index, test_index) in self.skf.split(self.data,self.label):
            self.fold_index.append((train_index, test_index))


    def get_data(self,fold=0):
        (train_index, test_index) = self.fold_index[fold]
        train_data, train_label = self.data[train_index], self.label[train_index]
        test_data, test_label = self.data[test_index], self.label[test_index]
        return train_data, train_label, test_data, test_label

    def get_multi_fold_data(self,n_fold=1):
        train_index_list = np.concatenate([ self.fold_index[i][1] for i in range(n_fold)])
        test_index_list = list(set(list(range(len(self.label))))-set(train_index_list))
        train_data, train_label = self.data[train_index_list], self.label[train_index_list]
        test_data, test_label = self.data[test_index_list], self.label[test_index_list]
        return train_data, train_label, test_data, test_label



    def splits(self):
        for i_fold in range(self.num_split):
            yield self.get_data(i_fold)

    @property
    def total_split(self):
        return self.n_splits * self.repeats
