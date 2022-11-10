import torch
from torch.nn import init
from torch import autocast
from torch.cuda.amp import GradScaler
import torch.functional as F
import numpy as np
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, roc_auc_score, f1_score, recall_score, precision_score


def calculate_metrics(metrics_name):#List
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

def move_to(obj,**kwargs):
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

class Recorder(dict):
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
            return_dict[k]=np.mean(self[k])
        return return_dict

class Model_Instance():
    def __init__(self,
                 model,
                 optimizer=None,
                 scheduler=None,
                 scheduler_iter=False,
                 loss_function=None,
                 evaluation_function=lambda x,y : {},
                 clip_grad=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 save_dir ='checkpoint',
                 amp=False,
                 accum_iter=1,
                 model_weight_init=None):
        self.model = model.to(device)
        self.model_weight_init=model_weight_init
        self.save_dir = save_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_iter=scheduler_iter
        self.loss_criterial = loss_function
        self.clip_grad = clip_grad
        self.evaluation_fn=evaluation_function
        if isinstance(self.evaluation_fn,list):
            self.evaluation_fn=calculate_metrics(self.evaluation_fn)
        self.device = device
        self.amp=amp
        self.accum_iter=accum_iter
        self.run_iter=1
        self.grad_scaler=GradScaler(self.amp)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if self.model_weight_init:
            if self.model_weight_init not in ['normal','xavier','kaiming','orthogonal']:
                print('use normal weight init.')
                self.model_weight_init='normal'
            init_weights(self.model,init_type=self.model_weight_init)

    def loss_fn(self,pred,label):
        loss_return = self.loss_criterial(pred,label)
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

    def model_update(self,loss):
        # amp enable check
        if self.amp and self.device != torch.device('cpu'):
            self.grad_scaler.scale(loss/self.accum_iter).backward()

            # accumulate gradient
            if self.run_iter % self.accum_iter == 0:
                if self.clip_grad:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_grad)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            # accumulate gradient
            (loss/self.accum_iter).backward()
            if self.run_iter % self.accum_iter == 0:
                if self.clip_grad:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

    def _run(self,data):
        data = move_to(data,device=self.device,non_blocking=True,dtype=torch.float)
        return self.model(data)

    def _run_model(self,data,label):
        pred = self._run(data)

        label= move_to(label,device=self.device,non_blocking=True,dtype=torch.long)

        loss_return=self.loss_fn(pred,label)
        return pred, loss_return

    def run_model(self,data,label,update=True):
        self.model.train(update)
        amp_enable=self.amp and update

        # amp enable check
        with autocast(device_type='cuda' if self.device != 'cpu' else 'cpu', dtype=torch.float16,enabled=amp_enable):
            if update:
                    pred, (loss,loss_dict) = self._run_model(data,label)
                    self.model_update(loss)
            else:
                with torch.no_grad():
                    pred, (loss,loss_dict) = self._run_model(data,label)

        pred=pred.detach().to(torch.device('cpu'))
        loss=loss.detach().to(torch.device('cpu')).item()

        return  pred, (loss,loss_dict)


    def run_dataloader(self,dataloader,logger=None,update=True):
        recorder = Recorder()
        self.run_iter=0
        trange = tqdm(dataloader,total=len(dataloader),desc=logger.tag if logger else '',bar_format='{desc:<5.5} {percentage:3.0f}%|{bar:20}{r_bar}')

        for data,label in dataloader :
            self.run_iter+=1

            pred,(loss,loss_dict) = self.run_model(data,label,update=update)

            if self.scheduler and self.scheduler_iter and update:
                self.scheduler.step()

            recorder(pred=pred,label=label,**loss_dict)
            trange.set_postfix(**recorder.get_avg(loss_dict.keys()))
            trange.update()

        outcome=recorder.get_dict(concat=['pred','label'])
        evaluate_dict = self.evaluation_fn(outcome['pred'],outcome['label'])
        avg_loss_dict=recorder.get_avg(loss_dict.keys())
        record_dict={**evaluate_dict,**avg_loss_dict}
        if logger:
            logger(**record_dict)
        trange.set_postfix(**record_dict)
        return outcome,record_dict

    @torch.no_grad()
    def inference(self,data):
        self.model.train(False)
        return self._run(data).to(torch.device('cpu'))

    def inferance_dataloader(self,dataloader):
        reocord = Recorder()
        trange = tqdm(dataloader,total=len(dataloader))
        for data in dataloader :
            pred= self.inference(data)
            reocord(pred=pred)
            trange.update()
        return reocord.get_dict(concat=['pred'])

    def save(self,only_model=True,filename='model_checkpoint.pkl'):
        save_path = os.path.join(self.save_dir,filename)
        if only_model:
            torch.save(self.model.state_dict(),save_path)
        else:
            #save model instance
            pass
    #TODO
    def load(self,only_model=True,filename='model_checkpoint.pkl'):
        path = os.path.join(self.save_dir,filename)
        self.model.load_state_dict(torch.load(path))


class Ensemble_Instance():
    def __init__(self,
                 ensemble_model=LogisticRegression,
                 model_list=[],
                 ):
        self.ensemble_model = ensemble_model
        self.model_list = model_list

    def run_dataloader(self,dataloader):
        pass

