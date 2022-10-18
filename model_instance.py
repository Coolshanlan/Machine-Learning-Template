import torch.nn as nn
from  torch.utils.data import Dataset,DataLoader
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
import torch.functional as F
import numpy as np
from tqdm.auto import tqdm
import os
import pandas as pd

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
                return_dict[k] = np.concatenate(self[k],axis=0)
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
                 scheduler_iter_unit=False,
                 loss_function=None,
                 evaluation_function=lambda x,y : {},
                 clip_grad=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 save_dir ='checkpoint',
                 amp=False,
                 accum_iter=1):
        self.model = model.to(device)
        self.save_dir = save_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_iter_unit=scheduler_iter_unit
        self.loss_criterial = loss_function
        self.clip_grad = clip_grad
        self.evaluation_fn=evaluation_function
        self.device = device
        self.amp=amp
        self.accum_iter=accum_iter
        self.run_iter=1
        self.grad_scaler=GradScaler(self.amp)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    def loss_fn(self,pred,label):
        loss_return = self.loss_criterial(pred,label)
        if not isinstance(loss_return,tuple):
            loss_return = (loss_return,{'loss':loss_return.detach().to(torch.device('cpu'))})
        else:
            loss,loss_dict=loss_return
            if 'loss' not in loss_dict.keys():
                loss_dict['loss'] = loss
            loss_dict = {k:loss_dict[k].detach().to(torch.device('cpu')) for k in loss_dict.keys()}
            loss_return=(loss,loss_dict)
        return loss_return

    def model_update(self,loss):
        if type(loss)==tuple:
            loss = loss[0]
        if self.amp and self.device != torch.device('cpu'):
            self.grad_scaler.scale(loss/self.accum_iter).backward()
            if self.run_iter % self.accum_iter == 0:
                if self.clip_grad:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_grad)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
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
        with autocast(device_type='cuda' if self.device != 'cpu' else 'cpu', dtype=torch.float16,enabled=amp_enable):
            if update:
                    pred, (loss,loss_dict) = self._run_model(data,label)
                    self.model_update(loss)
            else:
                with torch.no_grad():
                    pred, (loss,loss_dict) = self._run_model(data,label)

        pred=pred.detach().to(torch.device('cpu'))
        loss=loss.detach().to(torch.device('cpu')).item()
        label=label.detach().to(torch.device('cpu'))

        return  pred, (loss,loss_dict)


    def run_dataloader(self,dataloader,logger=None,update=True):
        recorder = Recorder()
        self.run_iter=0
        trange = tqdm(dataloader,total=len(dataloader),desc=logger.tag if logger else '',bar_format='{desc:<5.5} {percentage:3.0f}%|{bar:20}{r_bar}')

        for _iter,(data,label) in enumerate(dataloader) :
            self.run_iter=_iter+1

            pred,(loss,loss_dict) = self.run_model(data,label,update=update)

            if self.scheduler and self.scheduler_iter_unit and update:
                self.scheduler.step()

            recorder(pred=pred,**loss_dict)
            trange.set_postfix(**recorder.get_avg(loss_dict.keys()))
            trange.update()

        evaluate_dict = self.evaluation_fn(pred,label)
        avg_loss_dict=recorder.get_avg(loss_dict.keys())
        record_dict={**evaluate_dict,**avg_loss_dict}
        if logger:
            logger(**record_dict)
        trange.set_postfix(**record_dict)
        return recorder.get_dict(concat=['pred']),record_dict

    @torch.no_grad()
    def inference(self,data):
        self.model.train(False)
        return self._run(data).to(torch.device('cpu'))

    def inference_dataloader(self,dataloader):
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

    def load_model(self,path=None):
        path = path if path else os.path.join(self.save_dir,'model_checkpoint.pkl')
        self.model.load_state_dict(torch.load(path))

    def load_model_instance(self,path=None):
        pass