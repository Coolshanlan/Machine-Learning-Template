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
from utils import move_to
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
                 scheduler_iter=False,
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
        self.scheduler_iter=scheduler_iter
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
        trange = tqdm(dataloader,total=len(dataloader),desc=logger.name,bar_format='{desc:<5.5} {percentage:3.0f}%|{bar:20}{r_bar}')

        for _iter,(data,label) in enumerate(dataloader) :
            self.run_iter=_iter+1

            pred,(loss,loss_dict) = self.run_model(data,label,update=update)

            if self.scheduler and self.scheduler_iter and update:
                self.scheduler.step()

            recorder(pred=pred,**loss_dict)
            trange.set_postfix(**recorder.get_avg(loss_dict.keys()))
            trange.update()

        evaluate_dict = self.evaluation_fn(pred,label)
        avg_loss_dict=recorder.get_avg(loss_dict.keys())
        logger(**avg_loss_dict,**evaluate_dict)
        trange.set_postfix(**avg_loss_dict,**evaluate_dict)
        return recorder.get_dict(concat=['pred']),evaluate_dict

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


# def one_hot(x,num_class=num_class):
#         return torch.eye(num_class)[x,:]
# def evaluation_function(predict,label):
#     evaluation_dict={}
#     predict_category = predict.argmax(axis=1)

#     #ont_hot_label = one_hot(label)
#     #precision, recall, f1,_ = precision_recall_fscore_support(label,predict_category,average='macro')
#     #auroc = roc_auc_score(label,predict,average='macro')
#     acc= accuracy_score(label,predict_category)
#     evaluation_dict['acc'] =acc
#     # evaluation_dict['f1_score'] = f1
#     # evaluation_dict['recall'] = recall
#     # evaluation_dict['precision'] = precision
#     # evaluation_dict['auroc'] = auroc
#     return evaluation_dict

# def loss_function(pred,label):
#     one_hot_label=one_hot(label)
#     return nn.CrossEntropyLoss()(pred,label) + bi_tempered_logistic_loss(pred,one_hot_label.to('cuda'),t1=0.5,t2=2.0)


# model = torch.nn.Module
# loss_fn   = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,amsgrad=False)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max', factor=0.5, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08, verbose=True)
# model_instance= Model_Instance(model=model,
#                         optimizer=optimizer,
#                         loss_function=loss_fn,
#                         evaluation_function=evaluation_function,
#                         scheduler=scheduler,
#                         save_model_path=args.ckpt_dir/'best_model.pkl')