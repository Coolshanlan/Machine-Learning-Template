from typing_extensions import Self
import torch.nn as nn
from  torch.utils.data import Dataset,DataLoader
import torch
import numpy as npf
import torch.functional as F
from typing import Callable, Dict, Tuple
import numpy as np
class Model_Instance():
    def __init__(self,
                 model:torch.nn.Module,
                 optimizer:torch.optim,
                 scheduler:torch.optim.lr_scheduler,
                 loss_function:Callable,
                 evaluation_function:Callable=lambda x,y : {},
                 clip_grad=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 save_model_path:str ='best_model.pkl'):
        self.model = model.to(device)
        self.save_model_path = save_model_path
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_function
        self.clip_grad = clip_grad
        self.evaluation_fn=evaluation_function
        self.device = device

    def run_model(self,feature,label,update=True)-> Tuple[torch.tensor,torch.tensor,Dict]:

        feature = feature.to(self.device)
        label = label.to(self.device)

        if update:
            pred = self.model(feature)
        else:
            with torch.no_grad():
                pred = self.model(feature)

        loss = self.loss_fn(pred,label)

        if update:
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()

        pred=pred.cpu().detach()
        loss=loss.cpu().detach().item()
        label=label.cpu().detach()

        evaluate_dict = self.evaluate_fn(pred,label)

        return  pred, loss, evaluate_dict

    def run_one_epoch(self,dataloader,logger=None,update=True):
        pred_list=[]
        loss_list=[]
        if update:
            self.model.train()
        else:
            self.model.eval()

        trange = tqdm(dataloader,total=len(dataloader))
        for data,label in dataloader :
            data=data.long()# for gpu
            label=label.long()# for gpu
            pred,loss, eval_dict = self.run_model(data,label,update=update)
            pred_list.append(pred)
            loss_list.append(loss)

            logger(loss=loss,**eval_dict)
            avg_log = logger.get_current_epoch_avg()
            trange.set_postfix(loss=avg_log['loss'],**eval_dict)
            trange.update()
        logger.save_epoch()
        pred_list = np.concatenate(pred_list,axis=0)
        return pred_list,loss_list

    def evaluation(self,dataloader):
        pred_list=[]
        self.model.eval()

        trange = tqdm(dataloader,total=len(dataloader))
        for data in dataloader :
            data=data.long()
            pred= self.inference(data)
            pred_list.append(pred)
            trange.update()

        pred_list = np.concatenate(pred_list,axis=0)
        return pred_list

    def inference(self,data):
        data = data.to(self.device)
        pred = self.model(data)
        return pred

    def save(self,path=None,only_model=True):
        if only_model:
            path = path if path else self.save_model_path
            torch.save(self.model.state_dict(),path)
        else:
            pass

    def load_model(self,path=None):
        path = path if path else self.save_model_path
        self.model.load_state_dict(torch.load(path))


# def create_model_instance(args):
#     global num_class
#     def evaluation_function(predict,label):
#         def one_hot(x,num_class=num_class):
#             return torch.eye(num_class)[x,:]

#         evaluation_dict={}
#         predict_category = predict.argmax(axis=1)

#         #ont_hot_label = one_hot(label)
#         #precision, recall, f1,_ = precision_recall_fscore_support(label,predict_category,average='macro')
#         #auroc = roc_auc_score(label,predict,average='macro')
#         acc= accuracy_score(label,predict_category)
#         evaluation_dict['acc'] =acc
#         # evaluation_dict['f1_score'] = f1
#         # evaluation_dict['recall'] = recall
#         # evaluation_dict['precision'] = precision
#         # evaluation_dict['auroc'] = auroc
#         return evaluation_dict
#     def loss_function(self,pred,label):
#         global num_class
#         def one_hot(x,num_class=num_class):
#             return torch.eye(num_class)[x,:]

#     one_hot_label=one_hot(label)
#     return nn.CrossEntropyLoss()(pred,label) + bi_tempered_logistic_loss(pred,one_hot_label.to('cuda'),t1=0.5,t2=2.0)


#     model = torch.nn.Module
#     loss_fn   = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,amsgrad=False)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max', factor=0.5, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08, verbose=True)
#     return Model_Instance(model=model,
#                          optimizer=optimizer,
#                          loss_function=loss_fn,
#                          evaluation_function=evaluation_function,
#                          scheduler=scheduler,
#                          save_model_path=args.ckpt_dir/'best_model.pkl')