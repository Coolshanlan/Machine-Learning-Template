import torch
from torch.nn import init
from torch import autocast
from torch.cuda.amp import GradScaler
import torch.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import os
from utils import Recorder, init_weights, move_to, calculate_metrics


class Model_Instance():

#===================== Definition ============================
    def __init__(self,
                 model,
                 optimizer=None,
                 scheduler=None,
                 scheduler_epoch=False,
                 loss_function=None,
                 evaluation_metrics=lambda x,y : {},
                 clip_grad=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 amp=False,
                 accum_iter=1,
                 model_weight_init=None):

        self.model = model.to(device)
        self.model_weight_init=model_weight_init
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_epoch=scheduler_epoch
        self.loss_function = loss_function
        self.clip_grad = clip_grad
        self.evaluation_metrics=evaluation_metrics
        self.device = device
        self.amp=amp
        self.accum_iter=accum_iter
        self.update_counter=0
        self.grad_scaler=GradScaler(self.amp)

        if isinstance(self.evaluation_metrics,list):
            self.evaluation_metrics=calculate_metrics(self.evaluation_metrics)

        if self.model_weight_init:
            if self.model_weight_init not in ['normal','xavier','kaiming','orthogonal']:
                print('use normal weight init.')
                self.model_weight_init='normal'
            init_weights(self.model,init_type=self.model_weight_init)

    def get_loss(self,pred,label):
        loss_return = self.loss_function(pred,label)
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

#===================== Backward Model (Update)============================
    def model_update(self,loss):
        self.update_counter+=1
        # amp enable check
        if self.amp and self.device != torch.device('cpu'):
            self.grad_scaler.scale(loss/self.accum_iter).backward()

            # accumulate gradient
            if self.update_counter % self.accum_iter == 0:
                if self.clip_grad:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_grad)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            # accumulate gradient
            (loss/self.accum_iter).backward()
            if self.update_counter % self.accum_iter == 0:
                if self.clip_grad:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

#===================== Forward and Run Model (Run) ============================
    def forward(self,data):
        data = move_to(data,device=self.device,non_blocking=True,dtype=torch.float)
        return self.model(data)

    def run(self,data,label):
        pred = self.forward(data)
        label= move_to(label,device=self.device,non_blocking=True,dtype=torch.long)
        (loss,loss_dict)=self.get_loss(pred,label)
        return pred, (loss,loss_dict)

    def run_model(self,data,label,update=True):
        self.model.train(update or self.amp)
        # amp enable check
        if update:
            with autocast(device_type='cuda' if self.device != 'cpu' else 'cpu', enabled=self.amp):
                pred, (loss,loss_dict) = self.run(data,label)
                self.model_update(loss)
        else:
            with torch.no_grad():
                pred, (loss,loss_dict) = self.run(data,label)

        pred=pred.detach().to(torch.device('cpu'))
        loss=loss.detach().to(torch.device('cpu')).item()

        return  pred, (loss,loss_dict)

#===================== Train and Run Dataset ============================
    def print_record_dict(self,record_dict,tag):
        print(f'\n---------\n【{tag}】:')
        for key,value in record_dict.items():
            print(f'{key}\t:{np.round(value,3)}')
        print('---------')

    def get_recorder_result_dict(self,recorder):
        outcome=recorder.get_dict(concat=['pred','label'])
        loss_dict = list(recorder.keys())
        loss_dict.remove('pred')
        loss_dict.remove('label')
        evaluate_dict = self.evaluation_metrics(outcome['pred'],outcome['label'])
        avg_loss_dict=recorder.get_avg(loss_dict)
        record_dict={**evaluate_dict,**avg_loss_dict}
        for k,v in record_dict.items():
            record_dict[k] = np.round(v,3)
        return outcome,record_dict

    def run_dataloader(self,
                       dataloader,
                       logger=None,
                       update=True,
                       display_progress=False,
                       display_result=True
                       ):
        recorder = Recorder()
        current_iter=0
        self.update_counter=0
        logger_tag =logger.tag if logger else 'Run Dataloader'
        trange = tqdm(dataloader,total=len(dataloader),desc=logger_tag,bar_format='{desc} {percentage:3.0f}%|{bar:20}{r_bar}')

        for data,label in trange :

            pred,(loss,loss_dict) = self.run_model(data,label,update=update)

            if not self.scheduler_epoch and update and self.scheduler :
                self.scheduler.step()

            recorder(pred=pred,label=label,**loss_dict)

            if display_progress:
                trange.set_postfix(**recorder.get_avg(loss_dict.keys()))

            current_iter+=1

        if self.scheduler_epoch and update and self.scheduler :
                self.scheduler.step()

        outcome,record_dict = self.get_recorder_result_dict(recorder)

        record_dict['step']=current_iter

        if display_result:
            self.print_record_dict(record_dict,logger_tag)

        if logger:
            logger[logger_tag](**record_dict)

        return outcome,record_dict

    def run_step_dataloader(self,
                            dataloader,
                            run_step,
                            valid_step=None,
                            evaluation_function=None,
                            logger=None,
                            update=True,
                            display_progress=False,
                            display_result=True):

        current_iter=0
        self.update_counter=0
        logger_tag = logger.tag if logger else 'Train'

        while(current_iter < run_step):
            trange = tqdm(dataloader,total=len(dataloader),desc=logger_tag,bar_format='{desc} {percentage:3.0f}%|{bar:20}{r_bar}')
            recorder = Recorder()
            for data,label in trange :

                pred,(loss,loss_dict) = self.run_model(data,label,update=update)

                if not self.scheduler_epoch and update and self.scheduler :
                    self.scheduler.step()

                recorder(pred=pred,label=label,**loss_dict)

                if display_progress:
                    trange.set_postfix(**recorder.get_avg(loss_dict.keys()))

                current_iter+=1

                if (evaluation_function is not None) and (valid_step is not None) and  (current_iter % valid_step == 0):
                    print(f'\n================= Step: {current_iter} =================')
                    if logger is not None or display_result:
                        outcome,record_dict = self.run_dataloader(dataloader,update=False,display_progress=False,display_result=False)

                        if display_result:
                            self.print_record_dict(record_dict,logger_tag)
                        if logger:
                            record_dict['step']=current_iter
                            logger[logger_tag](**record_dict)

                    evaluation_function()


                if run_step == current_iter:
                    break


            if self.scheduler_epoch and update and self.scheduler :
                self.scheduler.step()


            outcome,record_dict = self.get_recorder_result_dict(recorder)
            print(record_dict)


        if run_step % valid_step != 0:
            outcome,record_dict = self.run_dataloader(dataloader,update=False,display_progress=False,display_result=True)
            record_dict['step']=current_iter

            if logger:
                logger[logger_tag](**record_dict)

            evaluation_function()

        return outcome,record_dict

    def run_epoch_dataloader(self,
                             dataloader,
                             run_epoch,
                             valid_epoch=None,
                             evaluation_function=None,
                             logger=None,
                             update=True,
                             display_progress=False,
                             display_result=True):
        current_epoch=0
        self.update_counter=0
        logger_tag = logger.tag if logger else 'Train'

        while(current_epoch < run_epoch):
            recorder = Recorder()
            current_epoch+=1

            trange = tqdm(dataloader,total=len(dataloader),desc=logger_tag,bar_format='{desc} {percentage:3.0f}%|{bar:20}{r_bar}')

            for data,label in dataloader :

                pred,(loss,loss_dict) = self.run_model(data,label,update=update)

                if not self.scheduler_epoch and update and self.scheduler :
                    self.scheduler.step()

                recorder(pred=pred,label=label,**loss_dict)
                if display_progress:
                    trange.set_postfix(**recorder.get_avg(loss_dict.keys()))

                if (evaluation_function is not None) and (valid_epoch is not None) and  (current_epoch % valid_epoch == 0):
                    print(f'================= Epoch: {current_epoch} =================')
                    if logger is not None or display_result:
                        outcome,record_dict = self.run_dataloader(dataloader,update=False,display_progress=False,display_result=False)

                        if display_result:
                            self.print_record_dict(record_dict,logger_tag)
                        if logger:
                            record_dict['epoch']=current_epoch
                            logger[logger_tag](**record_dict)


            if self.scheduler_epoch and update and self.scheduler :
                self.scheduler.step()

            outcome,record_dict = self.get_recorder_result_dict(recorder)
            print(record_dict)


        if run_epoch % valid_epoch  != 0:
            outcome,record_dict = self.run_dataloader(dataloader,update=False,display_progress=False,display_result=True)
            record_dict['epoch']=run_epoch

            if logger:
                logger[logger_tag](**record_dict)

            evaluation_function()

        return outcome,record_dict


#===================== Inference ============================
    @torch.no_grad()
    def inference(self,data):
        self.model.train(False)
        return self.forward(data).to(torch.device('cpu'))

    def inference_dataloader(self,dataloader):
        reocord = Recorder()
        trange = tqdm(dataloader,total=len(dataloader))
        for data in trange :
            pred= self.inference(data)
            reocord(pred=pred)

        return reocord.get_dict(concat=['pred'])

#===================== Save and Load ============================
    def save_model(self,path=None):
        if path is None:
            if not os.path.exists('checkpoint'):
                os.mkdir(self.save_dir)
            path = 'checkpoint/model_weighted.pkl'
        torch.save(self.model.state_dict(),path)

    #TODO
    def save_instance(self,filename='model_instance_checkpoint.pkl'):
        pass

    def load_model(self,path=None):
        if path is None:
            path = os.path.join(self.save_dir,)
        self.model.load_state_dict(torch.load(path))

    #TODO
    def load_instance(self,filename='model_instance_checkpoint.pkl'):
        pass


class Ensemble_Instance():
    def __init__(self,
                 ensemble_model=LogisticRegression,
                 model_list=[],
                 ):
        self.ensemble_model = ensemble_model
        self.model_list = model_list

    def run_dataloader(self,dataloader):
        pass

