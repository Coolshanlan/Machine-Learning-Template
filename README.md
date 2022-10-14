# Efficient-Pytorch-Template
 This is a efficient pytorch template support logger, dataset and all utils that you need during training and inference.

# Example Code
``` python
from logger import Logger
from model_instance import Model_Instance
from dataset import NormalDataset
from sklearn.metrics import accuracy_score

# Define Dataset
training_dataloader,valid_dataloader = get_dataloader()

# Define loss and evaluation function
def loss_fn(pred,label):
  return nn.CrossEntropyLoss()(pred,label)

def evaluation_fn(pred,label):
  return {'acc':accuracy_score(label,pred)}

# Create model instance
model = get_model()
optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.lr)
model_instance = Model_Instance(model=model,optimizer=optimizer,loss_fn=loss_fn,evaluation_fn=evaluation_fn)

# Create Logger
training_logger = Logger('train')
valid_logger = Logger('valid')

# Start training
for epoch in range(cfg.epoch):
  record,evaluation = model_instance.run_dataloader(train_dataloader,logger=training_logger,update=True)
  record,evaluation = model_instance.run_dataloader(valid_dataloader,logger=valid_logger,update=False)
  if valid_logger.check_best('loss',mode='min):
    model_instance.save(only_model=True,filename='best_model.pkl')

# Visualize training history
Logger.plot()
```
![](https://github.com/Coolshanlan/Efficient-Pytorch-Template/blob/main/image/logger_example1.png)

# Tutorial
## Model Instance
### OverView
```python
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
                 accum_iter=1)

def run_model(self,data,label,update=True)
  return model_outputs, (loss,loss_dict)

def run_dataloader(self,dataloader,logger=None,update=True)
  return record_dict, evaluation_dict

@torch.no_grad()
def inference(self,data)
  return model_outputs

def inference_dataloader(self,dataloader):
  return model_outputs

def save(self,only_model=True,filename='model_checkpoint.pkl')

def load_model(self,only_model=True,path=None)
```
- **scheduler_iter**
  if your learning scheduler is be updated during per iter then `scheduler_iter=True`
### run_model
```python
def run_model(self,data,label,update=True)
return pred, (loss,loss_dict)
```
- loss is your total loss that define in loss_fn