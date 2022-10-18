# Efficient Pytorch Template
 This is a efficient pytorch template support logger, dataset and all utils that you need during training and inference.

# Example Code
``` python
from logger import Logger
from model_instance import Model_Instance
from dataset import NormalDataset
from sklearn.metrics import accuracy_score

# init logger and define config
Logger.init('Version1')
config = Logger.config

# Define Dataset
training_dataloader,valid_dataloader = get_dataloader()

# Define loss and evaluation function
def loss_fn(pred,label):
  return nn.CrossEntropyLoss()(pred,label)

def evaluation_fn(pred,label):
  return {'acc':accuracy_score(label,pred)}

# Create model instance
model = get_model()
optimizer = torch.optim.AdamW(model.parameters(),lr=config.lr)
model_instance = Model_Instance(model=model,
                                optimizer=optimizer,
                                loss_fn=loss_fn,evaluation_fn=evaluation_fn)

# Create Logger

training_logger = Logger('train')
valid_logger = Logger('valid')

# Start training
for epoch in range(cfg.epoch):
  record,evaluation = model_instance.run_dataloader(train_dataloader,logger=training_logger,update=True)
  record,evaluation = model_instance.run_dataloader(valid_dataloader,logger=valid_logger,update=False)
  if valid_logger.check_best('loss',mode='min):
    model_instance.save(only_model=True,filename='best_model.pkl')

# Inference - Case1 dataloader
test_dataloader=get_dataloader()
preds = model_instance.inference_dataloader(test_dataloader)

# Inference -Case2 Only Data
preds = model.instance(data)



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
                 accum_iter=1):

def run_model(self,data,label,update=True):
  return model_outputs, (loss,loss_dict)

def run_dataloader(self,dataloader,logger=None,update=True):
  return record_dict, evaluation_dict

@torch.no_grad()
def inference(self,data):
  return model_outputs

def inference_dataloader(self,dataloader):
  return model_outputs

def save(self,only_model=True,filename='model_checkpoint.pkl'):

def load_model(self,only_model=True,path=None):
```
- **scheduler_iter** `scheduler_iter=True`, if your learning scheduler update during per iter then
- **clip_grad** Clips gradient of an iterable of parameters at specified value.
- **device** torch.device()
- **save_dir** the dir to store model checkpoint
- **amp** `amp=True`, Enable Automatic Mixed Precision
- **accum_iter** `accum_iter=N` if (N>1), Enable Gradient Accumulation else N=1

### Loss Function Define
  ```python
  #case 1
  def loss_fn(pred,label):
    return your_loss
  #case 2
  def loss_fn(pred,label):
    loss_A=...
    loss_B=...
    total_loss = loss_A+loss_B
    return total_loss,{'A_loss_Name':loss_A,'B_loss_Name':loss_B}
  ```
  Each different loss details will display in progress like
  ```console
  train  54%|██████████▊         | 63/117 [00:01<00:01, 46.42it/s, A_loss_Name=5, B_loss_Name=4.2, loss=9.2]
  ```

## Evaluation function Define
```python
def evaluation_fn(pred,label):
  acc = get_acc(pred,label)
  f1 = get_f1(pred,label)
  return {'accuracy':acc,'f1 score':f1}
# if you don't have evaluation metrics,
# you can ignore evaluation_fn parameter in Model Instance
```
you can use `Logger.plot()` to see metrics record after running `run_dataloader`

It also will display in terminal after each epoch.
```console
eval  100%|████████████████████| 24/24 [00:00<00:00, 80.28it/s, acc=0.214, A_loss_Name=3.83, B_loss_Name=3.69, loss=7.52]
```
## Logger
### Plot
```python
@staticmethod
def plot(show_logger=None,
         show_category=None,
         figsize=(7.6*1.5,5*1.5),
         cmp=mpl.cm.Set2.colors,
         ylim={},
         filename='logger_history.png',
         save=True,
         show=True):
```
- **show_logger** `show_logger=[logger1_name,logger2_name...]` witch **logger** you want to show
- **show_category** `show_logger=['acc','f1' ...]` witch **evaluation metrics** you want to show, it is depend on tou `evaluation_function`
- **save** save figure or not
- **show** plt.show() or not
