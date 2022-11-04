# Efficient Pytorch Template
 This is a efficient pytorch template support logger, dataset and all utils that you need during training and inference.

# Example Code
``` python
from logger import Logger
from model_instance import Model_Instance
from dataset import NormalDataset
from sklearn.metrics import accuracy_score

# init logger and define config
Logger.init('test1')
config = Logger.config
config.batch_size=32
config.lr=1e-3

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

# define training logger to record training log
train_logger = Logger('Train')
# define validation logger to record evaluation log
valid_logger = Logger('Valid')

# Start training
for epoch in range(cfg.epoch):
  record,evaluation = model_instance.run_dataloader(train_dataloader,logger=train_logger,update=True)
  record,evaluation = model_instance.run_dataloader(valid_dataloader,logger=valid_logger,update=False)

  # save best model
  if valid_logger.check_best('loss',mode='min'):
    model_instance.save()

# Inference - Case1 dataloader
test_dataloader=get_dataloader()
preds = model_instance.inference_dataloader(test_dataloader)

# Inference -Case2 Only Data
preds = model.instance(data)

# Visualize training history
Logger.plot()
Logger.export()
```
![](https://github.com/Coolshanlan/Efficient-Pytorch-Template/blob/main/image/logger_example1.png)


# Requests
## Dataset Output Format
Dataset output must be **(data,label)**

Data and label can be **dictionary** if you need multi-input or multiple label
> you can get your dictionary data in model input

```python
class DemoDataset(Dataset):
    def __getitem__(self,idx):
        return data[idx],labels[idx]
```

## Model Output
Your model output shape must be constant in every single data
> Slot tagging output shape are not constant.

You can overwrite `Model_Instance.run_dataloader` function or use `Model_Instance.run_model` get each batch output and run dataloader by your self.

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
                 accum_iter=1,
                 model_weight_init=None):


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
- **model_weight_init** options -> ['normal','xavier','kaiming','orthogonal']

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
  return {'accuracy':acc,'f1_score':f1}
# if you don't have evaluation metrics,
# you can ignore evaluation_fn parameter in Model Instance
```
you can use `Logger.plot()` to see metrics record after running `run_dataloader`

It also will display in terminal after each epoch.
```console
eval  100%|████████████████████| 24/24 [00:00<00:00, 80.28it/s, acc=0.214,f1_score=0.513, A_loss_Name=3.83, B_loss_Name=3.69, loss=7.52]
```
## Logger
### Example Cpde
```python
# initialize Logger with experiment name
Logger.init(experiment_name)

# define config (option)
config = Logger.config
config.batch_size=32
config.lr=1e-3
#etc.

# define training logger to record training log
train_logger = Logger('Train')
# define validation logger to record evaluation log
valid_logger = Logger('Valid')

for e in range(epoch):
  model_instance.run_dataloader(dataloader=train_dataloader,
                                logger=train_logger)
  model_instance.run_dataloader(dataloader=valid_dataloader,
                                logger=valid_logger)
  # check best epoch
  if valid_logger.check_best(category='loss',mode='min'):
    model_instance.save(filename='best_model.pkl')

Logger.plot()
Logger.export()
```
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
