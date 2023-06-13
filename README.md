# Machine Learning Template
 This is a efficient machine learning template support ml model ensemble, logger, dataset and all utils that you need during training and inference.

- <a href="#example_code">Example Code</a>
- <a href="#features">Features</a>
- <a href="#request_format">Request Format</a>
  - <a href="#dataset_output_format">Dataset Format</a>
  - <a href="#inference_dataset_format">Inference Dataset Format</a>
  - <a href="#model_output_format">Model Output Format</a>
- <a href="#tutorial">Tutorial</a>
  - <a href="#model_instance">Model Instance</a>
    - <a href="#overview">Overview</a>
    - <a href="#custom_loss_function">Custum Loss Function</a>
    - <a href="#custom_evaluation_function">Custum Evaluation Function</a>
  - <a href="#logger">Logger</a>
    - <a href="#logger_example_code">Example Code</a>
    - <a href="#record_experiments">Record Experiment</a>
    - <a href="#logger_plot">Plot</a>
    - <a href="#logger_plot">Plot Experiment</a>
 
<div id="example_code"></div>

# Example Code
``` python
from logger import Logger
from model_instance import Model_Instance

# init logger and record config
Logger.init('Experiment1')
config = Logger.config
config.batch_size=32
config.lr=1e-3

# Define Dataset
training_dataloader,valid_dataloader = get_dataloader()

# Create model instance
model = get_model()
model_instance = Model_Instance(model=model,
                                optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr),
                                loss_function=nn.CrossEntropyLoss(),
                                evaluation_function=['acc','f1score'])

# define training/validation logger to record
train_logger = Logger('Train')
valid_logger = Logger('Valid')

# Start training
for epoch in range(cfg.epoch):
  outcome,record = model_instance.run_dataloader(train_dataloader,logger=train_logger,update=True)
  outcome,record = model_instance.run_dataloader(valid_dataloader,logger=valid_logger,update=False)

  # save best model
  if valid_logger.check_best('acc',mode='max'):
    model_instance.save() # default path ./checkpoint/model_checkpoint.pkl

# Load best model
model_instance.load() # default path ./checkpoint/model_checkpoint.pkl

# Inference - Case1 dataloader
test_dataloader=get_dataloader()
preds = model_instance.inference_dataloader(test_dataloader)

# Visualize training history
Logger.plot()
Logger.export()
```
![](https://github.com/Coolshanlan/Efficient-Pytorch-Template/blob/main/image/logger_example1.png)

<div id="features"></div>

# Features
- Model Instance
  - run_model
  - run_dataloader
  - inference
  - inference_dataloader
- Logger
  - plot
  - plot_experiments
- Ensemble Model Instance

<div id="request_format"></div>

# Request Format

<div id="dataset_output_format"></div>

## Dataset Output Format
Dataset output must be **(data,label)**

Data and label can be **dictionary** if you need multi-input or multiple label
> you can get your dictionary data in model input

```python
class DemoDataset(Dataset):
    def __getitem__(self,idx):
        return data[idx],labels[idx]
```

<div id="inference_dataset_format"></div>

## Inference Dataset Format
inference data can use `model_instance.inference_dataloader(dataloader)` with dataloader or just use `model_instance.inference(data)` with raw input data
```python
class InferenceDemoDataset(Dataset):
    def __getitem__(self,idx):
        return data[idx]
```

<div id="model_output_format"></div>

## Model Output Format
Your model output shape must be constant in every single data
> Slot tagging output shape are not constant.

You can overwrite `Model_Instance.run_dataloader` function or use `Model_Instance.run_model` get each batch output and run dataloader by your self.

<div id="tutorial"></div>

# Tutorial

<div id="model_instance"></div>

## Model Instance

<div id="overview"></div>

### Overview
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
  return outcome, record_dict
  #outcome['pred']/outcome['label'] , record_dict['loss ... ']/record_dict['metrics ... ']

@torch.no_grad()
def inference(self,data):
  return model_outputs

def inference_dataloader(self,dataloader):
  return model_outputs

def save(self,only_model=True,filename='model_checkpoint.pkl'):
  save_path = os.path.join(self.save_dir,filename)

def load(self,only_model=True,filename='model_checkpoint.pkl'):
  load_path = os.path.join(self.save_dir,filename)
```
- **scheduler_iter** `scheduler_iter=True`, if your learning scheduler update during per iter then
- **clip_grad** Clips gradient of an iterable of parameters at specified value.
- **device** torch.device()
- **save_dir** the dir to store model checkpoint
- **amp** `amp=True`, Enable Automatic Mixed Precision
- **accum_iter** `accum_iter=N` if (N>1), Enable Gradient Accumulation else N=1
- **model_weight_init** options -> ['normal','xavier','kaiming','orthogonal']

<div id="custom_loss_function"></div>

### Custom Loss Function
  ```python
  #case 1 simple loss without name
  def loss_fn(pred,label):
    return your_loss
  #case 2 return multiple loss with different names to record
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

<div id="custom_evaluation_function"></div>

### Custom Evaluation Function
```python
# case1 custom define, return a dictionary
def evaluation_fn(pred,label):
  acc = get_acc(pred,label)
  f1 = get_f1(pred,label)
  return {'accuracy':acc,'f1_score':f1}

# case2 give a list of metrics names, support 'acc','f1score','precision','recall','auroc'
model_instance = Model_Instance(evaluation_fn = ['acc','f1score','precision','recall','auroc'])

# case3 ignore
# if you don't have evaluation metrics,
# you can ignore evaluation_fn parameter in Model Instance
```
you can use `Logger.plot()` to see metrics record after running `run_dataloader`

It also will display in terminal after each epoch.
```console
eval  100%|████████████████████| 24/24 [00:00<00:00, 80.28it/s, acc=0.214,f1_score=0.513, A_loss_Name=3.83, B_loss_Name=3.69, loss=7.52]
```

<div id="logger"></div>

## Logger

<div id="logger_example_code"></div>

### Example Code
```python
# initialize Logger with experiment name
Logger.init(experiment_name)

# define config (option)
config = Logger.config
config.batch_size=32
config.lr=1e-3
#etc.

# define training logger to record training log
train_logger = Logger('<Tag1>')#Ex. Training
# define validation logger to record evaluation log
valid_logger = Logger('<Tag2>')#Ex. Validation

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
<div id="record_experiments"></div>

### Record Experiments
you can define the experiment name when initialize Logger
```python
Logger.init('<experiment name>')
```

And plot your all experiment log with
```python
Logger.plot_experiments()
```

<div id="logger_plot"></div>

### Plot
```python
@staticmethod
def plot(experiment_name, #default Logger.experiment
         show_tag=None,
         show_category=None,
         figsize=(7.6*1.5,5*1.5),
         cmp=mpl.cm.Set2.colors,
         ylim={},
         filename='logger_history.png',
         save=True,
         show=True):
```
- **show_tag** `show_tag=[logger1_tag,logger2_tag...]` witch **logger** you want to show
- **show_category** `show_category=['acc','f1' ...]` witch **evaluation metrics** you want to show, it is depend on your `evaluation_function`
- **save** save figure or not
- **show** plt.show() or not

<div id="logger_plot_experiment"></div>
