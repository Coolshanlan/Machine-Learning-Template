# Efficient-Pytorch-Template
 This is a efficient pytorch template that support logger, dataset and all utils that you need during training and inference.

# Using Example
```=python
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
  return accuracy_score(label,pred)

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
  record,evaluation = model_instance.run_dataloader(valid_dataloader,logger=valid_logger,update=True)
  if valid_logger.check_best('loss',mode='min):
    model_instance.save(only_model=True,filename='best_model.pkl')

# Visualize training history
Logger.plot()
```
