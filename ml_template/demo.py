import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data.dataset import Dataset
from logger import Logger
from model_instance import Model_Instance
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

train_transform = transforms.Compose(
    [transforms.RandomRotation(20),
     transforms.RandomHorizontalFlip(0.3),
     transforms.ColorJitter(brightness = 0.1, # Randomly adjust color jitter of the images
                            contrast = 0.1,
                            saturation = 0.1),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_dataloader(batch_size):

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
    return trainloader,testloader

def evaluation(model_instance,dataloader,logger=None):
  # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    classes_acc={classname: 0 for classname in classes}
    correct=0
    total=0

    outcome,record_df = model_instance.run_dataloader(dataloader,update=False,display_result=True,display_progress=True,logger=logger)
    for prediction , label in zip(outcome['pred'],outcome['label']):
        if label == prediction.argmax(axis=0):
            correct+=1
            correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1
        total+=1

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        classes_acc[classname]=[accuracy]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    record_df = pd.DataFrame.from_dict(classes_acc)
    return record_df



classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

logger = Logger('test')
config=logger.config
config['train_step']=300000
config['valid_step']=50000
config['lr']=1e-3
config['batch_size']=512
config['device']= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config['amp']=False
print(config['device'])

trainloader,testloader = get_dataloader(config.batch_size)

torch.cuda.empty_cache()
model = torchvision.models.resnet101()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr=config['lr'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.train_step)
model_instance = Model_Instance(model=model,
                                optimizer=optimizer,
                                loss_function=lambda x,y: criterion(x.float(),y),
                                evaluation_metrics=['acc','f1score'],
                                scheduler=scheduler,#iter
                                scheduler_epoch=False,# default False
                                device=config.device,
                                amp=config.amp)

model_instance.run_step_dataloader(trainloader,
                                   run_step=config.train_step,
                                   valid_step=config.valid_step,
                                   logger=logger['Train'],
                                   display_progress=False,
                                   display_result=False,
                                   evaluation_function=lambda: evaluation(model_instance,
                                                                          testloader,
                                                                          logger['Valid']))#model_instance.run_dataloader(testloader,logger=logger['Valid'],update=False))

