from turtle import width
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import pickle
from pathlib import Path

plt.rcParams["font.family"] = "Serif"

class Logger:
    logger_dict={}
    log_category=set([])
    save_dir=Path('logger_dir')

    def __init__(self,name):
        self.name=name
        self.history=pd.DataFrame()
        self.current_history=pd.DataFrame()
        Logger.logger_dict[self.name] = pd.DataFrame()
        self.epoch=0
        if not Logger.save_dir.exists():
            Logger.save_dir.mkdir()


    def update_category(self):
        Logger.log_category.update(set(self.current_history.columns.to_list()))

    def __call__(self,**kwargs):
        new_row = pd.DataFrame(kwargs,index=[0])
        self.current_history=pd.concat([self.current_history,new_row])

    def __str__(self,r=3):
        output_str=''
        if len(self.current_history)>0:
            avg_epoch = self.get_current_epoch_avg()
            for log_c in avg_epoch.index:
                output_str+= 'avg{}:{} \t'.format(log_c,f'%.{r}f'%avg_epoch[log_c])
            return output_str
        else:
            avg_epoch = self.history[self.history.epoch == self.epoch-1].iloc[-1]
            for log_c in avg_epoch.index:
                output_str+= 'avg{}:{} \t'.format(log_c,f'%.{r}f'%avg_epoch[log_c])
            return output_str

    def get_current_epoch_avg(self):
        return self.current_history.mean()

    def get_last_epoch_avg(self):
        return self.history[self.history.epoch == (self.epoch-1)].mean()

    def get_best_record(self,category='loss',mode='min',unit='epoch'):
        _history = self.history.groupby('epoch').mean() if unit == 'epoch' else self.history
        best_index = _history[category].idxmin() if mode == 'min' else _history[category].idxmax()
        return best_index, _history.iloc[best_index]

    def check_best(self,category='loss',mode='min',unit='epoch'):
        _history = self.history.groupby('epoch').mean() if unit == 'epoch' else self.history
        best_index,best_record = self.get_best_record(category,mode,unit)
        return (len(_history[category])-1)==best_index

    def save_epoch(self):
        self.update_category()
        current_category=self.current_history.columns
        self.current_history['epoch'] = self.epoch
        self.epoch += 1
        self.history =pd.concat([self.history,self.current_history.copy()]) #self.history.append(self.current_history.copy()).reset_index(drop=True)
        self.current_history=pd.DataFrame(columns=current_category)
        Logger.logger_dict[self.name]=self.history

    @staticmethod
    def get_logger_names():
        return list(Logger.logger_dict.keys())

    @staticmethod
    def get_loggers():
        return Logger.logger_dict

    @staticmethod
    def plot(show_logger=None,
             show_category=None,
             figsize=(7.6*1.5,5*1.5),
             cmp=mpl.cm.Set2.colors,
             ylim={},
             unit='epoch',
             filename='logger_history.png',
             save=True,
             show=True):

        if not show_logger:
            show_logger = Logger.logger_dict.keys()

        if not show_category:
            show_category = Logger.log_category

        exist_category=set([])
        for logger_name in show_logger:
            exist_category.update(Logger.logger_dict[logger_name])

        show_category = [c for c in show_category if c in exist_category]

        fig, axs = plt.subplots(1,len(show_category),figsize=(len(show_category)*figsize[0]+len(show_category)*0.25,figsize[1]))#,constrained_layout=True)
        plt.ticklabel_format(style='plain', axis='x', useOffset=False)

        axs = np.array(axs).flatten()

        for lidx,logger_name in enumerate(show_logger):
            plot_color = cmp[lidx]
            for cidx,c in enumerate(show_category):
                if c in Logger.logger_dict[logger_name]:
                    _history=Logger.logger_dict[logger_name]
                    _history = _history.groupby('epoch').mean() if unit == 'epoch' else _history
                    _history = _history.reset_index(drop=True)
                    axs[cidx].plot(range(len(_history)),_history[c],label=logger_name,color=plot_color,linewidth=2)
                    axs[cidx].set_title('{}'.format(c), fontsize=20)
                    axs[cidx].legend(loc='upper left',fontsize=15)
                    axs[cidx].tick_params(axis='both', labelsize=15)
                    axs[cidx].grid(axis='y', linestyle='-', alpha=0.7,color='lightgray')
                    axs[cidx].xaxis.set_major_locator(MaxNLocator(integer=True))
                    if c in ylim.keys():
                        axs[cidx].set_ylim(ylim[c][0],ylim[c][1])
        plt.tight_layout()
        if save:
            plt.savefig(Logger.save_dir/filename)
        if show:
            plt.show()

    def export_logger(filename='logger_history.pkl'):
        path = Logger.save_dir/filename
        with open(path,'wb') as f:
            pickle.dump({'loggers':Logger.logger_dict,
                         'category':Logger.log_category}, f)

    def load_logger(filepath='logger_history.pkl'):
        path = filepath
        with open(path, 'rb') as f:
            load_data=pickle.load(f)
            Logger.logger_dict = load_data['loggers']
            Logger.log_category = load_data['category']



#教學
# check_best(category='loss',mode='min',unit='epoch')
# get_best_record(category='loss',mode='min',unit='epoch')
# logger.epoch_history['loss'].iloc[-1]
# logger.save_epoch()
# plot(logger_list=[],show_category=None,figsize=(6.2,3),ylim=None,unit='epoch')

# 無須設定存入變數種類，直接丟即可
# 可以在plot時設定show_category來選擇顯示參數
# 如果想看iter版本，plot時unit給iter(非epoch)
# save_epoch一定要call才可以畫圖，即使是iter版本
# 要先save_epoch 才能 check_best 以及 get_best_record

#example
if __name__ == '__main__':
    training_logger=Logger('Training')#default 0
    validation_logger=Logger('Validation')#default 0
    all_logger=Logger('All')#default 0
    for i in range(10):
        for j in range(10):
            training_logger(acc=np.random.rand(),f1score=np.random.rand())
            all_logger(acc=np.random.rand(),f1score=np.random.rand(),ff=np.random.rand(),f1=np.random.rand())
            validation_logger(acc=np.random.rand())
        validation_logger.save_epoch()
        training_logger.save_epoch()
        all_logger.save_epoch()
        validation_logger.check_best(category='acc',mode='max',unit='epoch')

    #Logger.load_logger('logger_dir/logger_history.pkl')
    Logger.plot(show_logger=Logger.get_logger_names(),
                show_category=['acc','f1score'],
                ylim={'acc':[0,1]},
                show=True,
                save=True)

    # Logger.plot(show_category= ['loss','acc'],unit='iter')
    # Logger.plot(show_logger=['Training','All'])
    # # Logger.export_logger()