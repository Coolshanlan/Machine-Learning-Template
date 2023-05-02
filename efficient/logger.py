import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import SubplotSpec
import matplotlib as mpl
import os
import yaml
import torch

plt.rcParams["font.family"] = "Serif"
class Config(dict):
    def __setattr__(self, key,val):
        if isinstance(val,torch.device):
            val = str(val)
        self.__dict__[key] = val

    def __setitem__(self, item,val):
        if isinstance(val,torch.device):
            val = str(val)
        self.__dict__[item] = val

    def __getattr__(self, attr):
        return self.__dict__[attr]

    def __getitem__(self, key):
        return self.__dict__[key]

    def __call__(self, **kwargs):
        for k,v in kwargs.items():
            self[k]=v

    def __str__(self):
        return str(self.__dict__)

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold',fontsize=24)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

class Logger():
    history = {'records':pd.DataFrame(columns=['experiment_name','tag']),
                        'configs':{}}

    #TODO not support multiple experiment
    def __init__(self,experiment_name,save_dir=None):

        Logger.history['configs'][experiment_name] = Config()
        self.config=Logger.history['configs'][experiment_name]
        self.experiment_name=experiment_name
        self.loaded=False

        if save_dir is None:
            self.save_dir=os.path.join('logger_file',self.experiment_name)
        else:
            self.save_dir=os.path.join(self.save_dir,self.experiment_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.tag='init'


    def __call__(self,**kwargs):
        new_row = pd.DataFrame(kwargs,index=[0],dtype=np.float64)
        new_row['tag']=self.tag
        new_row['experiment_name']=self.experiment_name
        Logger.history['records'] = pd.concat([Logger.history['records'],new_row])

    def __getitem__(self, item):
        self.tag = item
        return self


    def get_last_record(self):
        return Logger.history['records'][Logger.history['records'].tag == self.tag].iloc[-1]

    def get_best_record(self,category='loss',mode='min'):
        best_index = Logger.history['records'][Logger.history['records'].tag == self.tag][category].idxmin() if mode == 'min' else Logger.history['records'][Logger.history['records'].tag == self.tag][category].idxmax()
        return best_index, Logger.history['records'][Logger.history['records'].tag == self.tag].iloc[best_index]

    def check_best(self,category='loss',mode='min'):
        best_index,best_record = self.get_best_record(category,mode)
        return (len(Logger.history['records'][Logger.history['records'].tag == self.tag])-1)==best_index



    def plot(self,show_experiment=None,show_tag=None,
             show_category=None,
             figsize=(7.6*1.5,5*1.5),
             cmp=mpl.cm.Set2.colors,
             ylim={},
             filename='logger_plot.png',
             save=True,
             show=True):

        if not show_experiment:
            show_experiment=self.experiment_name

        if isinstance(show_experiment,list):
            self.plot_multi_experiment(show_experiment=show_experiment,show_tag=show_tag,
             show_category=show_category,
             figsize=figsize,
             cmp=cmp,
             ylim=ylim,
             filename=filename,
             save=save,
             show=show)
            return



        history_df =  Logger.history['records']
        history_df = history_df[history_df.experiment_name == show_experiment].drop(columns=['experiment_name'])

        if not show_tag:
            show_tag = list(history_df.tag.unique())

        exist_category=set([])
        for logger_tag in show_tag:
            _tmp=history_df[history_df.tag==logger_tag].drop(columns=['tag'])
            _tmp = _tmp.dropna(axis=1)
            exist_category.update(_tmp.columns)

        if show_category:
            show_category = [c for c in show_category if c in exist_category]
        else:
            show_category=sorted(exist_category)

        fig, axs = plt.subplots(1,len(show_category),figsize=(len(show_category)*figsize[0]+len(show_category)*0.25,figsize[1]))#,constrained_layout=True)
        plt.ticklabel_format(style='plain', axis='x', useOffset=False)

        axs = np.array(axs).flatten()

        for lidx,logger_tag in enumerate(show_tag):
            plot_color = cmp[lidx]
            for cidx,c in enumerate(show_category):
                if c in history_df[history_df.tag==logger_tag].drop(columns=['tag']).dropna(axis=1).columns:
                    _record=history_df[history_df.tag==logger_tag]
                    _record = _record.reset_index(drop=True)
                    axs[cidx].plot(range(len(_record)),_record[c],label=logger_tag,color=plot_color,linewidth=2)
                    axs[cidx].set_title('{}'.format(c), fontsize=20)
                    axs[cidx].legend(loc='upper left',fontsize=17)
                    axs[cidx].tick_params(axis='both', labelsize=17)
                    axs[cidx].grid(axis='y', linestyle='-', alpha=0.7,color='lightgray')
                    axs[cidx].xaxis.set_major_locator(MaxNLocator(integer=True))
                    if c in ylim.keys():
                        axs[cidx].set_ylim(ylim[c][0],ylim[c][1])
        fig.suptitle(self.experiment_name, fontsize=22)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir, filename))
        if show:
            plt.show()
        plt.close()


    def plot_experiments(self,show_experiment=None,show_tag=None,
             show_category=None,
             figsize=(7.6*1.5,5*1.5),
             cmp=mpl.cm.Set1.colors,
             ylim={},
             filename='experiments_history.png',
             save=True,
             show=True):

        if not show_experiment:
            show_experiment=list(Logger.history['records'].experiment_name.unique())

        if len(show_experiment)==1:
            show_experiment=show_experiment[0]

        if isinstance(show_experiment,str):
            self.plot(show_experiment=show_experiment,show_tag=show_tag,
             show_category=show_category,
             figsize=figsize,
             cmp=cmp,
             ylim=ylim,
             filename=filename,
             save=save,
             show=show)
            return
        history_df =  Logger.history['records']
        history_df = history_df[history_df.experiment_name.apply(lambda x: x in show_experiment )]
        if not show_tag:
            show_tag = list(history_df.tag.unique())

        exist_category=set([])
        for logger_tag in show_tag:
            _tmp=history_df[history_df.tag==logger_tag].drop(columns=['tag','experiment_name'])
            _tmp = _tmp.dropna(axis=1)
            exist_category.update(_tmp.columns)

        if show_category:
            show_category = [c for c in show_category if c in exist_category]
        else:
            show_category=sorted(exist_category)

        fig, axs = plt.subplots(len(show_tag),len(show_category),figsize=(len(show_category)*figsize[0]+len(show_category)*0.25,(figsize[1]+1)*len(show_tag)))#,constrained_layout=True)
        plt.ticklabel_format(style='plain', axis='x', useOffset=False)

        axs = np.array(axs).flatten()

        grid = plt.GridSpec(len(show_tag), len(show_category))
        for lidx,logger_tag in enumerate(show_tag):
            create_subtitle(fig, grid[lidx, ::], logger_tag)
            for cidx,c in enumerate(show_category):
                axidx = lidx*len(show_category)+cidx
                for nidx,n in enumerate(show_experiment):
                    plot_color = cmp[nidx]
                    _record=history_df[(history_df.tag==logger_tag) & (history_df.experiment_name==n)]
                    if c in _record.drop(columns=['tag']).dropna(axis=1).columns:
                        _record = _record.reset_index(drop=True)
                        axs[axidx].plot(range(len(_record)),_record[c],label='{}({})'.format(n,logger_tag),color=plot_color,linewidth=2)
                        axs[axidx].set_title('{}'.format(c), fontsize=20)
                        axs[axidx].legend(loc='upper left',fontsize=15)
                        axs[axidx].tick_params(axis='both', labelsize=15)
                        axs[axidx].grid(axis='y', linestyle='-', alpha=0.7,color='lightgray')
                        axs[axidx].xaxis.set_major_locator(MaxNLocator(integer=True))
                        if c in ylim.keys():
                            axs[axidx].set_ylim(ylim[c][0],ylim[c][1])
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir,filename))
        if show:
            plt.show()
        plt.close()


    def remove_history(self,experiment_name):
        Logger.history['records'] = Logger.history['records'][Logger.history['records'].experiment_name != experiment_name]
        if experiment_name in Logger.history['configs'].keys():
            del Logger.history['configs'][experiment_name]


    def save_config(self):
        save_dir=os.path.join(self.save_dir,'config.yaml')
        with open(save_dir,'w') as file:
            yaml.dump(self.config,file)

        print(f'Export Success! \n File Path:{save_dir}')


    def load_config(self,experiment_name=None):
        if experiment_name is None:
            experiment_name = self.experiment_name
        load_dir=os.path.join(self.save_dir,'config.yaml')
        with open(load_dir,'r') as file:
            config=yaml.load(file, Loader=yaml.FullLoader)
        return Config(config)


    def export_logger(self,dir_path=None,filename='logger_history',overwrite=False):

        if not dir_path: dir_path = self.save_dir
        save_path=os.path.join(dir_path,filename)
        if os.path.exists(save_path+'.csv') and not overwrite and not self.loaded:
            self.load_logger(dir_path=dir_path,filename=filename)

        Logger.history['configs'][self.experiment_name]=self.config

        with open(save_path+'.json','w') as f:
            json_str=json.dumps(Logger.history['configs'], indent=2, sort_keys=True)
            f.write(json_str)

        Logger.history['records'].to_csv(save_path+'.csv',index=False)

        print(f'Export Success! \n File Path:{save_path}')


    def load_logger(self,dir_path=None,filename='logger_history',overwrite=False):
        if not dir_path: dir_path = self.save_dir
        filepath = os.path.join(dir_path,filename)
        previous_history = pd.read_csv(filepath+'.csv')
        with open(filepath+'.json','r') as json_file:
            Logger.history['configs'] = json.load(json_file)

        repeat_names=set(previous_history.experiment_name.unique()) & set(Logger.history['records'].experiment_name.unique())
        # if len(repeat_names) != 0 and not overwrite:
        #     raise Exception(f'{repeat_names} experiment name already exist, please rename or set overwrite parameter to True.')
        if len(repeat_names) != 0 and overwrite:
            previous_history=previous_history.loc[~previous_history.experiment_name.isin(repeat_names)]
            Logger.history['configs'] = json.load(json_file)
            if len(repeat_names)>0:
                for k in Logger.history['configs'].keys():
                    if k in repeat_names:
                        del Logger.history['configs'][k]
        self.loaded=True
        Logger.history['records']=pd.concat([previous_history,Logger.history['records']])



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
    #self.load_logger()
    logger = Logger('test')
    config = logger.config
    config.batch=3
    config.lr=0.3
    config.seed=54
    for i in range(10):
        logger['train'](acc=np.random.rand(),f1score=np.random.rand(),fff=np.random.rand())
        logger['all'](acc=np.random.rand(),f1score=np.random.rand(),ff=np.random.rand(),f1=np.random.rand())
        logger['validation'](acc=np.random.rand())
        logger['validation'].check_best(category='acc',mode='max')

    logger.plot(show_category=None,
                ylim={'acc':[0,1]},
                show=False,
                save=True)
    logger.plot_experiments(show_category=None,
                ylim={'acc':[0,1]},
                show=False,
                save=True)

    # # export logger demo
    logger.export_logger()
