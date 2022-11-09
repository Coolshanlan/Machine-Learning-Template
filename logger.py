import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import SubplotSpec
import matplotlib as mpl
import os


plt.rcParams["font.family"] = "Serif"
class Config(dict):
    def __setattr__(self, attr,val):
        self[attr]=val
    def __getattr__(self, attr):
        return self[attr]
    def __call__(self, **kwargs):
        for k,v in kwargs.items():
            self[k]=v

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold',fontsize=24)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

class Logger:
    logger_dict={}
    log_category=set([])
    save_dir='logger_dir'
    config=None
    experiment_name=None
    history={'records':pd.DataFrame(columns=['experiment_name','epoch','tag']),
             'configs':{}}

    @staticmethod
    def init(experiment_name):
        Logger.config=Config()
        Logger.experiment_name=experiment_name
        Logger.history[Logger.experiment_name]={}

    def __init__(self,tag):
        if not Logger.experiment_name:
            raise Exception('Use Logger.init(experiment_name) to initial Logger first.')

        self.tag=tag
        self.record=pd.DataFrame()
        Logger.logger_dict[self.tag] = pd.DataFrame()
        self.epoch=0
        if not os.path.exists(Logger.save_dir):
            os.mkdir(Logger.save_dir)


    def update_category(self,new_row):
        Logger.log_category.update(set(new_row.columns.to_list()))

    def __call__(self,**kwargs):
        new_row = pd.DataFrame(kwargs,index=[self.epoch],dtype=np.float64)
        self.update_category(new_row)
        new_row['epoch']=self.epoch
        self.record=pd.concat([self.record,new_row])
        Logger.logger_dict[self.tag]=self.record
        self.epoch+=1

    def get_last_record(self):
        return self.record.iloc[-1]

    def get_best_record(self,category='loss',mode='min'):
        best_index = self.record[category].idxmin() if mode == 'min' else self.record[category].idxmax()
        return best_index, self.record.iloc[best_index]

    def check_best(self,category='loss',mode='min'):
        best_index,best_record = self.get_best_record(category,mode)
        return (len(self.record)-1)==best_index

    @staticmethod
    def plot(show_tag=None,
             show_category=None,
             figsize=(7.6*1.5,5*1.5),
             cmp=mpl.cm.Set2.colors,
             ylim={},
             filename='logger_history.png',
             save=True,
             show=True):

        if not show_tag:
            show_tag = Logger.logger_dict.keys()

        exist_category=set([])
        for logger_tags in show_tag:
            exist_category.update(Logger.logger_dict[logger_tags].drop(columns=['epoch']))

        if show_category:
            show_category = [c for c in show_category if c in exist_category]
        else:
            show_category = exist_category

        fig, axs = plt.subplots(1,len(show_category),figsize=(len(show_category)*figsize[0]+len(show_category)*0.25,figsize[1]))#,constrained_layout=True)
        plt.ticklabel_format(style='plain', axis='x', useOffset=False)

        axs = np.array(axs).flatten()

        for lidx,logger_tags in enumerate(show_tag):
            plot_color = cmp[lidx]
            for cidx,c in enumerate(show_category):
                if c in Logger.logger_dict[logger_tags]:
                    _record=Logger.logger_dict[logger_tags]
                    _record = _record.reset_index(drop=True)
                    axs[cidx].plot(range(len(_record)),_record[c],label=logger_tags,color=plot_color,linewidth=2)
                    axs[cidx].set_title('{}'.format(c), fontsize=20)
                    axs[cidx].legend(loc='upper left',fontsize=15)
                    axs[cidx].tick_params(axis='both', labelsize=15)
                    axs[cidx].grid(axis='y', linestyle='-', alpha=0.7,color='lightgray')
                    axs[cidx].xaxis.set_major_locator(MaxNLocator(integer=True))
                    if c in ylim.keys():
                        axs[cidx].set_ylim(ylim[c][0],ylim[c][1])
        fig.suptitle(Logger.experiment_name,fontsize=22)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(Logger.save_dir,filename))
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_experiment(show_experiment=None,show_tag=None,
             show_category=None,
             figsize=(7.6*1.5,5*1.5),
             cmp=mpl.cm.Set2.colors,
             ylim={},
             filename='experiments_history.png',
             save=True,
             show=True):

        if not show_experiment:
            show_experiment=Logger.experiment_name

        if isinstance(show_experiment,list):
            Logger.plot_multi_experiment(show_experiment=show_experiment,show_tag=show_tag,
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
            _tmp=history_df[history_df.tag==logger_tag].drop(columns=['epoch','tag'])
            _tmp = _tmp.dropna(axis=1)
            exist_category.update(_tmp.columns)

        if show_category:
            show_category = [c for c in show_category if c in exist_category]
        else:
            show_category=exist_category

        fig, axs = plt.subplots(1,len(show_category),figsize=(len(show_category)*figsize[0]+len(show_category)*0.25,figsize[1]))#,constrained_layout=True)
        plt.ticklabel_format(style='plain', axis='x', useOffset=False)

        axs = np.array(axs).flatten()

        for lidx,logger_tag in enumerate(show_tag):
            plot_color = cmp[lidx]
            for cidx,c in enumerate(show_category):
                if c in history_df[history_df.tag==logger_tag].drop(columns=['epoch','tag']).dropna(axis=1).columns:
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
        fig.suptitle(Logger.experiment_name, fontsize=22)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(Logger.save_dir, filename))
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_multi_experiment(show_experiment=None,show_tag=None,
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
            Logger.plot_experiment(show_experiment=show_experiment,show_tag=show_tag,
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
            _tmp=history_df[history_df.tag==logger_tag].drop(columns=['epoch','tag','experiment_name'])
            _tmp = _tmp.dropna(axis=1)
            exist_category.update(_tmp.columns)

        if show_category:
            show_category = [c for c in show_category if c in exist_category]
        else:
            show_category=exist_category

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
                    if c in _record.drop(columns=['epoch','tag']).dropna(axis=1).columns:
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
            plt.savefig(os.path.join(Logger.save_dir,filename))
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def remove_history(experiment_name):
        Logger.history['records'] = Logger.history['records'][Logger.history['records'].experiment_name != experiment_name]
        if experiment_name in Logger.history['configs'].keys():
            del Logger.history['configs'][experiment_name]

    @staticmethod
    def export_logger(dir_path=None,filename='logger_history',read_if_exist=True,overwrite_if_exist=True):
        if not dir_path: dir_path = Logger.save_dir
        save_path=os.path.join(dir_path,filename)
        if os.path.exists(save_path+'.csv') and read_if_exist:
            Logger.load_logger(dir_path=dir_path,filename=filename)

        combine_df = pd.DataFrame()
        for k,v in Logger.logger_dict.items():
            v['tag'] = k
            combine_df = pd.concat([combine_df,v])
        combine_df['experiment_name']=Logger.experiment_name

        if overwrite_if_exist:
            Logger.remove_history(Logger.experiment_name)

        Logger.history['records']=pd.concat([combine_df,Logger.history['records']])
        Logger.history['configs'][Logger.experiment_name]=Logger.config

        with open(save_path+'.json','w') as f:
            json_str=json.dumps(Logger.history['configs'], indent=2, sort_keys=True)
            f.write(json_str)

        Logger.history['records'].to_csv(save_path+'.csv',index=False)

    @staticmethod
    def load_logger(dir_path=None,filename='logger_history'):
        if not dir_path: dir_path = Logger.save_dir
        filepath = os.path.join(dir_path,filename)
        Logger.history['records']=pd.read_csv(filepath+'.csv')
        with open(filepath+'.json','r') as json_file:
            Logger.history['configs'] = json.load(json_file)



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
    # Logger.load_logger(os.path.join(Logger.save_dir,'logger_history.pkl'))
    Logger.init('test2')
    config = Logger.config
    config.batch=243
    config.lr=0.1
    config.seed=54
    training_logger=Logger('Training')#default 0
    validation_logger=Logger('Validation')#default 0
    all_logger=Logger('All')#default 0
    for i in range(10):
        training_logger(acc=np.random.rand(),f1score=np.random.rand(),fff=np.random.rand())
        all_logger(acc=np.random.rand(),f1score=np.random.rand(),ff=np.random.rand(),f1=np.random.rand())
        validation_logger(acc=np.random.rand())
        validation_logger.check_best(category='acc',mode='max')

    Logger.plot(show_category=None,
                ylim={'acc':[0,1]},
                show=False,
                save=True)

    # # export logger demo
    Logger.export_logger()

    # # load logger demo
    # Logger.load_logger(dir_path=Logger.save_dir,filename='logger_history')

    # # remove demo
    # Logger.remove_history(experiment_name='test3')
    # Logger.export_logger(read_if_exist=False)

    # Logger.plot_multi_history(show_category=['acc','f1score'],filename='all_history.png',show=False)
    # # print(Logger.history['configs']['test2'])