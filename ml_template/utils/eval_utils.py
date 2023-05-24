
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, roc_auc_score, f1_score, recall_score, precision_score, mean_squared_error, mean_absolute_error


def calculate_metrics(metrics_name ):#List
    '''
    return a function that contain multiple
    function input:
        function(pred,label)
    function return:
        dict{metric1:value1, metric2:value2 ...}
    '''
    if not isinstance(metrics_name,list):
        metrics_name=[metrics_name]

    def preprocess_pred_format(pred):
        pred=np.array(pred)
        if len(pred.shape)==1:
            if np.max(pred)>1:
                return pred
            return pred>=0.5
        else:
            return pred.argmax(axis=1)

    metrics_functions={}
    metrics_functions['acc']=lambda pred,label: accuracy_score(label,preprocess_pred_format(pred))
    metrics_functions['f1_score']=lambda pred,label: f1_score(label,preprocess_pred_format(pred), average='macro')
    metrics_functions['f1score']=lambda pred,label: f1_score(label,preprocess_pred_format(pred), average='macro')
    metrics_functions['recall']=lambda pred,label: recall_score(label,preprocess_pred_format(pred), average='macro')
    metrics_functions['precision']=lambda pred,label: precision_score(label,preprocess_pred_format(pred), average='macro')
    metrics_functions['auroc']=lambda pred,label: roc_auc_score(label,pred if len(pred.shape) == 1 else pred[:,1])
    metrics_functions['mse']=lambda pred,label: mean_squared_error(label,pred)
    metrics_functions['mae']=lambda pred,label: mean_absolute_error(label,pred)

    if not set(metrics_name).issubset(set(metrics_functions.keys())):
        raise Exception(f'{set(metrics_name) - set(metrics_functions.keys())} metrics not support.\nNow only support {list(metrics_functions.keys())}')
    return lambda pred,label: {name:metrics_functions[name](pred,label)for name in metrics_name}

def accuracy_each_calsses(prediction,labels,classes_name=None,top_k=1,verbose=True):
  # prepare to count predictions for each class
    labels = np.array(labels).astype(int)
    if not classes_name:
        classes_list = list(set(labels))
    if isinstance(classes_name,dict):
        classes_list = list(classes_name.keys())

    correct_pred = {classname: 0 for classname in classes_list}
    total_pred = {classname: 0 for classname in classes_list}
    classes_acc={classname: 0 for classname in classes_list}
    correct=0
    total=0

    for pred , label in zip(prediction,labels):
        if label == pred:
            correct+=1
            correct_pred[classes_list[label]] += 1
        total_pred[classes_list[label]] += 1
        total+=1
    total_acc = correct / total
    if verbose:
        print(f'Accuracy of the network on the {total} test images: {round(100 * total_acc,2)} %')
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        classes_acc[classname]=[accuracy]
        if verbose:
            print(f'Accuracy for class: {classname} is {accuracy:.1f} %')
    # record_df = pd.DataFrame.from_dict(classes_acc)
    return total_acc, classes_acc