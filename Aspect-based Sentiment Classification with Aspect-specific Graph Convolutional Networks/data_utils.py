import random
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,mean_absolute_error,confusion_matrix
from torch.utils.data import Dataset

def setSeed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

class myDataset(Dataset):
    pad=0
    def __init__(self,sents,aspect_index,aspect_len,adj_matrix,y,pad):
        self.sents=sents
        self.aspect_index=aspect_index
        self.aspect_len=aspect_len
        self.adj_matrix=adj_matrix
        self.y=y
        self.len=len(sents)
        myDataset.pad=pad
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        return self.sents[index],self.aspect_index[index],self.aspect_len[index],self.adj_matrix[index],self.y[index]
    
    @staticmethod
    def to_input_tensor(sents):
        return torch.nn.utils.rnn.pad_sequence([torch.tensor(input_id) for input_id in sents],batch_first=True,padding_value=myDataset.pad)
    
    @staticmethod
    def collate_fn(batch):
        sents,aspect_index,aspect_len,adj_matrix,y=zip(*batch)
        max_len=max([len(sent) for sent in sents])
        adj_mat=[]
        for adj in adj_matrix:
            adj=np.pad(adj,((0,max_len-len(adj)),(0,max_len-len(adj))),'constant')
            adj_mat.append(adj)
        return sents,aspect_index,aspect_len,adj_mat,y

def cuda2cpu(pred):
    #cuda变量转cpu变量
    if type(pred)==list:
        return pred
    if pred.is_cuda:
        pred_cpu=list(pred.cpu().numpy())
    else:
        pred_cpu=list(pred.numpy())
    return pred_cpu

def estimate(labels,results,preds):
    labels=cuda2cpu(labels)
    preds=cuda2cpu(preds)
    accuracy=accuracy_score(labels,results)
    precision=precision_score(labels,results,average='macro')
    recall=recall_score(labels,results,average='macro')
    f1score=f1_score(labels,results,average='macro')
    cm=confusion_matrix(labels,results)
    #auc=roc_auc_score(labels,preds, multi_class='ovr')
    #auc=roc_auc_score(labels,preds)
    return accuracy,precision,recall,f1score,cm

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m,nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m,nn.Linear):
        m.weight.data.normal_(-0.01,0.01)
        m.bias.data.zero_()