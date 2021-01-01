#!/usr/bin/env python
# coding: utf-8

# In[1]:


#论文复现：Uilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence
import string
import re
import pickle
from os import listdir
from nltk.corpus import stopwords
from pickle import dump,load
import numpy as np
from itertools import chain
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,mean_absolute_error,confusion_matrix,roc_auc_score
import torch.nn as nn
import tokenizer
import pandas as pd
import os
import random
import torch
import json
import sys
import math
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam
import math
from torch.optim.optimizer import Optimizer, required
from transformers import BertModel,get_linear_schedule_with_warmup,AutoTokenizer,AutoModel
from torch.utils.data import Dataset,DataLoader
from math import exp, log


# In[2]:


hidden_size=768
batch_size=16
drop=0.3
class_num=5
epoch=3


# In[3]:


seed=1234
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# In[4]:


gpu_list = [0,1,2,3]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
# 这里注意，需要指定一个 GPU 作为主 GPU。
# 否则会报错：module must have its parameters and buffers on device cuda:1 (device_ids[0]) but found one of them on device: cuda:2
# 参考：https://stackoverflow.com/questions/59249563/runtimeerror-module-must-have-its-parameters-and-buffers-on-device-cuda1-devi
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
print('use device: %s' %device,file=sys.stderr)
#print("device_count :{}".format(torch.cuda.device_count()))


# In[5]:


file="semeval_QA_M"
train_file=None
dev_file=None
test_file=None
if file=="semeval_QA_M":
    train_file='./ABSA-BERT-pair/data/semeval2014/bert-pair/train_QA_M.csv'
    test_file='./ABSA-BERT-pair/data/semeval2014/bert-pair/test_QA_M.csv'
elif file=="semeval_NLI_M":
    train_file='./ABSA-BERT-pair/data/semeval2014/bert-pair/train_NLI_M.csv'
    test_file='./ABSA-BERT-pair/data/semeval2014/bert-pair/test_NLI_M.csv'
elif file=="semeval_QA_B":
    train_file='./ABSA-BERT-pair/data/semeval2014/bert-pair/train_QA_B.csv'
    test_file='./ABSA-BERT-pair/data/semeval2014/bert-pair/test_QA_B.csv'
elif file=="semeval_NLI_M":
    train_file='./ABSA-BERT-pair/data/semeval2014/bert-pair/train_NLI_M.csv'
    test_file='./ABSA-BERT-pair/data/semeval2014/bert-pair/test_NLI_M.csv'


# In[6]:


tokenizer = BertTokenizer.from_pretrained("../../data/bert-based-uncase/")


# In[7]:


data=pd.read_csv(train_file,sep='\t',header=None)
data.columns=["idx","label","sentence1","sentence2"]


# In[8]:


set(data['label'])


# In[9]:


class myDataset(Dataset):
    def __init__(self,dataFile):
        self.data=pd.read_csv(dataFile,sep='\t',header=None)
        self.data.columns=["idx","label","sentence1","sentence2"]
        self.len=self.data.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        return self.data.loc[index,'sentence1'],self.data.loc[index,'sentence2'],self.data.loc[index,'label']
    
    @staticmethod
    def collate_fn(batch):
        sent1,sent2,label=zip(*batch)
        token_sent1=[tokenizer.tokenize(sent) for sent in sent1]
        token_sent2=[tokenizer.tokenize(sent) for sent in sent2]
        attention_mask=[]
        token_type_ids=[]
        sent=[]
        for i in range(len(token_sent1)):
            token_sent1[i].insert(0,'[CLS]')
            token_type_ids.append([1]*len(token_sent1[i]))
            token_sent2[i].insert(0,'[SEP]')
            alltoken=token_sent1[i]
            alltoken.extend(token_sent2[i])
            token_type_ids[i].extend([0]*len(token_sent2[i]))
            attention_mask.append([1]*len(alltoken))
            sent.append(alltoken)
        sent=[tokenizer.convert_tokens_to_ids(sen) for sen in sent]
        attention_mask=myDataset.to_input_tensor(attention_mask)
        token_type_ids=myDataset.to_input_tensor(token_type_ids)
        return sent,label,attention_mask,token_type_ids
    
    @staticmethod
    def sentiment2id(sentiments):
        senti2id={"positive":0,"negative":1,"none":2,"neutral":3,"conflict":4}
        return [senti2id[senti] for senti in sentiments]
    
    @staticmethod
    def to_input_tensor(sents):
        return torch.nn.utils.rnn.pad_sequence([torch.tensor(sent) for sent in sents],batch_first=True,padding_value=tokenizer.convert_tokens_to_ids(['[PAD]'])[0])


# In[10]:


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.bert=BertModel.from_pretrained("../../data/bert-based-uncase/")
        self.dropout=nn.Dropout(drop)
        self.predict=nn.Linear(hidden_size,class_num)
    
    def forward(self,sentence,attention_mask,token_type_ids):
        #print('sentence.device',sentence.device)
        #print('attention_mask.device',attention_mask.device)
        #print('token_type_ids.device',token_type_ids.device)
        output=self.bert(input_ids=sentence,attention_mask=attention_mask,token_type_ids=token_type_ids)
        result=self.dropout(output[1])
        result=self.predict(result)
        return result


# In[11]:


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
    auc=roc_auc_score(labels,preds, multi_class='ovr')
    #auc=roc_auc_score(labels,preds)
    return accuracy,precision,recall,f1score,cm,auc


# In[12]:


def train():
    model=Model()
    model=model.to(device)
    max_accuracy=0
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-5)
    trainLoader=DataLoader(myDataset(train_file),batch_size,shuffle=True,collate_fn=myDataset.collate_fn)
    crossentry=nn.CrossEntropyLoss()
    for i in range(epoch):
        model.train()
        allLoss=[]
        alllabel=[]
        allresult=[]
        for j,batch in enumerate(trainLoader):
            optimizer.zero_grad()
            sents,label,attention_mask,token_type_ids=batch
            label=torch.LongTensor(myDataset.sentiment2id(label)).to(device)
            #label=torch.LongTensor(label).to(device)
            sents=myDataset.to_input_tensor(sents).to(device)
            attention_mask=attention_mask.to(device)
            token_type_ids=token_type_ids.to(device)
            pred=model(sents,attention_mask,token_type_ids)
            _,result=torch.max(pred,1)
            loss=crossentry(pred,label)
            allLoss.append(loss.item())
            alllabel.append(cuda2cpu(label))
            allresult.append(cuda2cpu(result))
            loss.backward()
            optimizer.step()
            
        torch.save(model,'model_semeval_QA_M.pkl')


# In[13]:


def test(model,dataLoader):
    model.eval()
    with torch.no_grad():
        allLoss=[]
        allResult=[]
        allLabel=[]
        allPred=[]
        crossentry=nn.CrossEntropyLoss()
        for i,batch in enumerate(dataLoader):
            sents,label,attention_mask,token_type_ids=batch
            label=torch.LongTensor(myDataset.sentiment2id(label)).to(device)
            #label=torch.LongTensor(label).to(device)
            attention_mask=attention_mask.to(device)
            token_type_ids=token_type_ids.to(device)
            sents=myDataset.to_input_tensor(sents).to(device)
            pred=model(sents,attention_mask,token_type_ids)
            _,result=torch.max(pred,1)
            loss=crossentry(pred,label)
            allLoss.append(loss.item())
            allLabel.append(cuda2cpu(label))
            allResult.append(cuda2cpu(result))
            #pred=torch.index_select(torch.softmax(pred,1),1,torch.tensor([1]).to(device))
            pred=torch.softmax(pred,1)
            allPred.append(cuda2cpu(pred))
        allLabel=list(chain.from_iterable(allLabel))
        allResult=list(chain.from_iterable(allResult))
        allPred=list(chain.from_iterable(allPred))
        accuracy,_,_,_,_,auc=estimate(allLabel,allResult,allPred)
        allLoss=np.mean(allLoss)
        print('[INFO] TEST accuracy:{},AUC:{} loss:{}'.format(accuracy,auc,allLoss))
        


# In[14]:


train()


# In[15]:


model=torch.load('model_semeval_QA_M.pkl')
testLoader=DataLoader(myDataset(test_file),batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
test(model,testLoader)


# sentihood_QA_M ACC:93.8 AUC:97.6
# sentihood_NLI_M ACC:93.2 AUC:96.8
# sentihood_QA_B ACC:95.4 AUC:98.8
# sentihood_NL_B ACC:95.6 AUC:98.8
# 
# semeval_semeval_QA_M :91.6 AUC 96.3
# semeval_NLI_M ACC:91.3 AUC:98.8
# semeval_QA_B ACC:96.2 AUC: 98.6
# semeval_NLI_B ACC:96.6 AUC: 98.8
# 
# 
# 
# 
# 
# 发现在bert的最后加了softmax之后结果竟然有较大的下降？？

# In[ ]:




