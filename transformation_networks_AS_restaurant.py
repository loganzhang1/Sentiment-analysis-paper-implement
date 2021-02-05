#!/usr/bin/env python
# coding: utf-8

# In[1]:


#论文复现：Transformation Networks for Target-Oriented Sentiment Classification
import string
import re
import pickle
from os import listdir
from nltk.corpus import stopwords
from pickle import dump,load
from torch.nn import init
import numpy as np
from itertools import chain
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,mean_absolute_error,confusion_matrix
import torch.nn as nn
import tokenizer
import pandas as pd
import os
import random
import torch
import json
import sys
import math
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
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


dim_w=300
dim_h=50
dropout=0.3
batch_size=64
L=2
epochs=20
s=3
nk=50
C=40.0
class_num=3


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


class Vocab:
    def __init__(self,trainName,testName):
        self.trainSent,self.trainAspect,self.trainY,self.trainAspectIndex=self.readFile(trainName)
        self.testSent,self.testAspect,self.testY,self.testAspectIndex=self.readFile(testName)
        self.words=['pad']
        self.pad=0
        self.length=0
        self.word_vecs=None
        self._word2id=None
        self.buildVocab()
        self.inGlove=[]
    
    def readFile(self,fileName):
        sentences=[]
        aspects=[]
        y=[]
        aspectIndex=[]
        with open(fileName,'r',encoding='utf-8',newline='\n') as file:
            lines=file.readlines()
            file.close()
            for i in range(0,len(lines),3):
                text_left,_,text_right=[s.lower().strip() for s in lines[i].partition("$T$")]
                aspectIndex.append(len(text_left.split()))
                aspect=lines[i+1].lower().strip()
                text_raw=text_left+" "+aspect+" " +text_right
                text_raw=text_raw.split()
                label=int(lines[i+2])+1
                aspect=aspect.split()
                sentences.append(text_raw)
                aspects.append(aspect)
                y.append(label)
            return sentences,aspects,y,aspectIndex
    
    def words2indices(self,sents):
        if type(sents[0])==list:
            return [[self._word2id[w] for w in s] for s in sents]
        else:
            return [self._word2id[w] for w in sents]
    
    def buildVocab(self):
        tempo=self.trainSent.copy()
        tempo.extend(self.testSent)
        for i in tempo:
            for j in i:
                if j not in self.words:
                    self.words.append(j)
        reverse=lambda x:dict(zip(x,range(len(x))))
        self._word2id=reverse(self.words)
        self.length=len(self._word2id)
        self.word_vecs=np.random.uniform(-0.25,0.25,300*self.length).reshape(self.length,300)
        self.trainAspect=[self.words2indices(aspect) for aspect in self.trainAspect]
        self.testAspect=[self.words2indices(aspect) for aspect in self.testAspect]
        self.trainSent=self.words2indices(self.trainSent)
        self.testSent=self.words2indices(self.testSent)
        
    def load_pretrained_vec(self,fname):
        with open(fname) as fp:
            for line in fp.readlines():
                line=line.split(" ")
                word=line[0]
                if word in self.words:
                    self.inGlove.append(word)
                    self.word_vecs[self._word2id[word]]=np.array([float(x) for x in line[1:]])


# In[6]:


file="restaurant"
train_file=None
test_file=None
if file=="restaurant":
    train_file='../data/semeval14/Restaurants_Train.xml.seg'
    test_file='../data/semeval14/Restaurants_Test_Gold.xml.seg'
elif file=="laptop":
    train_file='../data/semeval14/Laptops_Train.xml.seg'
    test_file='../data/semeval14/Laptops_Test_Gold.xml.seg'
elif file=="twitter":
    train_file='../data/Twitter/raw_data/train.txt'
    test_file='../data/Twitter/raw_data/test.txt'


# In[7]:


'''vocab=Vocab(train_file,test_file)
vocab.load_pretrained_vec('../../data/glove.840B.300d.txt')
if file=="restaurant":
    dump(vocab,open('vocab_restaurant.pkl','wb'))
elif file=="laptop":
    dump(vocab,open('vocab_laptop.pkl','wb'))
elif file=="twitter":
    dump(vocab,open('vocab_twitter.pkl','wb'))'''


# In[8]:


if file=="restaurant":
    vocab=load(open('vocab_restaurant.pkl','rb'))
    batch_size=25
elif file=="laptop":
    vocab=load(open('vocab_laptop.pkl','rb'))
    batch_size=64
elif file=="twitter":
    vocab=load(open('vocab_twitter.pkl','rb'))
    batch_size=64


# In[9]:


class myDataset(Dataset):
    def __init__(self,sents,aspects,y,aspectIndex):
        self.sents=sents
        self.aspects=aspects
        self.y=y
        self.aspectIndex=aspectIndex
        self.len=len(sents)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        return self.sents[index],self.aspects[index],self.y[index],self.aspectIndex[index]
    
    @staticmethod
    def to_input_tensor(sents):
        return torch.nn.utils.rnn.pad_sequence([torch.tensor(input_id) for input_id in sents],batch_first=True,padding_value=vocab.pad)
    
    @staticmethod
    def collate_fn(batch):
        sent,aspect,y,aspectIndex=zip(*batch)
        return sent,aspect,y,aspectIndex


# In[10]:


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


# In[11]:


class Convolution_Feature_Extractor(nn.Module):
    def __init__(self):
        super(Convolution_Feature_Extractor,self).__init__()
    
    def forward(self,data,aspect_Index,aspect_len,sents_len):
        #data:[batch_size,sequence_length,2*dim_h]
        position=torch.zeros(data.shape).to(device)
        for batch in range(data.shape[0]):
            for i in range(data.shape[1]):
                if i in range(aspect_Index[batch]+aspect_len[batch]):
                    position[batch][i]=torch.tensor([1-(aspect_Index[batch]+aspect_len[batch]-i)/C]*(2*dim_h))
                elif i in range(aspect_Index[batch]+aspect_len[batch],sents_len[batch]):
                    position[batch][i]=torch.tensor([1-(i-aspect_Index[batch])/C]*(2*dim_h))
        data=data*position#[batch_size,sequence_length,2*embedding_h]
        return data


# In[12]:


class TNet_LF(nn.Module):
    def __init__(self):
        super(TNet_LF,self).__init__()
        self.bilstm1=nn.LSTM(dim_w,dim_h,batch_first=True,bidirectional=True)
        self.bilstm2=nn.LSTM(dim_w,dim_h,batch_first=True,bidirectional=True)
        weight=torch.FloatTensor(vocab.word_vecs)
        self.wordEmbedding=nn.Embedding.from_pretrained(weight)
        self.Wt=nn.Linear(4*dim_h,2*dim_h)
        self.position=Convolution_Feature_Extractor()
        self.conv=nn.Conv1d(2*dim_h,nk,s,padding=1)
        self.predict=nn.Linear(nk,class_num)
    
    def forward(self,sents,aspect,aspectIndex):
        sents_len=torch.sum(sents!=0,dim=-1)
        aspect_len=torch.sum(aspect!=0,dim=-1)
        sents_embedding=self.wordEmbedding(sents)#[batch_size,sequence_length,embedding_w]
        lstm_sent,(_,_)=self.bilstm1(sents_embedding)#[batch_size,sequence_length,2*embedding_h]
        #context_preserving_transformation
        aspect_embedding=self.wordEmbedding(aspect)#[batch_size,aspect_length,embedding_w]
        lstm_aspect,(_,_)=self.bilstm2(aspect_embedding)#[batch_size,aspect_length,2*embedding_h]
        for i in range(L):
            lstm_aspect=lstm_aspect.permute(0,2,1)#[batch_size,2*embedding_h,aspect_length]
            F_hihj=torch.softmax(torch.bmm(lstm_sent,lstm_aspect),2)#[batch_size,sequence_length,aspect_length]
            lstm_aspect=lstm_aspect.permute(0,2,1)
            ri=torch.bmm(F_hihj,lstm_aspect)#[batch_size,sequence_length,2*embedding_h]
            hi_l=torch.relu(self.Wt(torch.cat((lstm_sent,ri),2)))#[batch_size,sequence_length,2*embedding_h]
            lstm_sent=hi_l+lstm_sent#[batch_size,sequence_length,2*embedding_h]
            lstm_sent=self.position(lstm_sent,aspectIndex,aspect_len,sents_len)
        lstm_sent=lstm_sent.permute(0,2,1)#[batch_size,2*embedding_h,sequence_length]
        lstm_sent=torch.max_pool1d(self.conv(lstm_sent),lstm_sent.shape[2]).squeeze(2)#[batch_size,filter_num]
        pred=self.predict(lstm_sent)
        return pred


# In[25]:


class TNet_AS(nn.Module):
    def __init__(self):
        super(TNet_AS,self).__init__()
        self.bilstm1=nn.LSTM(dim_w,dim_h,batch_first=True,bidirectional=True)
        self.bilstm2=nn.LSTM(dim_w,dim_h,batch_first=True,bidirectional=True)
        weight=torch.FloatTensor(vocab.word_vecs)
        self.wordEmbedding=nn.Embedding.from_pretrained(weight)
        self.Wt=nn.Linear(4*dim_h,2*dim_h)
        self.position=Convolution_Feature_Extractor()
        self.conv=nn.Conv1d(2*dim_h,nk,s,padding=1)
        self.predict=nn.Linear(nk,class_num)
        self.Wtrans=nn.Linear(2*dim_h,1)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,sents,aspect,aspectIndex):
        sents_len=torch.sum(sents!=0,dim=-1)
        aspect_len=torch.sum(aspect!=0,dim=-1)
        sents_embedding=self.wordEmbedding(sents)#[batch_size,sequence_length,embedding_w]
        lstm_sent,(_,_)=self.bilstm1(sents_embedding)#[batch_size,sequence_length,2*embedding_h]
        lstm_sent=self.dropout(lstm_sent)
        #context_preserving_transformation
        aspect_embedding=self.wordEmbedding(aspect)#[batch_size,aspect_length,embedding_w]
        lstm_aspect,(_,_)=self.bilstm2(aspect_embedding)#[batch_size,aspect_length,2*embedding_h]
        for i in range(L):
            lstm_aspect=lstm_aspect.permute(0,2,1)#[batch_size,2*embedding_h,aspect_length]
            F_hihj=torch.softmax(torch.bmm(lstm_sent,lstm_aspect),2)#[batch_size,sequence_length,aspect_length]
            lstm_aspect=lstm_aspect.permute(0,2,1)
            ri=torch.bmm(F_hihj,lstm_aspect)#[batch_size,sequence_length,2*embedding_h]
            hi_l=torch.relu(self.Wt(torch.cat((lstm_sent,ri),2)))#[batch_size,sequence_length,2*embedding_h]
            ti=torch.sigmoid(self.Wtrans(lstm_sent))#[batch_size,sequence_length,1]
            lstm_sent=lstm_sent*(1-ti)+hi_l*ti#[batch_size,sequence_length,2*dim_h]
            lstm_sent=self.position(lstm_sent,aspectIndex,aspect_len,sents_len)
        lstm_sent=lstm_sent.permute(0,2,1)#[batch_size,2*embedding_h,sequence_length]
        lstm_sent=torch.max_pool1d(self.conv(lstm_sent),lstm_sent.shape[2]).squeeze(2)#[batch_size,filter_num]
        lstm_sent=self.dropout(lstm_sent)
        pred=self.predict(lstm_sent)
        return pred


# In[26]:


def test(model,dataLoader):
    model.eval()
    with torch.no_grad():
        allLoss=[]
        allResult=[]
        allLabel=[]
        allPred=[]
        crossentry=nn.CrossEntropyLoss()
        for i,batch in enumerate(dataLoader):
            sents,aspects,labels,aspectIndex=batch
            sents=myDataset.to_input_tensor(sents).to(device)
            aspects=myDataset.to_input_tensor(aspects).to(device)
            labels=torch.LongTensor(labels).to(device)
            aspectIndex=torch.tensor(aspectIndex).to(device)
            pred=model(sents,aspects,aspectIndex)
            _,result=torch.max(pred,1)
            loss=crossentry(pred,labels)
            allLoss.append(loss.item())
            allResult.append(cuda2cpu(result))
            allLabel.append(cuda2cpu(labels))
            allPred.append(cuda2cpu(pred))
        allLabel=list(chain.from_iterable(allLabel))
        allResult=list(chain.from_iterable(allResult))
        allPred=list(chain.from_iterable(allPred))
        allLoss=np.mean(allLoss)
        accuracy,precision,recall,f1score,cm=estimate(allLabel,allResult,allPred)
        print('[INFO]TEST accuracy:{} F1:{} Loss:{}'.format(accuracy,f1score,allLoss))
        return accuracy,f1score


# In[27]:


def train():
    #model=TNet_LF()
    model=TNet_AS()
    model.apply(weight_init)
    model=model.to(device)
    #model=torch.load('model.pkl')
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999))
    trainLoader=DataLoader(myDataset(vocab.trainSent,vocab.trainAspect,vocab.trainY,vocab.trainAspectIndex),batch_size,shuffle=True,collate_fn=myDataset.collate_fn)
    testLoader=DataLoader(myDataset(vocab.testSent,vocab.testAspect,vocab.testY,vocab.testAspectIndex),batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
    crossentry=nn.CrossEntropyLoss()
    max_acc=0
    for i in range(epochs):
        model.train()
        allLoss=[]
        for j,batch in enumerate(trainLoader):
            optimizer.zero_grad()
            sents,aspects,labels,aspectIndex=batch
            sents=myDataset.to_input_tensor(sents).to(device)
            aspects=myDataset.to_input_tensor(aspects).to(device)
            labels=torch.LongTensor(labels).to(device)
            aspectIndex=torch.tensor(aspectIndex).to(device)
            pred=model(sents,aspects,aspectIndex)
            _,result=torch.max(pred,1)
            loss=crossentry(pred,labels)
            allLoss.append(loss.item())
            loss.backward()
            optimizer.step()
        allLoss=np.mean(allLoss)
        accuracy,f1score=test(model,testLoader)
        if accuracy>max_acc:
            max_acc=accuracy
            torch.save(model,'model.pkl')


# In[ ]:


train()


# In[ ]:


model=torch.load('model.pkl')
testLoader=DataLoader(myDataset(vocab.testSent,vocab.testAspect,vocab.testY,vocab.testAspectIndex),batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
test(model,testLoader)


# TNet-LF
# 
# laptop:72.4 65%
# 
# restaurant:ACC 79.9 F1:68.0
# 
# twitter ACC: F1:
# 
# TNet-AS
# 
# laptop ACC: F1:70.5 
# 
# restaurant ACC: F1:63.9
# 
# twitter ACC: F1:

# In[ ]:




