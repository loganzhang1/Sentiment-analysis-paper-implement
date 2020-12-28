#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import re
from os import listdir
from nltk.corpus import stopwords
from pickle import dump,load
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error
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
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from math import exp, log


# In[2]:


hidden_size=300
filter_num=100
filter_sizes=[3,4,5]
class_num=4
batch_size=32
epoch=40


# In[3]:


seed=2020
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
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
print('use device: %s' %device,file=sys.stderr)
#print("device_count :{}".format(torch.cuda.device_count()))


# In[5]:


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


# In[6]:


class Vocab:
    def __init__(self,trainName,testName):
        self.trainSent,self.trainAspect,self.trainY=self.readFile(trainName)
        self.testSent,self.testAspect,self.testY=self.readFile(testName)
        self.words=['pad']
        self.aspects=[]
        self.pad=0
        self.length=0
        self.word_vecs=None
        self.aspect_vecs=None
        self._word2id=None
        self._aspect2id=None
        self._sentiment2id={"positive":0,"neutral":1,"negative":2,"conflict":3}
        self.buildVocab()
        self.inGlove=[]
    
    def readFile(self,fileName):
        sentence=[]
        aspect=[]
        y=[]
        with open(fileName,'r',encoding='utf-8') as file:
            data=json.load(file)
            for i in data:
                sentence.append(clean_str(i['sentence']).split())
                aspect.append(clean_str(i['aspect']))
                y.append(i['sentiment'])
        return sentence,aspect,y
    
    def buildVocab(self):
        tempo=self.trainSent.copy()
        tempo.extend(self.testSent)
        for i in tempo:
            for j in i:
                if j not in self.words:
                    self.words.append(j)
        self.aspects=self.trainAspect.copy()
        self.aspects.extend(self.testAspect)
        self.aspects=list(set(self.aspects))
        self.aspect_vecs=np.random.uniform(-0.25,0.25,300*len(self.aspects)).reshape(len(self.aspects),300)
        reverse=lambda x:dict(zip(x,range(len(x))))
        self._word2id=reverse(self.words)
        self._aspect2id=reverse(self.aspects)
        self.length=len(self.words)
        self.word_vecs=np.random.uniform(-0.25,0.25,300*self.length).reshape(self.length,300)
        self.trainY=[self._sentiment2id[i] for i in self.trainY]
        self.testY=[self._sentiment2id[i] for i in self.testY]
    
    def words2indices(self,sents):
        return [[self._word2id[w] for w in s] for s in sents]
    
    def aspects2indices(self,aspects):
        return [self._aspect2id[s] for s in aspects]
    
    def load_pretrained_vec(self,fname):
        with open(fname) as fp:
            for line in fp.readlines():
                line=line.split(" ")
                word=line[0]
                if word in self.words:
                    self.inGlove.append(word)
                    self.word_vecs[self._word2id[word]]=np.array([float(x) for x in line[1:]])
    
    def pad_sents(self,sents):
        sents_padded=[]
        max_length=max([len(s) for s in sents])
        for i in sents:
            data=i
            data.extend([self.pad for _ in range(max_length-len(i))])
            sents_padded.append(data)
        return sents_padded
    
    def to_input_tensor(self,sents):
        wordIndices=self.words2indices(sents)
        wordpad=self.pad_sents(wordIndices)
        wordten=torch.tensor(wordpad,dtype=torch.long,device=device)
        return wordten


# In[7]:


file="rest_large"
train_file=None
test_file=None
if file=="rest14":
    train_file='../data/acsa-restaurant-2014/acsa_train.json'
    test_file='../data/acsa-restaurant-2014/acsa_test.json'
elif file=="rest14hard":
    train_file='../data/acsa-restaurant-2014/acsa_hard_train.json'
    test_file='../data/acsa-restaurant-2014/acsa_hard_test.json'
elif file=="rest_large":
    train_file='../data/acsa-restaurant-large/acsa_train.json'
    test_file='../data/acsa-restaurant-large/acsa_test.json'
elif file=="rest_large_hard":
    train_file='../data/acsa-restaurant-large/acsa_hard_train.json'
    test_file='../data/acsa-restaurant-large/acsa_hard_test.json'
else:
    print('[ERROR]file should be one of rest14/rest14hard/rest_large/rest_large_hard')


# In[8]:


'''vocab=Vocab(train_file,test_file)
vocab.load_pretrained_vec('../../data/glove.840B.300d.txt')
dump(vocab,open('vocab.pkl','wb'))'''


# In[9]:


vocab=load(open('vocab.pkl','rb'))


# In[10]:


class Model(nn.Module):
    def __init__(self,vocab,device):
        super(Model,self).__init__()
        self.device=device
        weight=torch.FloatTensor(vocab.word_vecs).to(device)
        self.wordEmbedding=nn.Embedding.from_pretrained(weight)
        self.wordEmbedding.requires_grad=True
        aspectWeight=torch.FloatTensor(vocab.aspect_vecs).to(device)
        self.aspectEmbedding=nn.Embedding.from_pretrained(aspectWeight)
        self.aspectEmbedding.requires_grad=True
        self.convolution1=nn.ModuleList([nn.Conv1d(hidden_size,filter_num,fs) for fs in filter_sizes])
        self.convolution2=nn.ModuleList([nn.Conv1d(hidden_size,filter_num,fs) for fs in filter_sizes])
        self.aspect_map=nn.Linear(hidden_size,filter_num)
        self.linear=nn.Linear(len(filter_sizes)*filter_num,class_num)
        
        
    def forward(self,source,aspect):
        #print('source.shape:',source.shape)
        #print('aspect.shape:',aspect.shape)
        source_embedding=self.wordEmbedding(source).permute(0,2,1)
        #print('source_embedding.shape:',source_embedding.shape)
        aspect_embedding=self.aspectEmbedding(aspect)
        #print('aspect_embedding.shape:',aspect_embedding.shape)
        conv1_result=[F.tanh(conv(source_embedding)) for conv in self.convolution1]
        conv2_result=[F.relu(conv(source_embedding) + self.aspect_map(aspect_embedding).unsqueeze(2)) for conv in self.convolution2]
        conv_result=[x*y for x,y in zip(conv1_result,conv2_result)]
        #print('conv_result[0].shape:',conv_result[0].shape)
        #print('F.max_pool1d(conv_result[0],conv_result[0].shape[2]).shape:',F.max_pool1d(conv_result[0],conv_result[0].shape[2]).shape)
        pool_result=[F.max_pool1d(i,i.shape[2]).squeeze(2) for i in conv_result]
        cat_result=torch.cat(pool_result,1)
        #print('cat_result.shape:',cat_result.shape)
        result=F.softmax(self.linear(cat_result))
        #print('result.shape:',result.shape)
        return result


# In[11]:


def batch_iter(data,shuffle=True):
    batch_num=math.ceil(len(data)/batch_size)
    index_array=list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)
    for i in range(batch_num):
        indices=index_array[i*batch_size:(i+1)*batch_size]
        example=[data[idx] for idx in indices]
        exaxple=sorted(example,key=lambda e:len(e[0]),reverse=True)
        sents=[e[0] for e in example]
        aspects=[e[1] for e in example]
        labels=[int(e[2]) for e in example]
        yield sents,aspects,labels


# In[12]:


def test(model,sents,aspects,labels):
    labels=torch.LongTensor(labels).to(device)
    aspects=vocab.aspects2indices(aspects)
    aspects=torch.LongTensor(aspects).to(device)
    sents=vocab.to_input_tensor(sents)
    with torch.no_grad():
        prob=model(sents,aspects)
        result=np.argmax(prob.cpu(),axis=1)
        acc=accuracy_score(labels.cpu(),result)
        return acc


# In[13]:


def train():
    model=Model(vocab,device)
    #model=nn.DataParallel(model)
    model=model.to(device)
    optimizer=torch.optim.Adagrad(model.parameters(),lr=1e-2)
    train_data=list(zip(vocab.trainSent,vocab.trainAspect,vocab.trainY))
    entrycross=nn.CrossEntropyLoss()
    for i in range(epoch):
        allLoss=0
        for sents,aspects,labels in batch_iter(train_data):
            labels=torch.LongTensor(labels).to(device)
            aspects=vocab.aspects2indices(aspects)
            aspects=torch.LongTensor(aspects).to(device)
            optimizer.zero_grad()
            sents=vocab.to_input_tensor(sents)
            result=model(sents,aspects).to(device)
            loss=entrycross(result,labels)
            allLoss+=loss
            loss.backward()
            optimizer.step()
        test_acc=test(model,vocab.testSent,vocab.testAspect,vocab.testY)
        print('[INFO]epoch:{} loss in train dataset:{} Test Acc:{}'.format(i,allLoss,test_acc))
        torch.save(model,'model.pkl')
        sys.stdout.flush()


# In[14]:


train()


# In[15]:


model=torch.load('model.pkl')
test_acc=test(model,vocab.testSent,vocab.testAspect,vocab.testY)
print('[INFO]Acc in test dataset is {}'.format(test_acc))


# restaurant-2014:0.798
# restaurant-2014-hard:0.5056
# restaurant-large:0.8462
# restaurant-large-hard:0.697

# In[ ]:




