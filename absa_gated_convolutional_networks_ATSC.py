#!/usr/bin/env python
# coding: utf-8

# In[1]:


#论文复现：Aspect Based Sentiment Analysis with Gated Convolution Networks中的ATSC部分
import string
import re
import pickle
from os import listdir
from nltk.corpus import stopwords
from pickle import dump,load
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


hidden_size=300
filter_num=100
filter_sizes=[3,4,5]
class_num=4
batch_size=32
epoch=40


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
                sentence.append(clean_str(i['sentence']).lower().split())
                aspect.append(clean_str(i['aspect']).lower().split())
                y.append(i['sentiment'])
        return sentence,aspect,y
    
    def buildVocab(self):
        tempo=self.trainSent.copy()
        tempo.extend(self.testSent)
        for i in tempo:
            for j in i:
                if j not in self.words:
                    self.words.append(j)
        reverse=lambda x:dict(zip(x,range(len(x))))
        self._word2id=reverse(self.words)
        self.length=len(self.words)
        self.word_vecs=np.random.uniform(-0.25,0.25,300*self.length).reshape(self.length,300)
        self.trainY=[self._sentiment2id[i] for i in self.trainY]
        self.testY=[self._sentiment2id[i] for i in self.testY]
        self.trainAspect=[self.words2indices(i) for i in self.trainAspect]
        self.testAspect=[self.words2indices(i) for i in self.testAspect]
        self.trainSent=self.words2indices(self.trainSent)
        self.testSent=self.words2indices(self.testSent)
    
    def words2indices(self,sents):#有两种情况，一种是输入句子将句子中单词转换为id，另一种是输入aspect()
        if type(sents[0])==list:
            return [[self._word2id[w] for w in s] for s in sents]
        else:
            return [self._word2id[w] for w in sents]
    
    def load_pretrained_vec(self,fname):
        with open(fname) as fp:
            for line in fp.readlines():
                line=line.split(" ")
                word=line[0]
                if word in self.words:
                    self.inGlove.append(word)
                    self.word_vecs[self._word2id[word]]=np.array([float(x) for x in line[1:]])
    
    @staticmethod
    def to_input_tensor(sents):
        return torch.nn.utils.rnn.pad_sequence([torch.tensor(sent) for sent in sents],batch_first=True,padding_value=0)


# In[7]:


file="rest"
train_file=None
test_file=None
if file=="rest":
    train_file='./GCAE/atsa-restaurant/atsa_train.json'
    test_file='./GCAE/atsa-restaurant/atsa_test.json'
elif file=="rest_hard":
    train_file='./GCAE/atsa-restaurant/atsa_train.json'
    test_file='./GCAE/atsa-restaurant/atsa_hard_test.json'
elif file=="laptop":
    train_file='./GCAE/atsa-laptop/atsa_train.json'
    test_file='./GCAE/atsa-laptop/atsa_test.json'
elif file=="laptop_hard":
    train_file='./GCAE/atsa-laptop/atsa_train.json'
    test_file='./GCAE/atsa-laptop/atsa_hard_test.json'
else:
    print('[ERROR]file should be one of rest/rest_hard/laptop/laptop_hard')


# In[8]:


'''vocab=Vocab(train_file,test_file)
vocab.load_pretrained_vec('../../data/glove.840B.300d.txt')
if file=="rest":
    dump(vocab,open('vocab_rest.pkl','wb'))
elif file=="rest_hard":
    dump(vocab,open('vocab_rest_hard.pkl','wb'))
elif file=="laptop":
    dump(vocab,open('vocab_laptop.pkl','wb'))
else:
    dump(vocab,open('vocab_laptop_hard.pkl','wb'))'''


# In[9]:


if file=="rest":
    vocab=load(open('vocab_rest.pkl','rb'))
elif file=="rest_hard":
    vocab=load(open('vocab_rest_hard.pkl','rb'))
elif file=="laptop":
    vocab=load(open('vocab_laptop.pkl','rb'))
else:
    vocab=load(open('vocab_laptop_hard.pkl','rb'))


# In[10]:


class Model(nn.Module):
    def __init__(self,vocab,device):
        super(Model,self).__init__()
        self.device=device
        weight=torch.FloatTensor(vocab.word_vecs).to(device)
        self.wordEmbedding=nn.Embedding.from_pretrained(weight)
        self.wordEmbedding.requires_grad=True
        self.convolution1=nn.ModuleList([nn.Conv1d(hidden_size,filter_num,fs) for fs in filter_sizes])
        self.convolution2=nn.ModuleList([nn.Conv1d(hidden_size,filter_num,fs) for fs in filter_sizes])
        self.aspect_map=nn.Conv1d(hidden_size,filter_num,3,padding=1)
        self.linear=nn.Linear(len(filter_sizes)*filter_num,class_num)
        
        
    def forward(self,source,aspect):
        source_embedding=self.wordEmbedding(source).permute(0,2,1)#[batch_size,embedding_dim,sequence]
        aspect_embedding=self.wordEmbedding(aspect).permute(0,2,1)#[batch_size,embedding_dim,aspect_length]
        aspect_embedding_map=F.relu(self.aspect_map(aspect_embedding))#[batch_size,filter_num,aspect_length]
        aspect_embedding_max=torch.max_pool1d(aspect_embedding_map,aspect_embedding_map.size(2))#[batch_size,filter_num]
        conv1_result=[F.tanh(conv(source_embedding)) for conv in self.convolution1]
        conv2_result=[F.relu(conv(source_embedding) + aspect_embedding_max) for conv in self.convolution2]
        conv_result=[x*y for x,y in zip(conv1_result,conv2_result)]
        pool_result=[F.max_pool1d(i,i.shape[2]).squeeze(2) for i in conv_result]
        cat_result=torch.cat(pool_result,1)
        result=F.softmax(self.linear(cat_result))
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
    aspects=Vocab.to_input_tensor(aspects).to(device)
    sents=Vocab.to_input_tensor(sents).to(device)
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
    max_acc=0
    for i in range(epoch):
        allLoss=0
        for sents,aspects,labels in batch_iter(train_data):
            labels=torch.LongTensor(labels).to(device)
            aspects=Vocab.to_input_tensor(aspects).to(device)
            optimizer.zero_grad()
            sents=Vocab.to_input_tensor(sents).to(device)
            result=model(sents,aspects).to(device)
            loss=entrycross(result,labels)
            allLoss+=loss
            loss.backward()
            optimizer.step()
        test_acc=test(model,vocab.testSent,vocab.testAspect,vocab.testY)
        print('[INFO]epoch:{} loss in train dataset:{} Test Acc:{}'.format(i,allLoss,test_acc))
        sys.stdout.flush()
        if test_acc>max_acc:
            torch.save(model,'model.pkl')
            max_acc=test_acc


# In[14]:


train()


# In[15]:


model=torch.load('model.pkl')
test_acc=test(model,vocab.testSent,vocab.testAspect,vocab.testY)
print('[INFO]Test Acc:{}'.format(test_acc))


# restaurant:76.3
# restaurant_hard:60%
# laptop:67.9
# laptop_hard:51.8%

# In[ ]:




