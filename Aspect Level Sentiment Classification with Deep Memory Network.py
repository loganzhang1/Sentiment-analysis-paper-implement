#!/usr/bin/env python
# coding: utf-8

# In[1]:


#论文复现：Aspect Level Sentiment Classification with Deep Memory Network
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
batch_size=32
class_num=3
epoch=30
location=4
hops=2
addLocation=False


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


def words2indices(self,sents):
    return [[self._word2id[w] for w in s] for s in sents]


# In[7]:


class Vocab:
    def __init__(self,trainName,testName,hasConflict):
        #由于这篇论文中的context不包括aspect所以这里需要从sentence中将aspect去掉
        self.trainSent,self.trainAspect,self.trainY=self.readFile(trainName,hasConflict)
        self.testSent,self.testAspect,self.testY=self.readFile(testName,hasConflict)
        self.words=['pad']
        self.aspects=[]
        self.pad=0
        self.length=0
        self.word_vecs=None
        self._word2id=None
        self._sentiment2id={"positive":0,"neutral":1,"negative":2}
        self.buildVocab()
        self.inGlove=[]
    
    def readFile(self,fileName,hasConflict):
        sentence=[]
        aspect=[]
        y=[]
        with open(fileName,'r',encoding='utf-8') as file:
            data=json.load(file)
            for i in data:
                if hasConflict==True or (hasConflict==False and i['sentiment'] != 'conflict'):
                    sentence.append(clean_str(i['sentence']).split())
                    aspect.append(clean_str(i['aspect']).split())
                    y.append(i['sentiment'])
                else:
                    continue
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
        self.word_vecs=np.random.uniform(-0.01,0.01,300*self.length).reshape(self.length,300)
        self.trainY=[self._sentiment2id[i] for i in self.trainY]
        self.testY=[self._sentiment2id[i] for i in self.testY]
        self.trainAspect=[self.words2indices(i) for i in self.trainAspect]
        self.testAspect=[self.words2indices(i) for i in self.testAspect]
        self.trainSent=self.removeAspect(self.words2indices(self.trainSent),self.trainAspect)
        self.testSent=self.removeAspect(self.words2indices(self.testSent),self.testAspect)
    
    def removeAspect(self,sents,aspect):
        #这个函数的正确性明天再确定
        result_sents=[]
        for i in range(len(sents)):
            for j in range(len(sents[i])):
                if sents[i][j]==aspect[i][0] and sents[i][j+len(aspect[i])-1]==aspect[i][-1]:
                    result_sents.append(sents[i][0:j]+sents[i][j+len(aspect[i]):])
                    break
        return result_sents
    
    def words2indices(self,sents):
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


# In[8]:


file="restaurant"
train_file=None
test_file=None
if file=="restaurant":
    train_file='../data/atsa-restaurant/atsa_train.json'
    test_file='../data/atsa-restaurant/atsa_test.json'
elif file=="laptop":
    train_file='../data/atsa-laptop/atsa_train.json'
    test_file='../data/atsa-laptop/atsa_test.json'
else:
    print('[ERROR]file should be one of restaurant/laptop')


# In[9]:


'''vocab=Vocab(train_file,test_file,hasConflict=False)
vocab.load_pretrained_vec('../../data/glove.840B.300d.txt')
if file=="restaurant":
    dump(vocab,open('vocab_restaurant.pkl','wb'))
else:
    dump(vocab,open('vocab_laptop.pkl','wb'))'''


# In[10]:


if file=="restaurant":
    vocab=load(open('vocab_restaurant.pkl','rb'))
else:
    vocab=load(open('vocab_laptop.pkl','rb'))


# In[11]:


class myDataset(Dataset):
    def __init__(self,sent,aspect,y):
        self.sent=sent
        self.aspect=aspect
        self.y=y
        self.len=len(sent)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        return self.sent[index],self.aspect[index],self.y[index]
    
    @staticmethod
    def to_input_tensor(sents):
        return torch.nn.utils.rnn.pad_sequence([torch.tensor(input_ids) for input_ids in sents],batch_first=True,padding_value=vocab.pad)
    
    @staticmethod
    def collate_fn(batch):
        sent,aspect,y=zip(*batch)
        return sent,aspect,y


# In[12]:


class DeepMemnet(nn.Module):
    def __init__(self,vocab):
        super(DeepMemnet,self).__init__()
        self.vocab=vocab
        weight=torch.FloatTensor(vocab.word_vecs).to(device)
        self.wordEmbedding=nn.Embedding.from_pretrained(weight)
        self.wordEmbedding.requires_grad=True
        self.linear=nn.Linear(hidden_size,hidden_size)
        self.attention_layer=nn.Linear(hidden_size*2,1)
        self.predict=nn.Linear(hidden_size,class_num)
        self.location_embedding=nn.Embedding(512,hidden_size)
    
    def forward(self,sents,aspect,sequence_length):
        #sents:[batch_size,sequence_lengths]
        #aspects:[[aspect,,,,]]list中的每个元素不定长，需要下面求一个平均值
        location_id=torch.LongTensor([list(range(max(sequence_length))) for i in range(len(sequence_length))]).to(device)
        sents_embedding=self.wordEmbedding(sents).to(sents.device)#(batch_size,sequence,hidden_size)
        #记录下来aspect的长度，之后在计算aspect向量的平均值的时候除一下
        aspect_length=[len(aspe) for aspe in aspect]
        #下面函数是补齐成一样长度之后转换为tensor
        aspect_tensor=myDataset.to_input_tensor(aspect).to(sents.device)#[batch_size,max_length]
        #获得每个单词的词向量
        aspect_embedding=self.wordEmbedding(aspect_tensor)#[batch_size,max_length,hidden_size]
        #embedding中存储batch个列表,每个列表中存储对应的要求平均的词向量
        aspect_embedding_list=[]
        for i in range(aspect_embedding.shape[0]):
            #对于第i个句子，取出其前aspect_length[i]个词向量
            aspect_embedding_list.append([aspect_embedding[i][j] for j in range(aspect_length[i])])
        #遍历上面的embedding，对每个list中的词向量求平均值，之后再合并起来
        aspect_embedding_average=torch.stack([torch.mean(torch.stack(i),0) for i in aspect_embedding_list])#[batch_size,hidden_size]
        #最终embedding的维度是[batch_size,hidden_size]
        embedding_sum=aspect_embedding_average
        for i in range(1,hops+1):
            aspect_embedding_average=embedding_sum
            aspect_embedding_linear=self.linear(embedding_sum)
            if addLocation==True:
                if location==3:
                    v=self.location_embedding(location_id)
                    sents_embedding=torch.add(sents_embedding,v)
                else:
                    v=torch.sigmoid(self.location_embedding(location_id))
                    sents_embedding=sents_embedding*v
            aspect_embedding_average=aspect_embedding_average.unsqueeze(1).repeat(1,sents.shape[1],1)#[batch_size,squence_length,hidden_size]
            sent_embedding_aspect_embedding=torch.cat((sents_embedding,aspect_embedding_average),2)#[batch_size,squence_length,hidden_size*2]
            g=torch.tanh(self.attention_layer(sent_embedding_aspect_embedding)).squeeze(2)#[batch_size,squence_length]
            g=g.masked_fill(sents==0,-float('inf'))#[batch_size,sequence_length]
            atten=torch.softmax(g,1).unsqueeze(1)#[batch_size,1,sequence_length]
            context_embedding=torch.bmm(atten,sents_embedding).squeeze(1)#[batch_size,hidden_size]
            embedding_sum=torch.add(context_embedding,aspect_embedding_linear)#[batch_size,hidden_size]
        return self.predict(embedding_sum)


# In[13]:


def cuda2cpu(pred):
    #cuda变量转cpu变量
    if type(pred)==list:
        return pred
    if pred.is_cuda:
        pred_cpu=list(pred.cpu().numpy())
    else:
        pred_cpu=list(pred.numpy())
    return pred_cpu

def estimate(labels,preds):
    labels=cuda2cpu(labels)
    preds=cuda2cpu(preds)
    accuracy=accuracy_score(labels,preds)
    precision=precision_score(labels,preds,average='macro')
    recall=recall_score(labels,preds,average='macro')
    f1score=f1_score(labels,preds,average='macro')
    cm=confusion_matrix(labels,preds)
    return accuracy,precision,recall,f1score,cm


# In[14]:


def evaluate(model,dataLoader):
    model.eval()
    with torch.no_grad():
        allloss=[]
        allresult=[]
        alllabel=[]
        crossentry=nn.CrossEntropyLoss()
        for i,batch in enumerate(dataLoader):
            sents,aspect,label=batch
            sequence_length=[len(sent) for sent in sents]
            label=torch.LongTensor(label).to(device)
            sents=myDataset.to_input_tensor(sents).to(device)
            pred=model(sents,aspect,sequence_length)
            _,result=torch.max(pred,1)
            loss=crossentry(pred,label)
            allloss.append(loss.item())
            alllabel.append(cuda2cpu(label))
            allresult.append(cuda2cpu(result))
        alllabel=list(chain.from_iterable(alllabel))
        allresult=list(chain.from_iterable(allresult))
        accuracy,precision,recall,f1score,_=estimate(alllabel,allresult)
        return accuracy,precision,recall,f1score


# In[15]:


def train():
    model=DeepMemnet(vocab)
    #model=nn.DataParallel(model)
    model=model.to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
    trainDataset=myDataset(vocab.trainSent,vocab.trainAspect,vocab.trainY)
    trainLoader=DataLoader(trainDataset,batch_size,shuffle=True,collate_fn=myDataset.collate_fn)
    crossentry=nn.CrossEntropyLoss()
    max_accuracy=0
    for i in range(epoch):
        model.train()
        allloss=[]
        alllabel=[]
        allresult=[]
        for j,batch in enumerate(trainLoader):
            optimizer.zero_grad()
            sents,aspect,label=batch
            label=torch.LongTensor(label).to(device)
            sequence_length=[len(sent) for sent in sents]
            sents=myDataset.to_input_tensor(sents).to(device)
            pred=model(sents,aspect,sequence_length)
            loss=crossentry(pred,label)
            _,result=torch.max(pred,1)
            allresult.append(cuda2cpu(result))
            allloss.append(loss.item())
            alllabel.append(cuda2cpu(label))
            loss.backward()
            optimizer.step()
        alllabel=list(chain.from_iterable(alllabel))
        allresult=list(chain.from_iterable(allresult))
        #train_accuracy,train_precision,train_recall,train_f1score,_=estimate(alllabel,allresult)
        #print("train_accuracy:{} train_precision:{} train_recall:{} train_f1score:{}".format(train_accuracy,train_precision,train_recall,train_f1score))
        testDataset=myDataset(vocab.testSent,vocab.testAspect,vocab.testY)
        testLoader=DataLoader(testDataset,batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
        test_accuracy,test_precision,test_recall,test_f1score=evaluate(model,testLoader)
        print("Epoch:{} test_accuracy:{} test_precision:{} test_recall:{} test_f1score:{}".format(i,test_accuracy,test_precision,test_recall,test_f1score))
        if test_accuracy>max_accuracy:
            torch.save(model,'model_location'+str(location)+file+'.pkl')
            max_accuracy=test_accuracy


# In[ ]:


train()


# In[ ]:


model=torch.load('model_location'+str(location)+file+'.pkl')
testDataset=myDataset(vocab.testSent,vocab.testAspect,vocab.testY)
testLoader=DataLoader(testDataset,batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
test_accuracy,test_precision,test_recall,test_f1score=evaluate(model,testLoader)
print("test_accuracy:{} test_precision:{} test_recall:{} test_f1score:{}".format(test_accuracy,test_precision,test_recall,test_f1score))


# restaurant:79.4
# laptop:72.1
# 
# 不加location编码竟然效果差距这么大

# In[ ]:





# In[ ]:




