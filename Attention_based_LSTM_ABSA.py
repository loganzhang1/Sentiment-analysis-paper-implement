#!/usr/bin/env python
# coding: utf-8

# In[1]:


#通过关联应用Colaborate新建丘比特笔记本文件
#首先要设置丘比特笔记本，设置为使用GPU。点击菜单栏 修改->笔记本设置->选择硬件加速器为GPU
'''from google.colab import drive  #挂载到谷歌云盘上
drive.mount('/content/drive/')'''


# In[2]:


'''import os
os.chdir("/content/drive/My Drive/SC/Attention_Based_ABSA")#这个地方就已经挂载到了你的谷歌云盘上，之后ls打印当前目录下的文件
!ls'''


# In[3]:


import string
import re
from os import listdir
from nltk.corpus import stopwords
from pickle import dump,load
import numpy as np
import torch.nn as nn
import torch
import sys
import math
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


# In[4]:


class Vocab:
    def __init__(self,trainName,testName,devName):
        self.trainSent,self.trainAspect,self.trainY=self.readFile(trainName)
        self.testSent,self.testAspect,self.testY=self.readFile(testName)
        self.devSent,self.devAspect,self.devY=self.readFile(devName)
        self.words=['pad']
        self.aspects=[]
        self.pad=0
        self.length=0
        self.word_vecs=None
        self.aspect_vecs=None
        self._word2id=None
        self._aspects2id=None
        self.buildVocab()
        self.inGlove=[]
    def readFile(self,fileName):
        sentence=[]
        aspect=[]
        y=[]
        fp=open(fileName)
        line=fp.readline().strip()
        while line:
            sentence.append(line.split())
            aspect.append(fp.readline().strip())
            y.append(int(fp.readline().strip()))
            line=fp.readline().strip()
        return sentence,aspect,y
    
    def buildVocab(self):
        tempo=self.trainSent.copy()
        tempo.extend(self.testSent)
        tempo.extend(self.devSent)
        for i in tempo:
            for j in i:
                if j not in self.words:
                    self.words.append(j)
        self.aspects=self.trainAspect.copy()
        self.aspects.extend(self.testAspect)
        self.aspects.extend(self.devAspect)
        self.aspects=list(set(self.aspects))
        self.aspect_vecs=np.random.uniform(-0.25,0.25,300*len(self.aspects)).reshape(len(self.aspects),300)
        reverse=lambda x:dict(zip(x,range(len(x))))
        self._word2id=reverse(self.words)
        self._aspects2id=reverse(self.aspects)
        self.length=len(self.words)
        self.word_vecs=np.zeros((self.length,300))
    
    def words2indices(self,sents):
        return [[self._word2id[w] for w in s] for s in sents]
    
    def aspects2indices(self,aspects):
        return [self._aspects2id[s]  for s in aspects]
    
    def load_pretrained_vec(self,fname):
        with open(fname) as fp:
            for line in fp.readlines():
                line=line.split(" ")
                word=line[0]
                if word in self.words:
                    self.inGlove.append(word)
                    self.word_vecs[self._word2id[word]]=np.array([float(x) for x in line[1:]])
            self.add_unknown_words()
    
    def pad_sents(sef,sents,pad_token):
        sents_padded=[]
        max_length=max([len(s) for s in sents])
        for i in sents:
            data=i
            data.extend([pad_token for _ in range(max_length-len(i))])
            sents_padded.append(data)
        return sents_padded
    
    def to_input_tensor(self,sents,device):
        wordIndices=self.words2indices(sents)
        wordpad=self.pad_sents(wordIndices,self._word2id['pad'])
        wordten=torch.tensor(wordpad,dtype=torch.long,device=device)
        return wordten
    
    def add_unknown_words(self):
        for word in self.words:
            if word not in self.inGlove:
                self.word_vecs[self._word2id[word]]=np.random.uniform(-0.25,0.25,300)


# In[5]:


'''vocab=Vocab('./data/train.cor','./data/test.cor','./data/dev.cor')
vocab.load_pretrained_vec('../../data/glove.6B.300d.txt')
dump(vocab,open('vocab.pkl','wb'))'''


# In[6]:


vocab=load(open('vocab.pkl','rb'))


# In[7]:


class Model(nn.Module):
    def __init__(self,vocab,device):
        super(Model,self).__init__()
        self.device=device
        self.vocab=vocab
        weight=torch.FloatTensor(vocab.word_vecs).to(device)
        self.wordEmbedding=nn.Embedding.from_pretrained(weight)
        self.wordEmbedding.requires_grad=True
        #print('self.wordEmbedding.device:',self.wordEmbedding.device)
        aspectWeight=torch.FloatTensor(vocab.aspect_vecs).to(device)
        self.aspectEmbedding=nn.Embedding.from_pretrained(aspectWeight)
        self.aspectEmbedding.requires_grad=True
        self.lstm=nn.LSTM(input_size=600,hidden_size=300)
        self.hidden_projection=nn.Linear(in_features=300,out_features=300,bias=False)
        self.aspect_projection=nn.Linear(in_features=300,out_features=300,bias=False)
        self.w_projection=nn.Linear(in_features=600,out_features=1,bias=False)
        self.Wp_projection=nn.Linear(in_features=300,out_features=300,bias=False)
        self.Wx_projection=nn.Linear(in_features=300,out_features=300,bias=False)
        self.ws_projection=nn.Linear(in_features=300,out_features=3)
        self.softmax=nn.Softmax(dim=1)
        
    def forward(self,source,aspect):
        source_padded=self.vocab.to_input_tensor(source,self.device)#[10, 38]
        #print('source_padded.shape:',source_padded.shape)
        source_embedding=self.wordEmbedding(source_padded)#[10, 38, 300]
        #print('source_embedding.shape:',source_embedding.shape)
        single_aspect_embedding=self.aspectEmbedding(aspect)#[10, 300]
        #print('single_aspect_embedding.shape:',single_aspect_embedding.shape)
        aspect_embedding=self.aspectEmbedding(aspect).unsqueeze(1).repeat(1,source_embedding.shape[1],1)
        #print('aspect_embedding.shape:',aspect_embedding.shape)
        source_aspect_embedding=torch.cat((source_embedding,aspect_embedding),2).permute(1,0,2)#[38, 10, 600]
        #print('source_aspect_embedding.shape:',source_aspect_embedding.shape)
        output,(hn,cn)=self.lstm(source_aspect_embedding)
        #print('output.shape:',output.shape)
        #print('hn.shape:',hn.shape)
        #print('cn.shape:',cn.shape)
        output=output.permute(1,0,2)#[10, 38, 300]
        #print('output.shape:',output.shape)
        
        Wh_H=self.hidden_projection(output)#[10, 38, 300]
        #print('Wh_H.shape:',Wh_H.shape)
        Wv_va=self.aspect_projection(single_aspect_embedding).unsqueeze(1).repeat(1,Wh_H.shape[1],1)#[10, 38, 300]
        #print('Wv_va.shape:',Wv_va.shape)
        M=torch.tanh(torch.cat((Wh_H,Wv_va),2))#[10, 38, 600]
        #print('M.shape:',M.shape)
        alpha=F.softmax(self.w_projection(M),dim=1).permute(0,2,1)#[10, 1, 38]
        #print('alpha.shape:',alpha.shape)
        output=output.permute(0,1,2)#[10, 38, 300]
        #print('output.shape:',output.shape)
        r=torch.matmul(alpha,output).squeeze(1)#[10, 300]
        #print('r.shape:',r.shape)
        hn=hn.squeeze(0)
        h=torch.tanh(torch.add(self.Wp_projection(r),self.Wx_projection(hn)))#[10, 300]
        #print('h.shape:',h.shape)
        e=self.ws_projection(h)#[10, 3]
        #print('e.shape:',e.shape)
        y=self.softmax(e)#[10, 3]
        return y


# In[8]:


def batch_iter(data,batch_size,shuffle=True):
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


# In[ ]:


from torch.nn import init
def weight_init(m):
    if isinstance(m,nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m,nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m,nn.Linear):
        m.weight.data.normal_(0,0.01)
        #m.bias.data.zero_()


# In[ ]:


epoch=50
batch_size=10
def train():
    device=torch.device("cpu")
    #device=torch.device("cuda:0")
    print('use device: %s' %device,file=sys.stderr)
    model=Model(vocab,device)
    model.apply(weight_init)
    '''
    optimizer=torch.optim.Adagrad(model.parameters(),lr=0.01,weight_decay=0.001)#使用Adagrad无法在GPU上训练，在pytorch官网上找到了下面的话，可能是因为这个
    Keep in mind that only a limited number of optimizers support sparse gradients: currently it’s optim.SGD (CUDA and CPU),
    optim.SparseAdam (CUDA and CPU) and optim.Adagrad (CPU)
    '''
    optimizer=torch.optim.Adagrad(model.parameters(),lr=0.01,weight_decay=0.001)
    train_data=list(zip(vocab.trainSent,vocab.trainAspect,vocab.trainY))
    test_data=list(zip(vocab.testSent,vocab.testAspect,vocab.testY))
    model=model.to(device)
    for i in range(epoch):
        allLoss=0
        for sents,aspects,labels in batch_iter(train_data,batch_size):
            labels=torch.LongTensor(labels).to(device)
            #print('original labels:',labels)
            labels=labels+1
            #print('labels:',labels)
            aspects=vocab.aspects2indices(aspects)
            #print('aspects:',aspects)
            aspects=torch.LongTensor(aspects).to(device)
            optimizer.zero_grad()
            result=model(sents,aspects).to(device)
            entrycross=nn.CrossEntropyLoss()
            loss=entrycross(result,labels)
            #print('labels:',labels)
            #print('result:',result)
            #print('loss:',loss)
            allLoss+=loss
            loss.backward()
            optimizer.step()
        print('[INFO] epoch:{} loss: {}'.format(i,allLoss))
    torch.save(model,'model.pkl')


# In[ ]:


train()


# In[ ]:


def test(sents,aspects,labels):
    model=torch.load('model.pkl')
    #device=torch.device("cuda:0")
    device=torch.device("cpu")
    print('use device: %s' %device,file=sys.stderr)
    model=model.to(device)
    labels=torch.LongTensor(labels).to(device)
    aspects=vocab.aspects2indices(aspects)
    aspects=torch.LongTensor(aspects).to(device)
    with torch.no_grad():
        labels=labels+1
        prob=model(sents,aspects)
        result=np.argmax(prob.cpu(),axis=1)
        print('accuracy:{}'.format(accuracy_score(labels.cpu(),result)))


# In[ ]:


test(vocab.testSent,vocab.testAspect,vocab.testY)


# In[ ]:


test(vocab.devSent,vocab.devAspect,vocab.devY)


# In[ ]:




