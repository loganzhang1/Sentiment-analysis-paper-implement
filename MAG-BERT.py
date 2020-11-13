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
import os
import random
import torch
import sys
import math
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import math
from torch.optim.optimizer import Optimizer, required
from transformers import BertModel,get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# In[2]:


device=torch.device("cuda:1")
print('use device: %s' %device,file=sys.stderr)


# In[3]:


alldata=load(open('../data/mosi.pkl','rb'))


# In[4]:


train_data,valid_data,test_data=alldata[0],alldata[1],alldata[2]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[5]:


VIS_DIM=47
TEXT_DIM=768
ACO_DIM=74
beta=1
hidden_size=768


# In[6]:


seed=2020
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#torch.backends.cudnn.deterministic = True


# In[7]:


def _init_weights(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


# In[8]:


class MAG(nn.Module):
    def __init__(self,beta,hidden_size,dropout,device):
        super().__init__()
        self.device=device
        self.Wgv=nn.Linear(TEXT_DIM+VIS_DIM,TEXT_DIM)
        self.Wga=nn.Linear(TEXT_DIM+ACO_DIM,TEXT_DIM)
        self.Wa=nn.Linear(ACO_DIM,TEXT_DIM)
        self.Wv=nn.Linear(VIS_DIM,TEXT_DIM)
        self.beta=beta
        self.dropout=nn.Dropout(dropout)
        self.layernorm=nn.LayerNorm(hidden_size)
    def forward(self,text,vis,aco):
        #text:[15, 43]
        #vis:[15, 43, 47]
        #aco:[15, 43, 74]
        giv=F.relu(self.Wgv(torch.cat((text,vis),-1)))
        gia=F.relu(self.Wga(torch.cat((text,aco),-1)))
        Hi=gia*self.Wa(aco)+giv*self.Wv(vis)
        alpha=torch.min(torch.Tensor([1]*Hi.shape[0]).to(device),self.beta*(torch.norm(text,p=2,dim=(1,2))/torch.norm(Hi,p=2,dim=(1,2))))
        alpha=alpha.unsqueeze(-1).unsqueeze(-1)
        Zi_=text+alpha*Hi
        result=self.dropout(self.layernorm(Zi_))
        return result


# In[9]:


def dataProcess(data):#对文本数据执行BPE算法之后对两个维度进行对齐
    text_withoutCLS=[e[0][0] for e in data]
    vis=[e[0][1] for e in data]
    aco=[e[0][2] for e in data]
    score=[e[1][0][0] for e in data]
    attention_mask=[]
    token_type_ids=[]
    texts=[]
    for i in text_withoutCLS:
        j=['[CLS]']
        j.extend(i)
        j.append('[SEP]')
        texts.append(j)
    token_texts=[]
    token_vis=[]
    token_aco=[]
    vis_zero=[0 for i in range(VIS_DIM)]
    aco_zero=[0 for i in range(ACO_DIM)]
    
    for i in range(len(texts)):
        temp_texts=[]
        temp_texts.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(texts[i][0]))[0])
        temp_vis=[]
        temp_aco=[]
        texts_without_CLS=texts[i][1:-1]
        for j in range(len(texts_without_CLS)):
            tokenized_text=tokenizer.tokenize(texts_without_CLS[j])
            token_id=tokenizer.convert_tokens_to_ids(tokenized_text)
            temp_texts.extend(token_id)
            temp_vis.extend([vis[i][j] for k in range(len(tokenized_text))])
            temp_aco.extend([aco[i][j] for k in range(len(tokenized_text))])
        temp_texts.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(texts[i][-1]))[0])
        token_texts.append(temp_texts)
        each_vis=[]
        each_vis.append(vis_zero)
        each_vis.extend(temp_vis)
        each_vis.append(vis_zero)
        token_vis.append(each_vis)
        each_aco=[]
        each_aco.append(aco_zero)
        each_aco.extend(temp_aco)
        each_aco.append(aco_zero)
        token_aco.append(each_aco)
        attention_mask.append([1]*len(temp_texts))
        token_type_ids.append([0]*len(temp_texts))
    
    #下面将数据补齐成最大的长度
    max_len=max([len(text) for text in token_texts])
    for i in range(len(token_texts)):
        if len(token_texts[i])<max_len:
            padding=[0]*(max_len-len(token_texts[i]))
            token_texts[i].extend(padding)
            attention_mask[i].extend(padding)
            token_type_ids[i].extend(padding)
            token_vis[i].extend([vis_zero for j in range(max_len-len(token_vis[i]))])
            token_aco[i].extend([aco_zero for j in range(max_len-len(token_aco[i]))])
    tensor_texts=torch.LongTensor(token_texts)
    tensor_vis=torch.tensor(token_vis)
    tensor_aco=torch.tensor(token_aco)
    tensor_attention_mask=torch.LongTensor(attention_mask)
    tensor_type=torch.LongTensor(token_type_ids)
    tensor_score=torch.tensor(score)
    return tensor_texts,tensor_vis,tensor_aco,tensor_attention_mask,tensor_type,tensor_score


# In[10]:


class MAG_bertModel(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.config=config
        self.embeddings=BertEmbeddings(config)
        # transformer blocks * N
        self.encoder=BertEncoder(config)
        self.pooler=BertPooler(config)
        self.MAG=MAG(beta=1.0,hidden_size=hidden_size,dropout=0.5,device=device)
        self.MAG.apply(_init_weights)
        self.init_weights()
        
    #texts,vis,aco,attention_mask,token_type,position_enc
    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
            ):
        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)
        
        texts_embeddings=self.embeddings(input_ids=input_ids,
                             position_ids=position_ids,
                             token_type_ids=token_type_ids)
        magData=self.MAG(texts_embeddings,visual,acoustic)
        bert_out = self.encoder(
            magData,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = bert_out[0]
        pooled_output = self.pooler(sequence_output)
        return pooled_output


# In[11]:


class MAG_bert_classification(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.num_labels=config.num_labels
        self.bert=MAG_bertModel(config)
        self.classifier=nn.Linear(hidden_size,config.num_labels)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
        
    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        ):
        outputs=self.bert(
            input_ids=input_ids,
            visual=visual,
            acoustic=acoustic,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        pooled_output=self.dropout(outputs)
        pred=self.classifier(pooled_output)
        return pred


# In[12]:


def metrics(y_true,y_pred):
    y_pred=y_pred.cpu()
    y_pred_bin=[1 if i>=0 else 0 for i in y_pred]
    y_true_bin=[1 if i >=0 else 0 for i in y_true]
    
    bi_acc=accuracy_score(y_true_bin,y_pred_bin)
    f1=f1_score(y_true_bin,y_pred_bin)
    mae=mean_absolute_error(y_true_bin,y_pred_bin)
    corr=np.corrcoef(y_pred.cpu(),y_true)[0][1]
    return bi_acc,f1,mae,corr


# In[13]:


def test_model(model,data):
    #print(data)
    with torch.no_grad():
        texts,vis,aco,attention_mask,token_type,label=dataProcess(data)
        #print('label:',label)
        dataTensorDataset=TensorDataset(texts,vis,aco,attention_mask,token_type,label)
        test_loader=DataLoader(dataTensorDataset,batch_size=128,shuffle=False,num_workers=1)
        result=torch.Tensor().to(device)
        #print('result:',result)
        for i,data in enumerate(test_loader):
            texts,vis,aco,attention_mask,token_type,_=data
            texts=texts.to(device)
            vis=vis.to(device)
            aco=aco.to(device)
            attention_mask=attention_mask.to(device)
            token_type=token_type.to(device)
            pred=model(input_ids=texts,visual=vis,acoustic=aco,attention_mask=attention_mask,token_type_ids=token_type)
            result=torch.cat((result,pred),dim=0)
        result=result.squeeze(-1)
    return metrics(label,result)


# In[14]:


epoch=40
batch_size=48
def train():
    model=MAG_bert_classification.from_pretrained("../data/bert-based-uncase/",num_labels=1)
    tensor_texts,tensor_vis,tensor_aco,tensor_attention_mask,tensor_type,tensor_score=dataProcess(train_data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model=model.to(device)
    dataTensorDataset=TensorDataset(tensor_texts,tensor_vis,tensor_aco,tensor_attention_mask,tensor_type,tensor_score)
    train_loader=DataLoader(dataTensorDataset,batch_size=batch_size,shuffle=True,num_workers=1)
    scheduler=get_linear_schedule_with_warmup(optimizer,)
    for i in range(epoch):
        allLoss=0
        for j,data in enumerate(train_loader):
            optimizer.zero_grad()
            texts,vis,aco,attention_mask,token_type,label=data
            texts=texts.to(device)
            vis=vis.to(device)
            aco=aco.to(device)
            attention_mask=attention_mask.to(device)
            token_type=token_type.to(device)
            label=label.to(device)
            pred=model(texts,vis,aco,attention_mask,token_type).squeeze()
            #print('pred:',pred)
            #print('label:',label)
            entrycross=nn.MSELoss(reduce=True, size_average=True)
            loss=entrycross(pred,label)
            #print('loss:',loss)
            loss.backward()
            allLoss+=loss
            optimizer.step()
        with torch.no_grad():
            bi_acc,f1,mae,corr=test_model(model,valid_data)
            print('[INFO] epoch {} Loss: {}'.format(i,allLoss))
            print("[INFO] epoch {}: ACC:{} f1:{} mae:{} Corr:{}".format(i,bi_acc,f1,mae,corr))
    torch.save(model,'model.pkl')


# In[15]:


train()


# In[15]:


model=torch.load('model.pkl')
bi_acc,f1,mae,corr=test_model(model,test_data)
print("[INFO] Test result: ACC:{} f1:{} mae:{} Corr:{}".format(bi_acc,f1,mae,corr))


# In[ ]:





# In[ ]:




