import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class GraphConvolution(nn.Module):
    def __init__(self,input_dim,output_dim,use_bias=True):
        super(GraphConvolution,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.use_bias=use_bias
        self.weight=nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.use_bias:
            self.bias=nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_normal_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
            
    def forward(self,adjacency,input_feature):
        #adjacency.shape: [batch_size,sequence_length,sequence_length]
        #input_feature.shape: [batch_size,sequence_length,2*dim_h]
        #self.weight.shape: [2*dim_h,2*dim_h]
        support=torch.matmul(input_feature,self.weight)#[batch_size,sequence_length,2*dim_h]
        d=torch.sum(adjacency,2, keepdim=True)+1#[batch_size,sequence_length,1]
        output=torch.bmm(adjacency.float(),support.float())/d.float()#[batch_size,sequence_length,2*dim_h]
        if self.use_bias:
            output=output+self.bias
        return output

class ASGCN_DG(nn.Module):
    def __init__(self,vocab,opt):
        super(ASGCN_DG,self).__init__()
        self.vocab=vocab
        self.embedding=nn.Embedding.from_pretrained(torch.FloatTensor(vocab.word_vecs))
        self.bilstm=nn.LSTM(opt.dim_w,opt.dim_h,batch_first=True,bidirectional=True)
        self.GCN1=GraphConvolution(2*opt.dim_h,2*opt.dim_h)
        self.GCN2=GraphConvolution(2*opt.dim_h,2*opt.dim_h)
        self.predict=nn.Linear(2*opt.dim_h,opt.class_num)
        self.opt=opt
        self.dropout=nn.Dropout(0.3)
    
    def position(self,data,aspect_index,aspect_len,sents_len):
        batch_size,sequence_len=data.shape[0],data.shape[1]
        position=[[] for i in range(batch_size)]
        #aspect_index=3代表从第4个 aspect_len=2
        #1 2 3 0 0 
        for batch in range(batch_size):
            for i in range(1,aspect_index[batch]+1):
                position[batch].append(1-(aspect_index[batch]+1-i)/sents_len[batch])
            for i in range(aspect_index[batch]+1,aspect_index[batch]+aspect_len[batch]+1):
                position[batch].append(0)
            for i in range(aspect_index[batch]+aspect_len[batch]+1,sents_len[batch]+1):
                position[batch].append(1-(i-aspect_index[batch]-aspect_len[batch])/sents_len[batch])
            for i in range(sents_len[batch]+1,sequence_len+1):
                position[batch].append(0)
        position=torch.FloatTensor(position).unsqueeze(2).to(self.opt.device)
        data=data*position
        return data
    

    def mask(self,data,aspect_index,aspect_len,sents_len):
        batch_size,sequence_length=data.shape[0],data.shape[1]
        mask_data=[[] for i in range(batch_size)]
        for batch in range(batch_size):
            #aspect_index=3 代表第4个,len=2
            #0 1 2 3 4 5 6 7
            #0 0 0 1 1 0 0 0
            #(0)0->aspect_index-1(2) range(aspect_index)
            #(1)aspect_index(3)->aspect_index+aspect_len-1(4) range(aspect_index,aspect_index+aspect_len)
            #(0)aspect_index+aspect_len(5),len-1(7)  range(aspect_index+aspect_len,sequence_len)
            for i in range(aspect_index[batch]):
                mask_data[batch].append(0)
            for i in range(aspect_index[batch],aspect_index[batch]+aspect_len[batch]):
                mask_data[batch].append(1)
            for i in range(aspect_index[batch]+aspect_len[batch],sequence_length):
                mask_data[batch].append(0)
        mask_data=torch.FloatTensor(mask_data).unsqueeze(2).to(self.opt.device)
        data=data*mask_data
        return data
    
    def forward(self,sents,aspect_index,aspect_len,adj_matrix,sents_len):
        #sents:[batch_size,sequence_length]
        #aspects:[batch_size,max_aspect_length]
        sents_embedding=self.embedding(sents)#[batch_size,sequence_length,dim_w]
        sents_embedding=self.dropout(sents_embedding)
        lstm_out,(_,_)=self.bilstm(sents_embedding)#[batch_size,sequence_length,2*dim_h]
        gcn_out=torch.relu(self.GCN1(adj_matrix,self.position(lstm_out,aspect_index,aspect_len,sents_len)))#[batch_size,sequence_length,2*dim_h]
        gcn_out=torch.relu(self.GCN2(adj_matrix,self.position(gcn_out,aspect_index,aspect_len,sents_len)))#[batch_size,sequence_length,2*dim_h]
        mask_aspect=self.mask(gcn_out,aspect_index,aspect_len,sents_len).permute(0,2,1)#[batch_size,2*dim_h,sequence_length]
        beta=torch.sum(torch.bmm(lstm_out,mask_aspect),2)#[batch_size,sequence_length]
        alpha=torch.softmax(beta,1).unsqueeze(1)#[batch_size,1,sequence_length]
        r=torch.bmm(alpha,lstm_out).squeeze(1)#[batch_size,2*dim_h]
        result=self.predict(r)#[batch_size,class_num]
        return result