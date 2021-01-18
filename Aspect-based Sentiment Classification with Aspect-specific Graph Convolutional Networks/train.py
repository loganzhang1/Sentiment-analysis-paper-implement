import torch
from data_utils import weight_init,myDataset
from torch.utils.data import DataLoader
from data_utils import setSeed
from vocab import buildVocab,Vocab
import argparse
import torch.nn as nn
from test import test
import numpy as np
from models import ASGCN_DG

def train(opt,vocab):
    model=opt.model(vocab,opt)
    model.apply(weight_init)
    model=model.to(opt.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=opt.lr,weight_decay=opt.l2reg)
    trainLoader=DataLoader(myDataset(vocab.trainSent,vocab.trainAspectIndex,vocab.trainAspectLen,vocab.trainAdjMatrix,vocab.trainY,vocab.pad),batch_size=opt.batch_size,shuffle=True,collate_fn=myDataset.collate_fn)
    testLoader=DataLoader(myDataset(vocab.testSent,vocab.testAspectIndex,vocab.testAspectLen,vocab.testAdjMatrix,vocab.testY,vocab.pad),batch_size=opt.batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
    crossentry=nn.CrossEntropyLoss()
    max_acc=0
    global_step=0
    max_f1=0
    model_path="model_"+opt.dataset+"_"+opt.model_name+".pkl"
    for i in range(opt.epoch):
        allLoss=[]
        for j,batch in enumerate(trainLoader):
            global_step=global_step+1
            model.train()
            optimizer.zero_grad()
            sents,aspect_index,aspect_len,adj_matrix,labels=batch
            sents_len=torch.LongTensor([len(sent) for sent in sents]).to(opt.device)
            sents=myDataset.to_input_tensor(sents).to(opt.device)
            aspect_index=torch.LongTensor(aspect_index).to(opt.device)
            aspect_len=torch.LongTensor(aspect_len).to(opt.device)
            labels=torch.LongTensor(labels).to(opt.device) 
            adj_matrix=torch.LongTensor(adj_matrix).to(opt.device)
            pred=model(sents,aspect_index,aspect_len,adj_matrix,sents_len)
            _,result=torch.max(pred,1)
            loss=crossentry(pred,labels)
            allLoss.append(loss.item())
            loss.backward()
            optimizer.step()
            if global_step%5==0:
                accuracy,f1score=test(model,testLoader,opt)
                if accuracy>max_acc:
                    max_acc=accuracy
                    max_f1=f1score
                    torch.save(model,model_path)
    print('Max Accuracy:{} f1 score:{}'.format(max_acc,max_f1))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ASGCN_DG', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, rest14, laptop, rest15, rest16')
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--dim_w', default=300, type=int)
    parser.add_argument('--dim_h', default=300, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    opt = parser.parse_args()
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    model_class = {
        'ASGCN_DG':ASGCN_DG
    }
    opt.model=model_class[opt.model_name]
    opt.optimizer=optimizers[opt.optim]
    opt.class_num=3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    setSeed()
    vocab=buildVocab(opt)
    train(opt,vocab)
