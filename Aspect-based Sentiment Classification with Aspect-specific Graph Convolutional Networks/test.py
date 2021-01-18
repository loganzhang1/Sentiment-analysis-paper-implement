from itertools import chain
import torch
from models import ASGCN_DG
import torch.nn as nn
from data_utils import estimate,cuda2cpu,setSeed,myDataset
from torch.utils.data import DataLoader
from vocab import buildVocab
import numpy as np
import sys
import argparse

def test(model,dataLoader,opt):
    model.eval()
    with torch.no_grad():
        allLoss=[]
        allResult=[]
        allLabel=[]
        allPred=[]
        crossentry=nn.CrossEntropyLoss()
        for i,batch in enumerate(dataLoader):
            sents,aspect_index,aspect_len,adj_matrix,labels=batch
            sents_len=torch.LongTensor([len(sent) for sent in sents]).to(opt.device)
            sents=myDataset.to_input_tensor(sents).to(opt.device)
            aspect_index=torch.LongTensor(aspect_index).to(opt.device)
            aspect_len=torch.LongTensor(aspect_len).to(opt.device)
            adj_matrix=torch.LongTensor(adj_matrix).to(opt.device)
            labels=torch.LongTensor(labels).to(opt.device)
            pred=model(sents,aspect_index,aspect_len,adj_matrix,sents_len)
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
        sys.stdout.flush()
        return accuracy,f1score

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_name',default='ASGCN_DG',type=str)
    parser.add_argument('--dataset',default='laptop',type=str,help='twitter,rest14,laptop,rest15,rest16')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--device',default="cuda:0",type=str)
    opt=parser.parse_args()
    model_class={
        'ASGCN_DG':ASGCN_DG
    }
    setSeed()
    vocab=buildVocab(opt)
    model_path="model_"+opt.dataset+"_"+opt.model_name+".pkl"
    model=torch.load(model_path)
    testLoader=DataLoader(myDataset(vocab.testSent,vocab.testAspectIndex,vocab.testAspectLen,vocab.testAdjMatrix,vocab.testY,vocab.pad),batch_size=opt.batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
    test(model,testLoader,opt)