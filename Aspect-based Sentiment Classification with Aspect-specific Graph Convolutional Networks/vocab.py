import spacy
import os
from pickle import load,dump
from spacy.tokens import Doc
import numpy as np

class WhitespaceTokenizer(object):
    def __init__(self,vocab):
        self.vocab=vocab
    
    def __call__(self,text):
        words=text.split()
        spaces=[True]*len(words)
        return Doc(self.vocab,words=words,spaces=spaces)

class Vocab:
    def __init__(self,trainName,testName):
        self.nlp=spacy.load('en_core_web_sm')
        self.nlp.tokenizer=WhitespaceTokenizer(self.nlp.vocab)
        self.trainSent,self.trainAspectIndex,self.trainAspectLen,self.trainAdjMatrix,self.trainY=self.readFile(trainName)
        self.testSent,self.testAspectIndex,self.testAspectLen,self.testAdjMatrix,self.testY=self.readFile(testName)
        self.words=['pad']
        self.pad=0
        self.length=0
        self.word_vecs=None
        self._word2id=None
        self.build()
        self.inGlove=[]
    
    def dependency_adj_matrix(self,text):
        tokens=self.nlp(text)
        words=text.split()
        matrix=np.zeros((len(words),len(words))).astype('float32')
        assert len(words)==len(list(tokens))

        for token in tokens:
            matrix[token.i][token.i] = 1
            for child in token.children:
                matrix[token.i][child.i] = 1
                matrix[child.i][token.i] = 1
        return matrix
    
    def readFile(self,fileName):
        sentences=[]
        aspect_len=[]
        y=[]
        aspect_index=[]
        adj_matrix=[]
        with open(fileName,'r',encoding='utf-8',newline='\n') as file:
            lines=file.readlines()
            file.close()
            for i in range(0,len(lines),3):
                text_left,_,text_right=[s.lower().strip() for s in lines[i].partition("$T$")]
                aspect_index.append(len(text_left.split()))
                aspect=lines[i+1].lower().strip()
                text_raw=text_left+" "+aspect+" " +text_right
                label=int(lines[i+2])+1
                aspect=aspect.split()
                aspect_len.append(len(aspect))
                adj_matrix.append(self.dependency_adj_matrix(text_raw))
                text_raw=text_raw.split()
                sentences.append(text_raw)
                y.append(label)
            return sentences,aspect_index,aspect_len,adj_matrix,y
    
    def words2indices(self,sents):
        if type(sents[0])==list:
            return [[self._word2id[w] for w in s] for s in sents]
        else:
            return [self._word2id[w] for w in sents]
    
    def build(self):
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

def buildVocab(opt):
    train_file=None
    test_file=None
    if opt.dataset=="rest14":
        train_file='../data/semeval14/Restaurants_Train.xml.seg'
        test_file='../data/semeval14/Restaurants_Test_Gold.xml.seg'
    elif opt.dataset=="laptop":
        train_file='../data/semeval14/Laptops_Train.xml.seg'
        test_file='../data/semeval14/Laptops_Test_Gold.xml.seg'
    elif opt.dataset=="twitter":
        train_file='../data/Twitter/raw_data/train.txt'
        test_file='../data/Twitter/raw_data/test.txt'
    elif opt.dataset=="rest15":
        train_file='../data/semeval15/restaurant_train.raw'
        test_file='../data/semeval15/restaurant_test.raw'
    elif opt.dataset=="rest16":
        train_file='../data/semeval16/restaurant_train.raw'
        test_file='../data/semeval16/restaurant_test.raw'
    vocab_path="vocab_"+opt.dataset+".pkl"
    if os.path.exists(vocab_path):
        print('Load vocab.pkl')
        vocab=load(open(vocab_path,'rb'))
    else:
        print('Creat vocab.pkl')
        vocab=Vocab(train_file,test_file)
        vocab.load_pretrained_vec('../data/glove/glove.840B.300d.txt')
        dump(vocab,open(vocab_path,'wb'))
    return vocab