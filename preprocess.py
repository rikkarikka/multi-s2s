import sys
import os
import torch
from collections import Counter

class dataset:
  def __init__(self,args,fn,vocab=False):
    self.args = args
    self.srcvectors = []
    self.tgtvectors = []
    self.srcvocabs = []
    self.tgtvocabs = []
    self.vocab = vocab
    self.data = self.load(fn)

  def lenvocabs(self):
    return [len(x[0]) for x in self.srcvocabs], [len(x[0]) for x in self.tgtvocabs]

  def mk_vocab(self,fn):
    with open(fn) as f:
      data = f.read().split()
    c = Counter(data)
    itos = ['<pad>','<start>','<end>','<unk>'] + [x for x in c if c[x]>self.args.oov]
    stoi = {x:itos.index(x) for x in itos}
    return itos,stoi 

  def mk_one_hot(self,item,vocab):
    return [vocab[x] if x in vocab else vocab['<unk>'] for x in item]

  def load(self,fn):
    fns = os.listdir(fn)
    srcs = [fn+x for x in fns if "src" in x]
    tgts = [fn+x for x in fns if "tgt" in x]
    srcs.sort()
    tgts.sort()
    if self.vocab:
      print('using vocabs')
      self.srcvocabs, self.tgtvocabs = self.vocab
    else:
      self.srcvocabs = [self.mk_vocab(f) for f in srcs]
      self.tgtvocabs = [self.mk_vocab(f) for f in tgts]
    print(len(self.srcvocabs))
    #vectors 
    self.srcvectors = [self.get_vecs(f,self.srcvocabs[i][1]) for i,f in enumerate(srcs)]
    self.tgtvectors = [self.get_vecs(f,self.tgtvocabs[i][1]) for i,f in enumerate(tgts)]
      
  def get_vecs(self,filename,vocab):
    tmpvectors = []
    with open(filename) as g:
      for l in g:
        tmpvectors.append(self.mk_one_hot(l.split(" "),vocab))
    return tmpvectors

if __name__=="__main__":
  import opts
  args = opts.preprocess_params()
  train = dataset(args,args.train)
  val = dataset(args,args.val,vocab=(train.srcvocabs,train.tgtvocabs))
  torch.save(train, args.data+"train.pt")
  torch.save(val, args.data+"val.pt")
