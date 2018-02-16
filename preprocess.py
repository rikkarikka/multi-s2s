import sys
import os
import torch
from collections import Counter

class dataset:
  def __init__(self,args,fn):
    self.args = args
    self.srcvectors = []
    self.tgtvectors = []
    self.srcvocabs = []
    self.tgtvocabs = []
    self.data = self.load(fn)

  def mk_vocab(self,data):
    c = Counter(data)
    itos = ['<pad>','<start>','<end>','<unk>'] + [x for x in c if c[x]>self.args.oov]
    stoi = {x:itos.index(x) for x in itos}
    return itos,stoi 

  def mk_one_hot(self,item,vocab):
    return [vocab[x] if x in vocab else vocab['<unk>'] for x in item]

  def load(self,fn):
    fns = os.listdir(fn)
    srcs = [x for x in fns if "src" in x]
    tgts = [x for x in fns if "tgt" in x]
    srcs.sort()
    tgts.sort()
    def get_vocab_vecs(filename,src=True):
      tmpvectors = []
      with open(filename) as g:
        tmpvocab = self.mk_vocab(g.read().split())
        tmpstoi = tmpvocab[1]
        g.seek(0)
        for l in g:
          tmpvectors.append(self.mk_one_hot(l.split(" "),tmpstoi))
      if src:
        self.srcvocabs.append(tmpvocab)
        self.srcvectors.append(tmpvectors)
      else:
        self.tgtvocabs.append(tmpvocab)
        self.tgtvectors.append(tmpvectors)
    for f in srcs:
      get_vocab_vecs(fn+f)
    for f in tgts:
      get_vocab_vecs(fn+f,src=False)

if __name__=="__main__":
  import opts
  args = opts.preprocess_params()
  train = dataset(args,args.train)
  val = dataset(args,args.val)
  torch.save(train, args.data+"train.pt")
  torch.save(val, args.data+"val.pt")
