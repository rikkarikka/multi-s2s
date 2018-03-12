import sys
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from preprocess import dataset
from random import shuffle
from allennlp.modules.attention import Attention
from model import model

def pad(l,m=False):
  if not m:
    m = len(l[0])
  for i in range(len(l)):
    p = [0]*(m-len(l[i]))
    l[i] = l[i]+p
  return l

def mk_mask(l):
  m = max(l)
  ones = [[1]*x for x in l]
  return pad(ones,m=m)
  
def batchify(data,bsz):
  ds = []
  for i in range(len(data.tgtvectors[0])):
    ds.append((data.tgtvectors[0][i],data.srcvectors[0][i],data.srcvectors[1][i]))
  ds.sort(key=lambda x: len(x[0]),reverse=True)
  cut = len(ds)%bsz
  ds = ds[cut:]
  #ds.reverse()
  batches = []
  i = 0
  while i < len(ds):
    tgts,src1,src2 = [list(x) for x in zip(*ds[i:i+bsz])]
    i+=bsz
    idx1 = sorted(range(len(src1)),key=lambda k:src1[k],reverse=True)
    lens1 = [len(x) for x in src1]
    mask1 = mk_mask(lens1)
    src1.sort(key=lambda x:len(x),reverse=True)
    lens1.sort(reverse=True)
    idx2 = sorted(range(len(src2)),key=lambda k:src2[k],reverse=True)
    lens2 = [len(x) for x in src2]
    mask2 = mk_mask(lens2)
    lens2.sort(reverse=True)
    src2.sort(key=lambda x:len(x),reverse=True)
    tgts = pad(tgts)
    src1 = pad(src1)
    src2 = pad(src2)
    batches.append((src1,src2,tgts,(idx1,lens1,mask1),(idx2,lens2,mask2)))
  return batches
  
def epoch(data,loss,model,optim,args,gpu=False):
  if gpu: tt = torch.cuda
  else: tt = torch
  #shuffle(batches)
  losses = []
  for b in data[:1]:
    optim.zero_grad()
    src1, src2, tgts = b[:3]
    idx1,lens1,mask1 = b[3]
    idx2,lens2,mask2 = b[4]
    idx1 = Variable(tt.LongTensor(idx1))
    idx2 = Variable(tt.LongTensor(idx2))
    mask1 = Variable(tt.FloatTensor(mask1))
    mask2 = Variable(tt.FloatTensor(mask2))
    tgts = Variable(tt.LongTensor(tgts))
    src1 = Variable(tt.LongTensor(src1))
    src2 = Variable(tt.LongTensor(src2))
    inputs = (src1,src2,tgts)
    bookkeeping = (idx1,mask1,idx2,mask2,lens1,lens2)
    preds = model.forward(inputs,bookkeeping)
    l = loss(preds.view(-1,args.ovsz1),tgts.view(-1))
    l.backward()
    losses.append(l.data.cpu()[0])
    print(losses[-1])
    optim.step()
  epo = sum(losses)/len(losses)
  return epo

  
def validate(data,loss,model,optim,args,gpu=False):
  if gpu: tt = torch.cuda
  else: tt = torch
  batches = batchify(data,args.bsz)
  t_loss = 0
  t_tokens = 0
  acc = 0
  model.eval()
  for b in batches[:1]:
    optim.zero_grad()
    src1, src2, tgts = b[:3]
    idx1,lens1,mask1 = b[3]
    idx2,lens2,mask2 = b[4]
    idx1 = Variable(tt.LongTensor(idx1))
    idx2 = Variable(tt.LongTensor(idx2))
    mask1 = Variable(tt.FloatTensor(mask1))
    mask2 = Variable(tt.FloatTensor(mask2))
    tgts = Variable(tt.LongTensor(tgts))
    src1 = Variable(tt.LongTensor(src1))
    src2 = Variable(tt.LongTensor(src2))
    inputs = (src1,src2,tgts)
    bookkeeping = (idx1,mask1,idx2,mask2,lens1,lens2)
    preds = model.forward(inputs,bookkeeping)
    preds = preds.view(-1,args.ovsz1)
    tgts = tgts.view(-1)
    t_loss += preds.size(0) * loss(preds,tgts).data[0]
    _,midxs = preds.max(1)
    acc += midxs.eq(tgts).data[0]
    t_tokens += preds.size(0)
  
  model.train()
  t_loss = t_loss / t_tokens
  print("Val loss: ",t_loss)
  print("Val accuracy: ", acc / t_tokens)
  if t_loss<100:
    print("Val ppl: ",math.exp(t_loss))
  else:
    print("Val ppl: real big")
  return t_loss

if __name__=="__main__":
  import opts
  args = opts.main_params()
  train = torch.load(args.data+"train.pt")
  val = torch.load(args.data+"val.pt")
  svsz,tvsz = train.lenvocabs()
  args.vsz1,args.vsz2 = svsz
  print(args.vsz1,args.vsz2)
  args.ovsz1 = tvsz[0]
  m = model(args)
  optim = torch.optim.Adam(params=m.parameters(),lr=args.lr)
  weights = torch.FloatTensor(args.ovsz1).fill_(1)
  weights[0] = 0
  loss = nn.CrossEntropyLoss(weights)
  lastloss = sys.maxsize
  traindata = batchify(train,args.bsz)
  while True:
    e = epoch(traindata,loss,m,optim,args)
    print("Train Loss",e)
    v = validate(train,loss,m,optim,args)
    if v>lastloss:
      args.lr = args.lr*0.5
      print("decaying lr to ",args.lr)
    lastloss = v

