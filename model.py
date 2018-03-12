import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from allennlp.modules.attention import Attention

class model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.enc1 = encoder(args.vsz1,args.esz,args.hsz)
    self.enc2 = encoder(args.vsz2,args.esz,args.hsz)
    self.dec = decoder(args)

  def reorganize(self,x,idxs):
    return x.index_select(0,idxs)

  def forward(self,inputs,bookkeeping):
    l1,h1 = self.enc1(inputs[0],bookkeeping[4])
    l2,h2 = self.enc2(inputs[1],bookkeeping[5])
    l1 = l1.index_select(0,bookkeeping[0])
    h1 = h1.index_select(1,bookkeeping[0])
    l2 = l2.index_select(0,bookkeeping[2])
    h2 = h2.index_select(1,bookkeeping[2])
    mask1 = bookkeeping[1]
    mask2 = bookkeeping[3]
    print(inputs[2].size())
    outputs = self.dec(h1,l1,mask1,h2,l2,mask2,inputs[2])
    outputs = torch.stack(outputs,dim=1)
    return outputs

class encoder(nn.Module):
  def __init__(self, vsz, esz, hsz):
    super().__init__()
    self.embed = nn.Embedding(vsz,esz)
    self.lstm = nn.LSTM(esz,hsz,batch_first=True)

  def forward(self,inp,lens):
    e = self.embed(inp)
    e = nn.utils.rnn.pack_padded_sequence(e,lens,batch_first=True)
    l, (h,_) = self.lstm(e)
    l = nn.utils.rnn.pad_packed_sequence(l,batch_first=True)[0]
    return l,h

class decoder(nn.Module):
  def __init__(self,args,gpu=False):
    super().__init__()
    self.embed = nn.Embedding(args.ovsz1,args.esz)
    self.cell = nn.GRUCell(args.esz,args.hsz)
    self.encproj = nn.Linear(args.hsz*2,args.hsz)
    self.hproj = nn.Linear(args.hsz*3,args.hsz)
    self.gproj = nn.Linear(args.hsz,args.ovsz1)
    self.attn = Attention()
    if gpu: self.tt = torch.cuda
    else: self.tt = torch

  def mkstart(self,bsz):
    return self.embed(Variable(self.tt.LongTensor(bsz,1).fill_(1)))

  def gen(self,g):
    _, idxs = torch.max(g,1)
    return idxs

  def forward(self,h1,l1,mask1,h2,l2,mask2,dec_outs,ss=False):
    h = F.tanh(self.encproj(torch.cat((h1,h2),dim=2)))
    print(h.size())
    prevw = self.mkstart(1)
    prevw = prevw.squeeze(0)
    h = h.squeeze(0)
    
    outputs = []
    print(dec_outs.size())
    for i in range(dec_outs.size(1)):
      op = self.cell(prevw,h)
      attn1 = self.attn(h,l1,mask1)
      attn2 = self.attn(h,l2,mask2)
      attended1 = torch.bmm(attn1.unsqueeze(1),l1).squeeze()
      attended2 = torch.bmm(attn2.unsqueeze(1),l2).squeeze()
      combined = torch.cat([attended1,attended2,op],dim=1)
      h = F.tanh(self.hproj(combined))
      g = self.gproj(h)
      outputs.append(g)
      if ss:
        prevw = self.embed(dec_outs[i])
      else:
        prevw = self.embed(self.gen(g)) 
    return outputs
  

