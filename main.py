import sys
import torch
from torch.autograd import Variable
from preprocess import dataset
from random import shuffle

class doubleenc(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.embed1 = nn.Embedding(args.vsz1,args.esz)
    self.embed2 = nn.Embedding(args.vsz2,args.esz)
    self.lstm1 = nn.LSTM(args.esz,args.hsz)
    self.lstm2 = nn.LSTM(args.esz,args.hsz)

  def forward(self,input1,input2):
    e1 = self.embed1(input1)
    e2 = self.embed2(input2)
    l1, (h1,_) = self.lstm1(e1)
    l2, (h2,_) = self.lstm2(e2)
    return (l1,h1),(l2,h2)

class decoder(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.embed = nn.Embedding(args.ovsz1,args.esz)
    self.cell = nn.LSTMCell(args.esz,args.hsz)

  def forward(self,h1,l1,h2,l2,dec_outs):
    attn1 = self.attn(h,l1)
    attn2 = self.attn(h,l2)
    combined = 
    
  
def epoch(data,loss,model,optim,gpu=False):
  if gpu: tt = torch.cuda
  else: tt = torch
  data_size = len(data.srcvectors[0])
  ordering = list(range(data_size))
  shuffle(ordering)
  losses = []
  for i in ordering:
    optim.zero_grads()
    srcs = [Variable(tt.LongTensor(data.srcvectors[j][i])) for j in len(data.srcvectors)]
    tgts = [Variable(tt.LongTensor(data.tgtvectors[j][i])) for j in len(data.tgtvectors)]
    preds = model.forward(srcs,tgts)
    l = loss(preds,tgts)
    l.backward()
    losses.append(l.data.cpu())
    optim.step()



if __name__=="__main__":
  import opts
  args = opts.main_params()
  train = torch.load(args.data+"train.pt")
  val = torch.load(args.data+"val.pt")
