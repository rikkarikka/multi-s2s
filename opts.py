import argparse

def preprocess_params():
  parser = argparse.ArgumentParser(description='none')
  parser.add_argument('-data',default='./data/')
  parser.add_argument('-train',default='./data/train/')
  parser.add_argument('-val',default='./data/val/')
  parser.add_argument('-oov',default=5)
  
  
  args = parser.parse_args()
  return args

def main_params():
  parser = argparse.ArgumentParser(description='none')
  parser.add_argument('-data',default='./data/')
  parser.add_argument('-bsz',default=2)
  parser.add_argument('-esz',default=300)
  parser.add_argument('-hsz',default=500)
  parser.add_argument('-vsz1',default=500)
  parser.add_argument('-vsz2',default=500)
  parser.add_argument('-ovsz1',default=500)
  parser.add_argument('-ovsz2',default=500)
  parser.add_argument('-lr',default=0.01)

  args = parser.parse_args()
  return args
