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

  args = parser.parse_args()
  return args
