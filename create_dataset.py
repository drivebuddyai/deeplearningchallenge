import pickle
import os

with open('gt.pickle', 'rb') as f:
  data=pickle.load(f)
  for dt in data:
    if len(dt)>0 and (len(dt[1])>0) and (len(dt[0])>0):
      os.system("cp JPEGImages/"+str(dt[0])+" dataset/")
      os.system("cp datatext/"+str(dt[0][:-4]+".txt")+" dataset/")
