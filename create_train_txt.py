import glob

tr = open("train.txt", "a")
for elm in glob.glob("dataset/*.jpg"):
      tr.write(elm+"\n")
tr.close()