import glob
import os
tr = open("test.txt", "a")
for elm in glob.glob("test_data/*.jpg"):
      tr.write(elm+"\n")
tr.close()
