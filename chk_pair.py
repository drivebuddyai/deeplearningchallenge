import glob
import os

t=glob.glob("dataset_cp/*.txt")

for f in glob.glob("dataset_cp/*.jpg"):
  if f[:-4]+".txt" in t:
    continue
  else:
    print("removed : "+f)
    os.system("rm "+f)


t=glob.glob("dataset_cp/*.jpg")
for f in glob.glob("dataset_cp/*.txt"):
  if f[:-4]+".jpg" in t:
    continue
  else:
    print("removed : "+f)
    os.system("rm "+f)

