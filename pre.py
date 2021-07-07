import pickle

def convert(name,box_list,size=(640,480)):
    dw=1.0/size[0]
    dh=1.0/size[1]
    wdata=""
    f = open("datatext/"+name[:-4]+".txt", "w") # co-ordinates for YOLO will be saved in this file
    print(box_list)
    for box_elem in box_list:
      box=box_elem[3]
      class_id=box_elem[0]
      x = (box[0] + box[2])/2.0
      y = (box[1] + box[3])/2.0
      w = box[2] - box[0]
      h = box[3] - box[1]
      x = x*dw
      w = w*dw
      y = y*dh
      h = h*dh
      wdata+="\n"+str(class_id)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h)
    f.write(wdata[1:])
    f.close()
with open('gt.pickle', 'rb') as f:
  data=pickle.load(f)
  for dt in data:
    if len(dt)>0 and (len(dt[1])>0) and (len(dt[0])>0):
      convert(dt[0],dt[1])
