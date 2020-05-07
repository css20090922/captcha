import numpy as np
import sys, os
dataset_dir = os.path.dirname(os.path.abspath(__file__))
file_path = dataset_dir + "\\copy"
allimg = os.listdir(file_path)
# tmp = "1000.jpg"
# allimg = np.delete(allimg,0)
# allimg = np.append(allimg,tmp)

for i in range(len(allimg)):
   
    fname = allimg[i]
    fnum = fname.replace(".jpg","")
    fnum =str(int(fnum)-20000) +".jpg"
    print(fnum)
    os.rename(file_path+"\\"+fname,file_path+"\\"+fnum)


# print(fname)