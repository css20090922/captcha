# coding: utf-8
import os.path
import pickle
import os
import cupy as cp
import numpy as np
import cv2
from PIL import Image
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from skimage import transform,data
dataset_dir = os.path.dirname(os.path.abspath(__file__))
key_file = {
    'train_img':'trainimg',
    'train_label':'trainlabel.txt',
    'test_img':'testimg',
    'test_label':'testlabel.txt'
}
# dic= {  '0':0,'1': 1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
#         'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'i':18,'j':19,'k':20,'m':21,'n':22,
#         'p':23,'q':24,'r':25,'s':26,'t':27,'u':28,'v':29,'w':30,'x':31,'y':32,'z':33}
dic= {  '2':0, '3':1, '4':2, '5':3, '7':4, '9':5, 'a':6, 'c':7, 'f':8, 'h':9, 'k':10, 'm':11, 'n':12, 'p':13, 'q':14, 'r':15, 't':16, 'y':17, 'z':18}
imgdic = [[0] * 2 for i in range(100)]


WHITE=[255,255,255]

#標籤one hot label化
def to_onelist(text,out_shape):
    label_list = []
    for c in text:
        onehot = [0 for _ in range(out_shape)]
        if c in dic:
            onehot[ dic[c] ] = 1
            label_list.append(onehot)
    return label_list

def to_text(l_list):
    
    text=[]
    pos = []
    for i in range(4):
        for j in range(19):
            if(l_list[i][j]):
                pos.append(j)
                break

    for i in range(4):
        char_idx = pos[i]
        text.append(list(dic.keys())[list(dic.values()).index(char_idx)])
        return "\n".join(text)
#讀標籤
def _load_label(step):
    labelkey = step + "_label"
    file_name = key_file[labelkey]
    file_path = dataset_dir + "\\" + file_name

    print("Converting " + step + "txt to NumPy Array ...")

    with open(file_path,"r",encoding='utf-8') as f:
        labels = f.read().encode('utf-8').decode('utf-8-sig')
        labels = labels.split('\n')
    return labels

#圖片處理
def _img_plus(img):
    height, width, channels = img.shape #get img height and width
    img = cv2.fastNlMeansDenoisingColored(img,None,31,31,7,21)
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    imgarr[:,100:width-40] = 0
    imagedata = np.where(imgarr == 255) #find where are white
    
    
    X = np.array([imagedata[1]])
    Y = height - imagedata[0]


    poly_reg= PolynomialFeatures(degree = 2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)

    X2 = np.array([[i for i in range(0,width)]])
    X2_ = poly_reg.fit_transform(X2.T)
    for ele in np.column_stack([regr.predict(X2_).round(0),X2[0],] ):
        pos = height - int(ele[0])
        thresh[pos-int(imgdic[height][0]):pos+int(imgdic[height][1]), int(ele[1])] = 255 - thresh[pos-int(imgdic[height][0]):pos+int(imgdic[height][1]),int(ele[1])] #這裡可以更改回歸線條上下範圍
    
    newdst=transform.resize(thresh, (60, 160)) #resize (h,w)
    return newdst

class data_loader:
   
    def init_generator(self,width = 160,height = 60,channel = 3,data_size=20000,validate_rate =0.2,out_shape = 34):
        self.data_size = data_size
        self.validate_num = int(data_size*validate_rate)
        self.height = height
        self.width = width
        self.img_size = width*height
        self.channel=channel
        self.out_shape = out_shape
        self.mask = []
        for i in range(data_size): 
            self.mask.append(i)
        np.random.shuffle(self.mask)

        #把索引分開來
        self.trainmask = self.mask[:-self.validate_num]
        self.validmask = self.mask[-self.validate_num:]
        for i in range(100):
            imgdic[i][0]=25
            imgdic[i][1]=25
            imgdic[50][0]=26
            imgdic[50][1]=24
            imgdic[48][0]=23
            imgdic[48][1]=30
            imgdic[46][0]=27
            imgdic[46][1]=25
            imgdic[45][0]=21
            imgdic[45][1]=30
     
    def init_test_generator(self,width = 160,height = 60,channel = 3,data_size = 10000,out_shape = 34):
        self.data_size = data_size
        
        self.height = height
        self.width = width
        self.img_size = width*height
        self.channel=channel
        self.out_shape = out_shape

    def getlabel(self,step):
        
        datalabel = _load_label(step)   
        return datalabel

    def data_generator(self,batch_size,step):
        if step == "train" or step =="valid":
           
            self.step = "train"
        else:
            self.step = "test"

         #讀取img內所有檔案的列表
        imgkey = self.step + "_img"
        file_name =  key_file[imgkey]
        file_path = dataset_dir + "\\" + file_name
        allimg = os.listdir(file_path)
    
        pardir = "C:\\programming\\ai_test\\dataset\\"+key_file[imgkey]+"\\"

        #載入label，然後把訓練資料的label分開來
        datalabel = _load_label(self.step)
        
        if step =="train":
            trainimg = np.array(allimg)[self.trainmask]
        elif step=="valid":
            validimg = np.array(allimg)[self.validmask]
        else:
            validimg = np.array(allimg)
        
        epoch=1
        start_ptr = 0
        

        while True:
            data = []
            
            if(step == "train"):
                dataimg = trainimg

            else :
                dataimg = validimg

             #取出該批次的label進行one hot label
            batch_label = []
            datalabels = []
            for label in dataimg:
                label = label.replace(".jpg","")
                datalabels.append(datalabel[int(label)-1])
            
            datalabels = datalabels[start_ptr:start_ptr+batch_size]
           
            datalabels =  [to_onelist(row,self.out_shape) for row in datalabels]
            datalabels = np.array(datalabels)
            
            label = [[] for _ in range(4)]
            
            for arr in datalabels:
                for i in range(4):      
                    label[i].append(arr[i])
               
            datalabels = label
            #載入data
            for file in dataimg [start_ptr:start_ptr+batch_size]:
                     
                file = pardir + file      
                
                img = cv2.imread(file)

                #對圖片做處裡 
                img = _img_plus(img)

                #把圖轉成160*60大小然後轉numpy
                img = cv2.copyMakeBorder(img,0,max(self.width-img.shape[1],0),0,max(self.height-img.shape[0],0),cv2.BORDER_CONSTANT,value = WHITE)/255.0
                if cp.asarray(data).shape[0] == 0:
                    data = img
                else :
                    data = cp.concatenate((cp.array(data),cp.asarray(img)))

            data = cp.asnumpy(data).reshape(batch_size,self.height,self.width,self.channel)
            print("\n"+step+str(epoch)+" : "+str(data.shape)+" ,img loading Done")

            yield data,datalabels
            start_ptr = start_ptr+batch_size
            epoch+=1
    # def load_test(self,test_num):

    #     file_path = dataset_dir + "\\" + key_file['test_img']
    #     # file = None
    #     print("Converting " + key_file['test_img'] + " to NumPy Array ...")
    #     allimg = os.listdir(file_path)
    #     pardir = "C:\\programming\\ai_test\\dataset\\"+key_file['test_img']+"\\"
    #     data = []
    #     pickdata = allimg[:test_num]
        
    #     i=1
    #     for file in pickdata: 
             
    #             file = pardir + file
              
    #             img = cv2.imread(file)

    #             #對圖片做處裡 
    #             img = _img_plus(img)

    #             #把圖轉成160*60大小然後轉numpy
    #             img = cv2.copyMakeBorder(img,0,max(self.width-img.shape[1],0),0,max(self.height-img.shape[0],0),cv2.BORDER_CONSTANT,value = WHITE)/255.0
    #             if cp.asarray(data).shape[0] == 0:
    #                 data = img
    #             else :
    #                 data = cp.concatenate((cp.array(data),cp.asarray(img)))
    #     data = cp.asnumpy(data).reshape(test_num,self.height,self.width,self.channel)
    #     print(data.shape)
    #     print(step+" img loading Done")
        
        
    #     return data    
            

    
