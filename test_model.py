import numpy as np
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import tensorflow as tf
import numpy as np
import cupy as cp
import cv2
import matplotlib.pyplot as plt
import datetime
from dataset.loaddata import data_loader
from keras.models import load_model,Model

# dic= {  '0':0,'1': 1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
        # 'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'i':18,'j':19,'k':20,'m':21,'n':22,
        # 'p':23,'q':24,'r':25,'s':26,'t':27,'u':28,'v':29,'w':30,'x':31,'y':32,'z':33}

dic= {  '2':0, '3':1, '4':2, '5':3, '7':4, '9':5, 'a':6, 'c':7, 'f':8, 'h':9, 'k':10, 'm':11, 'n':12, 'p':13, 'q':14, 'r':15, 't':16, 'y':17, 'z':18}

batch_size=40
epoch=25
loader = None
model = None
out_shape = 19

def test_runner():
    print("start testing")
    start_time =datetime.datetime.now()
    init()
    # score = testloss(loader)
    testacc(loader)
    # res_plt(score)
    end_time = datetime.datetime.now()
    delta = end_time - start_time
    print("總共耗時:"+str(delta))

def init():
    global loader,model
    #測試資料generator
    loader = data_loader
    loader .init_test_generator(loader ,width = 160,height = 60,channel = 3,data_size=1000,out_shape = out_shape)
    model = load_model('my_model.h5')
def testloss(test_gen):
    #計算測試的loss function
    score = model.evaluate_generator(generator=test_gen.data_generator(self=test_gen,batch_size=batch_size,step="test"),  steps=epoch, max_queue_size=1, workers=1, verbose=1)
    print("test loss :"+str(score))
    return score

def testacc(test_gen):
    #計算測試的準確率
    res = model.predict_generator(generator=test_gen.data_generator(self=test_gen,batch_size=batch_size,step="test"),  steps=epoch, max_queue_size=1, workers=1, verbose=1)
    reslabel = [[[] for _ in range(4)] for _ in range(int(batch_size*epoch))]

    for i in range(len(res)):
        for j in range(len(res[i])):
            value = np.argmax(res[i][j],axis=0)
            tmp = list (dic.keys()) [list (dic.values()).index (value)]
            reslabel[j][i].append(tmp)

    uncorrect = 0
    testlabels = test_gen.getlabel(test_gen,"test")
    for i in range(batch_size*epoch) :
        label = testlabels[i]   
        
        res = str(i)+"  ans:"+str(label)+",res:"+str(reslabel[i][0])+str(reslabel[i][1])+str(reslabel[i][2])+str(reslabel[i][3])
        for j in range(len(label)):
            if (label[j]!=reslabel[i][j][0]):
                uncorrect+=1
                res += "wrong"
                break
        print(res)
    error_rate =(uncorrect/(batch_size*epoch))
    print((1-error_rate))

def res_plt(score):
    plt.plot(score)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend('test', loc='upper left')
    plt.show()


if __name__ == '__main__':
    test_runner()
    
    