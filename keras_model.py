import numpy as np
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

filepath = os.getcwd()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import tensorflow as tf
import numpy as np
import cupy as cp
import cv2
import matplotlib.pyplot as plt
import datetime
from dataset.loaddata import data_loader
from test_model import test_runner
# from dataset.loadimg import load_img
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.utils  import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# dic= {  '0':0,'1': 1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
#         'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'i':18,'j':19,'k':20,'m':21,'n':22,
#         'p':23,'q':24,'r':25,'s':26,'t':27,'u':28,'v':29,'w':30,'x':31,'y':32,'z':33}

dic= {  '2':0, '3':1, '4':2, '5':3, '7':4, '9':5, 'a':6, 'c':7, 'f':8, 'h':9, 'k':10, 'm':11, 'n':12, 'p':13, 'q':14, 'r':15, 't':16, 'y':17, 'z':18}
start_time =datetime.datetime.now()
def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
solve_cudnn_error()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#modifying

#creat CNN model
print('Creating CNN model...')
out_shape = 19
k_initializer = "glorot_uniform"
k_regularizer = l2(0.01)
tensor_in = Input((60,160,3))
tensor_out = tensor_in
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',kernel_initializer=k_initializer)(tensor_out)
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu',kernel_initializer=k_initializer)(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',kernel_initializer=k_initializer)(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',kernel_initializer=k_initializer)(tensor_out)

tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',kernel_initializer=k_initializer)(tensor_out)
tensor_out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',kernel_initializer=k_initializer)(tensor_out)
tensor_out = BatchNormalization(axis=1)(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',kernel_initializer=k_initializer,kernel_regularizer=k_regularizer)(tensor_out)
tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',kernel_initializer=k_initializer,kernel_regularizer=k_regularizer)(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu',kernel_initializer=k_initializer,kernel_regularizer=k_regularizer)(tensor_out)
tensor_out = BatchNormalization()(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)

tensor_out = Flatten()(tensor_out)
tensor_out = Dropout(0.5)(tensor_out)
tensor_out = [Dense(out_shape , name='digit1', activation='softmax')(tensor_out),
              Dense(out_shape , name='digit2', activation='softmax')(tensor_out),
              Dense(out_shape , name='digit3', activation='softmax')(tensor_out),
              Dense(out_shape , name='digit4', activation='softmax')(tensor_out)]

model = Model(inputs=tensor_in, outputs=tensor_out)
model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
model.summary()

# set generator parameter
batch_size=20
valid_rate =0.2
data_size =180000
train_batch = int(batch_size*(1-valid_rate))
valid_batch = int(batch_size*valid_rate)

steps_per_epoch = 20
epoch = (data_size/batch_size)/steps_per_epoch
t_gen = data_loader
t_gen.init_generator(t_gen,width = 160,height = 60,channel = 3,data_size =data_size,validate_rate =valid_rate,out_shape = out_shape)
v_gen = t_gen
# t_gen.getdata(t_gen,"train")
# v_gen.getdata(v_gen,"valid")
reduced = ReduceLROnPlateau(monitor=' digit4_accuracy', factor=0.5, patience=3, verbose=1, mode='auto', cooldown=0, min_lr=0.00001)
checkpoint = ModelCheckpoint(filepath, monitor='val_digit4_acc', verbose=1, save_best_only=True, mode='auto')
earlystop = EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='auto')

callbacks_list = [checkpoint,earlystop]

print("start training")
#create train generator
history = model.fit_generator(generator=t_gen.data_generator(self=t_gen,batch_size=train_batch,step="train"),steps_per_epoch=steps_per_epoch,
                                validation_data=v_gen.data_generator(self=v_gen,batch_size=valid_batch,step="valid"),validation_steps=steps_per_epoch,
                                max_queue_size=1, epochs=epoch,verbose=2, validation_freq=1,callbacks= callbacks_list)



end_time = datetime.datetime.now()
delta = end_time - start_time
print("總共耗時:"+str(delta))


model.save('my_model.h5') 

# 绘制训练 & 验证的准确率值

trainacc = np.add(history.history['digit1_accuracy'],history.history['digit2_accuracy'])
trainacc = np.add(trainacc,history.history['digit3_accuracy'])
trainacc = np.add(trainacc,history.history['digit4_accuracy'])
trainacc/=4

valacc = np.add(history.history['val_digit1_accuracy'],history.history['val_digit2_accuracy'])
valacc = np.add(valacc,history.history['val_digit3_accuracy'])
valacc = np.add(valacc,history.history['val_digit4_accuracy'])
valacc/=4


# 绘制训练 & 验证的损失值

plt.plot(trainacc)
plt.plot(valacc)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

test_runner()