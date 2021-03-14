import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D,Activation
from keras import backend as k
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(1000)
X=np.load("D:/DerinOgrenme/onlyFaceArray.npy") #oluşturulan label ve array  dizileri yükleniyor
y=np.load("D:/DerinOgrenme/onlyFaceLabel.npy")

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1000) # test ve train olarak ikiye ayrılıyor setimiz
batch_size=128
num_classes=5 #çıkış sayımız
epoch=100# tekrar sayımız

img_row,img_clm=128,128

if k.image_data_format()=="channels_first":
    x_train=x_train.reshape(x_train.shape[0],1,img_row,img_clm)
    x_test=x_test.reshape(x_test.shape[0],1,img_row,img_clm)
    image_shape=(1,img_row,img_clm)
else:
    x_train=x_train.reshape(x_train.shape[0],img_row,img_clm,3)
    x_test=x_test.reshape(x_test.shape[0],img_row,img_clm,3)
    image_shape=(img_row,img_clm,3)
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

#ALEX-NET mimarisi oluşturma adımlarımız aşağıda
model=Sequential()

model.add(Conv2D(filters=64,input_shape=image_shape,kernel_size=(3,3),activation="relu",strides=(1,1),padding="valid"))
#konvolüsyon pooling ve dense katmaları tanımlıyoruz
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="valid"))

model.add(Conv2D(filters=128,input_shape=image_shape,kernel_size=(3,3),activation="relu",strides=(1,1),padding="valid"))

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="valid"))

model.add(Conv2D(filters=256,input_shape=image_shape,kernel_size=(3,3),activation="relu",strides=(1,1),padding="valid"))
model.add(Conv2D(filters=256,input_shape=image_shape,kernel_size=(3,3),activation="relu",strides=(1,1),padding="valid"))
model.add(Conv2D(filters=256,input_shape=image_shape,kernel_size=(3,3),activation="relu",strides=(1,1),padding="valid"))

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="valid"))

model.add(Conv2D(filters=512,input_shape=image_shape,kernel_size=(3,3),activation="relu",strides=(1,1),padding="valid"))
model.add(Conv2D(filters=512,input_shape=image_shape,kernel_size=(3,3),activation="relu",strides=(1,1),padding="valid"))
model.add(Conv2D(filters=512,input_shape=image_shape,kernel_size=(3,3),activation="relu",strides=(1,1),padding="valid"))
model.add(Conv2D(filters=512,input_shape=image_shape,kernel_size=(3,3),activation="relu",strides=(1,1),padding="valid"))
model.add(Conv2D(filters=512,input_shape=image_shape,kernel_size=(3,3),activation="relu",strides=(1,1),padding="valid"))

model.add(Flatten())

model.add(Dense(128,input_shape=(img_row,img_clm,3),activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(65,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(65,activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(65,activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(65,activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(65,activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(num_classes,activation="softmax"))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adadelta(),
metrics=['accuracy'])

model.fit(x_train,y_train,
batch_size=batch_size,
epochs=epoch,
verbose=1,
# validation_split=0.5,
validation_data=(x_test,y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', score[0]) #modelimiz doğruluk ve kayıp oranlarınız yazdırıyoruz
print('Test Accuracy:', score[1])

model_test = model.save('onlyFace128(1-50).h5') # modelimizi kaydediyoruz++








