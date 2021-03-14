import cv2
from cv2 import cv2
import keras

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as k
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

def getImgFromFile(filePath):
    img=cv2.imread("%s"%filePath)
    return img
num_classes=5
img_row=128
img_clm=128
model=Sequential()
image_shape=(128,128,3)

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

model.load_weights("D:/DerinOgrenme/alexnet_128(1-50).h5")


input=np.array([]) #boş array tanımlıyoruz


# Load the cascade

face_cascade = cv2.CascadeClassifier('D:/DerinOgrenme/FaceDetection/haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('D:/DerinOgrenme/FaceDetection/ikiyuz.jpg')
resizedImg=cv2.resize(img,(255,255))
img=resizedImg
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    img=cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    newimg=img[y:y+h,x:x+w]
# Display the output

cv2.imshow("a",newimg)
cv2.waitKey()




# file='D:/DerinOgrenme/FaceDetection/besyas.jpg' #verimizi alıyoruz
# img=getImgFromFile(file) #verimizi okuyoruz fonksiyon aracılığıyla
# resizedImg=cv2.resize(newimg,(128,128))#resize ediyoruz görselimizi
img=cv2.resize(newimg,(128,128))
# input=np.append(input,resizedImg) #sonra arraya ekliyoruz
# input=np.reshape(input,(-1,128,128,3)) 
# img_path = 'D:/archive (1)/age_prediction_up/age_prediction/test/033/1576.jpg'
# img = image.load_img(img_path, target_size=(128, 128))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# # x = preprocess_input(x)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(model.predict(x))



#print('Tahminler:', decode_predictions(tahmin,top=3)[0])
# print(model.predict(x)) #test ediyoruz görseli ve sonuları yazdırıyoruz
