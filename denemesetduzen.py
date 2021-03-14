import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import random
#, '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100'
#, '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100'
DATADIR="D:/yeniveriseti/"
CATEGORIES=['1-10', '11-20', '21-30','31-40','41-50']
# DATADIR="D:/yeniveriseti/aa/"
# CATEGORIES=['bb']
IMG_SIZE=128
training_data=[]
def create_training_data():
    sayac=0
    face_cascade = cv2.CascadeClassifier('D:/DerinOgrenme/FaceDetection/haarcascade_frontalface_default.xml')
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):

            try:
                sayac=sayac+1
                img_array = cv2.imread(os.path.join(path, img))
                
               
                # Convert into grayscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                # Draw rectangle around the faces
                for (x, y, w, h) in faces:
                    img_array=cv2.rectangle(img_array, (x, y), (x+w, y+h), (255,0,0) ,1)
                    newimg=img_array[y:y+h,x:x+w]
                # Display the output

                # cv2.imshow('img',newimg)

                # cv2.waitKey()
               
                new_array = cv2.resize(newimg, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
                print(sayac)
            except Exception as e:
                pass
            import numpy as np

create_training_data()


print(len(training_data))
random.shuffle(training_data)

for sample in training_data[:100]:
    print(sample[1])

X=[]
y=[]
for features,label in training_data:
    
    X.append(features)
    y.append(label)

X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
np.save("onlyFaceArray",X)
np.save("onlyFaceLabel",y)