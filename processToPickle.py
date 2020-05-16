# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from tqdm import tqdm

# directory
img_dir = "Malaria Dataset[High]/Train"
img_classes=["Parasitized","Uninfected"]

img_size = 224

training_data=[]
def create_training_data():
 
    for clas in tqdm(img_classes):
            path=os.path.join(img_dir,clas)
            #index no start from 0 1 2 3
            print(clas)
            
            class_num=img_classes.index(clas)
            print(class_num)
            
            for img in os.listdir(path):
                try:
                    img_array=cv2.imread(os.path.join(path,img))
                    new_array=cv2.resize(img_array,(img_size,img_size))
                    training_data.append([new_array,class_num])
                except Exception as e:
                    pass
    
       
create_training_data()
print(len(training_data))

xs=[]
ys=[]

for features,labels in tqdm(training_data):
    xs.append(features)
    ys.append(labels)

xs=np.array(xs).reshape(-1,img_size,img_size,3)
ys=np.array(ys).reshape(-1,1)

import pickle
pickle_out=open("X_Train_224.pickle","wb")
pickle.dump(xs,pickle_out)
pickle_out.close()

pickle_out=open("Y_Train_224.pickle","wb")
pickle.dump(ys,pickle_out)
pickle_out.close()