# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from tqdm import tqdm
import time

# directory
img_dir = "Malaria Dataset/Test/Uninfected"
img_classes=["Parasitized","Uninfected"]

img_size = 96
scale_percent = 250

training_data=[]
def create_training_data():        
    for img in os.listdir(img_dir):
        try:
            filename = img
            img_array=cv2.imread(os.path.join(img_dir,img))
            
            width = int(img_array.shape[1] * scale_percent / 100)
            height = int(img_array.shape[0] * scale_percent / 100)
            
            dim = (width, height)
            # resize image
            resized = cv2.resize(img_array, dim, interpolation = cv2.INTER_AREA)
            
            cv2.imwrite(filename, resized)
            time.delay(0.25)
        except Exception as e:
            pass

       
create_training_data()