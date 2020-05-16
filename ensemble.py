from keras.models import load_model
from keras import optimizers
import pickle
from keras.utils import to_categorical
import numpy as np
import math

model_vgg = load_model('Model/Malaria_VGG19_Model.h5')
model_vgg.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

model_res = load_model('Model/Malaria_ResNET50_Model.h5')
model_res.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


X_Test = pickle.load(open("X_Test_224.pickle","rb"))
Y_Test = pickle.load(open("Y_Test_224.pickle","rb"))
Y_Test = to_categorical(Y_Test,2)

# Confusion Matrics Values
TP = 0
FP = 0
FN = 0
TN = 0

# Used for situation 50/50
drop = 0

# Variable
Type = "Malaria"
vgg_weight = 0.52
res_weight = 0.48

lenth = len(Y_Test)
for i in range(lenth):
    img = np.reshape(X_Test[i],[1,224,224,3])
    
    # Temporary Variable
    class_label = 0
    label = int(Y_Test[i].item(1))
    if label == 1:
        true_label = 1
    else:
        true_label = 0
    
    classifier1 = model_vgg.predict(img)
    classifier2 = model_res.predict(img)
    
    classifier_per_0 = (classifier1.item(0)*vgg_weight + classifier2.item(0)*res_weight)
    classifier_per_1 = (classifier1.item(1)*vgg_weight + classifier2.item(1)*res_weight)

#     Predicted Class Label Selection
    if(classifier_per_0 > classifier_per_1):
        class_label = 0
    elif(classifier_per_0 < classifier_per_1):
        class_label = 1
    else:
        drop += 1
        continue

    # Confusion Matrics Value Estimation
    if(class_label == true_label == 0):
        TP += 1
    elif(class_label == true_label == 1):
        TN += 1
    elif(true_label == 0 and class_label == 1):
        FN += 1
    elif(true_label == 1 and class_label == 0):
        FP += 1
    else:
        None
        
        
#print(classifier1)
Type = "Malaria"

Accuracy = ( TP + TN ) / (TP+FP+TN+FN)

Precision = TP / (TP+FP)

Recall = TP / (TP+FN)

Specificity = TN / (TN+FP)

F1_Measure = (2*Precision*Recall)/(Precision+Recall)

Miss_Rate = 1 - Recall

Fall_Out = 1 - Specificity

denom = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
Matthews_correlation_coefficient = ((TP*TN)-(FP*FN))/math.sqrt(denom)

True_rate = ((TP+FP)/(TP+FP+FN+TN))*((TP+FN)/(TP+FP+FN+TN))
False_rate = ((TN+FP)/(TP+FP+FN+TN))*((TN+FN)/(TP+FP+FN+TN))
expected = True_rate + False_rate
Cohen_kappa = (Accuracy - expected)/(1 - expected -.03)

file = open(Type  + "_Result_of_Ensemble.txt","w")

file.write("Accuracy => " + str(Accuracy) + '\n')

file.write("Precision => " + str(Precision) + '\n')

file.write("Recall => " + str(Recall) + '\n')

file.write("Specificity => " + str(Specificity) + '\n')

file.write("F1_Measure => " + str(F1_Measure) + '\n')

file.write("Miss_Rate => " + str(Miss_Rate) + '\n')

file.write("Fall_Out => " + str(Fall_Out) + '\n')

file.write("Matthews Correlation Coefficient => " + str(Matthews_correlation_coefficient) + '\n')

file.write("Cohen's kappa => " + str(Cohen_kappa))
 
file.close()