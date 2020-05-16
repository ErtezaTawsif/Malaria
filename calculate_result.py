# -*- coding: utf-8 -*-

import math

TP = 2051
FP = 49
FN = 94
TN = 2006
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

file = open(Type  + "_Result_of_VGG_Model.txt","w")

file.write("Confusion Matrix:\nTP => " + str(TP) + "  |  FP => " + str(FP) + "\nFN => " + str(FN) + "  |  TN => " + str(TN) +'\n\n')

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