from sklearn.datasets import load_digits 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import numpy as np
import time
import sys
from joblib import dump, load 
import os 
import torch 
import torch.nn as nn
import pdb 


#NO model quantization, Noi data quantization
#Load data 
digits= load_digits() 
X,y = digits.data ,digits.target 

print(f"Max value: {X.max()}, Min value: {X.min()}")

#Split data 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


model =LogisticRegression(max_iter=1000, solver='lbfgs') 
model.fit(X_train, y_train)

#Accuracy 

y_pred = model.predict(X_test)
accuracy= (y_pred==y_test).mean()

#Inference

y_pred= model.predict(X_test)

acc=(y_pred==y_test).mean() 

model_name="unquantised_model_unquantiused_data.joblib" 
dump(model, model_name) 
saved_size= os.path.getsize(model_name)

print(f"Unquantised Model, Unquantised Data Acc={acc*100}, saved size={saved_size/1024} KB")

print("======================================================================")