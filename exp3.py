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

#No model quantization, Data quantization
#Load data 
digits= load_digits() 
X,y = digits.data ,digits.target 
X=(X-X.min())/(X.max()-X.min())  # Normalize data to [0,1]
print(f"Max value: {X.max()}, Min value: {X.min()}")

#Split data 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


def quantize_input_reg(norm_inp,n_bits=3):
    levels= 2**n_bits-1 
    val = np.round(norm_inp*levels).astype(np.uint8)
    return val 

X_train_q= quantize_input_reg(X_train, n_bits=3) 
X_test_q= quantize_input_reg(X_test, n_bits=3) 

import pickle 

pickle.dump(X_train, open("X_train.pkl", "wb"))
pickle.dump(X_train_q, open("X_train_q.pkl", "wb"))

saved_size_X_train= os.path.getsize("X_train.pkl")
saved_size_X_train_q= os.path.getsize("X_train_q.pkl")

print(f"Max value: {X_train.max()}, Min value: {X_train.min()}",X_train[0][:5])
print(f"Max Q value: {X_train_q.max()}, Min Q value: {X_train_q.min()}", X_train_q[0][:5])

print(f"====> Saved X_train {saved_size_X_train/1024} KB, X_train_q {saved_size_X_train_q/1024} KB")

model =LogisticRegression(max_iter=1000, solver='lbfgs') 
model.fit(X_train_q, y_train)

#Accuracy 

y_pred = model.predict(X_test_q)
accuracy= (y_pred==y_test).mean()

#Inference

y_pred= model.predict(X_test_q)

acc=(y_pred==y_test).mean() 

model_name="unquantised_model_normalised_data.joblib" 
dump(model, model_name) 
saved_size= os.path.getsize(model_name)

print(f"Unquantised Model, Normalised Quantized Data Acc={acc*100}, saved size={saved_size/1024} KB")

print("======================================================================")