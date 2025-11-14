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


def quantize_model(inp, n_bits=3):
    levels= 2**n_bits-1 
    val = np.round(inp*levels).astype(np.uint8)
    return val

def dequantize_model(inp, n_bits=3):
    levels= 2**n_bits-1 
    val = inp.astype(np.float64)/levels
    return val

model =LogisticRegression(max_iter=1000, solver='lbfgs') 
model.fit(X_train_q, y_train)


#weights
weights=np.array(model.coef_)
bias= np.array(model.intercept_)

w_min= weights.min()
w_max= weights.max()
b_min=bias.min()
b_max=bias.max() 

#normalization of weights
weights_norm=(weights- w_min)/(w_max - w_min)
bias_norm=(bias - b_min)/(b_max - b_min) 


#quantization of weights
quantized_weights= quantize_model(weights_norm, n_bits=4)
quantized_bias= quantize_model(bias_norm, n_bits=4) 

model_name= "unquantized_model_weights_quantized_normalized_int_data.joblib"
model_wt_bias={} 
model_wt_bias["weights"]= weights 
model_wt_bias["bias"]=bias 
dump(model_wt_bias,model_name)
saved_size= os.path.getsize(model_name)

model_name= "quantized_model_weights_quantized_normalized_int_data.joblib"
model_wt_bias={}
model_wt_bias["weights"]= quantized_weights
model_wt_bias["bias"]= quantized_bias 
dump(model_wt_bias, model_name) 
saved_size_quant= os.path.getsize(model_name)

#Accuracy 
weights_rec=dequantize_model(quantized_weights, n_bits=4)*(w_max - w_min) + w_min
bias_rec= dequantize_model(quantized_bias, n_bits=4)*(b_max - b_min) + b_min
outputs= np.dot(X_test_q, weights_rec.T) + bias_rec

y_pred= np.argmax(outputs, axis=1)

acc= (y_pred==y_test).mean()



print(f"Quantised Model, Normalised Quantized Data Acc={acc*100}, saved size={saved_size/1024} KB,\nUnquantized model size:{saved_size/1024}KB ,saved Qunatized model size={saved_size_quant/1024}KB")

print("======================================================================")