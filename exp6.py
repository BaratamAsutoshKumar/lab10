
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

#Pytorch model quantization, qunatized data
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



model =LogisticRegression(max_iter=1000, solver='lbfgs') 
model.fit(X_train_q, y_train)


#copying the learnede weights and biases 
weights=np.asarray(model.coef_,dtype=np.float32)
bias=np.asarray(model.intercept_,dtype=np.float32) 


num_classes, num_features= weights.shape

class LogisticRegressionTorch(nn.Module):
    def __init__(self, num_features,num_classes):
        super().__init__() 
        self.linear=nn.Linear(num_features, num_classes)
    def forward(self,x):
        return self.linear(x)


pt_model=LogisticRegressionTorch(num_features, num_classes)
pt_model.linear.weight.data= torch.from_numpy(weights)
pt_model.linear.bias.data= torch.from_numpy(bias) 
#pdb.set_trace()

pt_model.eval()
torch.backends.quantized.engine = "qnnpack"

quantized_model = torch.quantization.quantize_dynamic(
    pt_model, {nn.Linear}, dtype= torch.qint8
)

#inference
X_input = torch.from_numpy(X_test_q.astype(np.float32))

outputs= quantized_model(X_input) 

y_pred= torch.argmax(outputs, axis=1).numpy() 
acc=(y_pred==y_test).mean()

model_name= "torch_unquantized_model_weights_quantized_normalized_int_data.joblib"
dump(pt_model,model_name)
saved_size= os.path.getsize(model_name)

model_name= "torch_quantized_model_weights_quantized_normalized_int_data.joblib"
dump(quantized_model, model_name) 
saved_size_quant= os.path.getsize(model_name)





print(f"Quantised Model, Normalised Quantized Data Acc={acc*100}, saved size={saved_size/1024} KB,\nUnquantized model size:{saved_size/1024}KB ,saved Qunatized model size={saved_size_quant/1024}KB")
#pdb.set_trace()
print("======================================================================")