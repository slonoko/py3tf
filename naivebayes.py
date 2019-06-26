#%% loading training data

import pandas as pd
import numpy as np
from sklearn import datasets
from collections import defaultdict
from tempfile import NamedTemporaryFile
import os

X, y = datasets.load_wine(True)

__classes__ = {"c1":{}, "c2":{}, "c3":{}}
__temp_cls__=NamedTemporaryFile(dir='.', suffix=".cls")

for i, c in enumerate(X):
    for item in c:
        _cls = "c1" if i<60 else "c2" if 60<=i<120 else "c3"
        if item in __classes__[_cls]:
            __classes__[_cls][item] +=1
        else:
            __classes__[_cls][item] = 1
np.savez_compressed(__temp_cls__, __classes__)

#%% General probabilities
c_c1 = len(__classes__["c1"])
c_c2 = len(__classes__["c2"])
c_c3 = len(__classes__["c3"])

p_c1 = c_c1 / (c_c1 + c_c2 + c_c3)
p_c2 = c_c2 / (c_c1 + c_c2 + c_c3)
p_c3 = c_c3 / (c_c1 + c_c2 + c_c3)

print(f"P(c1) = {p_c1 * 100}")
print(f"P(c2) = {p_c2 * 100}")
print(f"P(c3) = {p_c3 * 100}")
#%%
def probWrtCls(k, arr, cls):
    denominator =0
    for a in arr:    
        if a in __classes__[cls]:
            denominator +=__classes__[cls][a]+1
        else:
            denominator+=1
    if k in __classes__[cls]:
        nominator = __classes__[cls][k]+1
    else:
        nominator = 1
    return nominator/denominator

#%%
def calcNB(arr):
    p_k_c1 = 1
    p_k_c2 = 1
    p_k_c3 = 1
    for k in arr:
        p_k_c1 *= probWrtCls(k, arr, "c1")
        p_k_c2 *= probWrtCls(k, arr, "c2")
        p_k_c3 *= probWrtCls(k, arr, "c3")
    p_arr_c1 = p_c1*p_k_c1
    p_arr_c2 = p_c2*p_k_c2
    p_arr_c3 = p_c3*p_k_c3

    return p_arr_c1, p_arr_c2, p_arr_c3 