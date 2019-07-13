#%% Imports
import warnings
warnings.simplefilter('ignore')
!pip install  --upgrade pip
!pip install  numpy
!pip install  scipy
!pip install  scikit-learn
!pip install  matplotlib
!pip install  pandas
!pip install  tensorflow-gpu
!pip install  sklearn
!pip install  pyprind

#%% Checking tensor version and gpu
from tensorflow.python.client import device_lib
import tensorflow as tf
print(tf.__version__)
device_lib.list_local_devices()

#%%
