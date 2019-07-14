#%% Imports
import warnings
warnings.simplefilter('ignore')
!pip install --user  --upgrade pip
!pip install --user  numpy
!pip install --user  scipy
!pip install --user  scikit-learn
!pip install --user  matplotlib
!pip install --user  pandas
!pip install --user tensorflow
!pip install --user  sklearn
!pip install --user  pyprind
!pip install --user  keras
#%% Checking tensor version and gpu
from tensorflow.python.client import device_lib
import tensorflow as tf
print(tf.__version__)
device_lib.list_local_devices()

#%%
