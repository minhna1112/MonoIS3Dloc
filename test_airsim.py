import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import math
def euclidean_distance_loss(y_true, y_pred):
      return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
#model.load_weights("/content/drive/MyDrive/Project3/Airsim_project/training/cp-0005.ckpt")
CSV_RESULT_PATH="/home/ivsr/CV_Group/phuc/airsim/result_test_epoch14.csv"
IMG_VAL_FOLDER_PATH="/home/ivsr/CV_Group/phuc/airsim/data"
model=tf.keras.models.load_model("/home/ivsr/CV_Group/phuc/airsim/training/cp-0014.ckpt",custom_objects={"euclidean_distance_loss":euclidean_distance_loss})
df=pd.read_csv("/home/ivsr/CV_Group/phuc/airsim/test.csv")
IMG=df.img

B=[]
for i in range(len(IMG)-1):
    img_path=os.path.join(IMG_VAL_FOLDER_PATH,IMG[i+1])
    img=tf.keras.preprocessing.image.load_img(img_path,color_mode='grayscale',target_size=(398,224))
    input_arr=tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr]) 
    input_arr=input_arr/255
    predictions=model.predict(input_arr)  
    B.append(predictions*30)
B=np.array(B)
a=B.shape
C=B.reshape(a[0],3)
print(C)

dictresults={'x_pred':C.T[0],'y_pred':C.T[1],'z_pred':C.T[2]}


D=np.sqrt(df.x**2+df.y**2+df.z**2)
abs_err = np.sqrt((df.x-C.T[0])**2+(df.y-C.T[1])**2+(df.z-C.T[2])**2)
pro=abs_err/D
dictresult={'img':df.img,'x_label':df.x,'y_label':df.y,'z_label':df.z,'x_pred':C.T[0],'y_pred':C.T[1],'z_pred':C.T[2],"real_distance":D,"abs_err":abs_err, 'prop_err':pro}
df_predict = pd.DataFrame(dictresults)

print(df_predict)
df_predict.to_csv(CSV_RESULT_PATH)
print("DONE!")