
#For this program run, you need to use cNoise.py code hosted on RubensSauter github
#I strongly recomend that you comment line 116 and 127 for the version of 02/2022

#!git clone https://github.com/rsautter/Noisy-Complex-Ginzburg-Landau.git

#%cd Noisy-Complex-Ginzburg-Landau/

#import the cNoise function
import cNoise as cNoise
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import cv2

#%cd ..


images_for_classes = 1000

#Generating several colored noise from `cNoise` mudule
#%%capture 
size = 128
noise = []
beta  = []
for j in range(0,images_for_classes):
  for i in range(0,15):
    noise.append(cNoise.cNoise(i/4,(size,size),maxCorrections=100,maxAvgError=0.001, eta=0.05))
    beta.append(i/4)
    print('data_set '+str(j)+' Noise: Beta '+str(i/4))

df_data = pd.DataFrame({'image':noise, 'class':beta})
df_data.to_pickle('MultipleCNN_Noise.plk')

