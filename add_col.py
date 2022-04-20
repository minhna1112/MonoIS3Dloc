import pandas as pd
import numpy as np

df = pd.read_csv('~/25_val.csv', sep=',')

df['derived_label'] = np.zeros(len(df['img'], ), dtype=np.int8)
df['derived_label'].loc[(df['x'] > 10) & (df['x'] <=20)] = 1
df['derived_label'].loc[df['x'] > 20] = 2
df.to_csv('~/25_val_new.csv')