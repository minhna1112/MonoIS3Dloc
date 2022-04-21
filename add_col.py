import pandas as pd
import numpy as np
from path import Path

train_path = "./train.csv"
val_path = "./val.csv"
test_path = "./test.csv"


def deriving_label(path):
    df = pd.read_csv(path, sep=',')

    df['derived_label'] = np.zeros(len(df['img'], ), dtype=np.int8)
    df['derived_label'].loc[(df['x'] > 10) & (df['x'] <=20)] = 1
    df['derived_label'].loc[df['x'] > 20] = 2
    df.to_csv(path[:-4]+'_new.csv')


deriving_label(train_path)
deriving_label(val_path)
deriving_label(test_path)
