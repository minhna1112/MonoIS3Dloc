import pandas as pd
import numpy

label_dir = "/home/dung/KITTI/airsim_kitti/2022-11-15-19-56-40/Cordinate_Noformat.csv"

df = pd.read_csv(label_dir, sep=',')
print(len(df['x']))
# print(df['x'])
derived_data = []
for i in range(len(df)):
    t_x = df.loc[i,'x']
    t_x = -t_x
    # print(t_x)
    if t_x <= 10:
        derived = 0
    if t_x >10 and t_x <= 20:
        print('ok')
        derived = 1
    if t_x > 20:
        print('ok')
        derived = 2
    # print(derived)
    derived_data.append(derived)
    
df['derived'] = derived_data
print(df)
df.to_csv('/home/dung/KITTI/airsim_kitti/2022-11-15-19-56-40/derived_label.csv')