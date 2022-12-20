from re import A
import time
import random
from uuid import RESERVED_FUTURE
from webbrowser import get
# import tensorflow as tf
import pandas as pd
from tqdm import tqdm
# from path import Path
import numpy as np
import argparse
import os
import warnings
import math

# from keras_preprocessing.image import validate_filename

# parser = argparse.ArgumentParser(description='Select between small or big data',
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument('-d', '--data-size', type=str, choices=['big', 'small', 'real'], default='small')
def get_iou(pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    
    ixmin = np.maximum(pred_boxes[0], gt_box[0])
    ixmax = np.minimum(pred_boxes[2], gt_box[2])
    iymin = np.maximum(pred_boxes[1], gt_box[1])
    iymax = np.minimum(pred_boxes[3], gt_box[3])

    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
    inters = iw * ih

    # 3.calculate the area of union
    uni = ((pred_boxes[2] - pred_boxes[0] + 1.) * (pred_boxes[3] - pred_boxes[1] + 1.) +
        (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
        inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
    iou = inters / uni
    # iou_max = np.max(iou)
    # nmax = np.argmax(iou)
    return iou
count = 0
# Predict folder (txt file with bbox info)
predict_path = "/home/dung/KITTI/ALP/yolov7_fake/yolov7/predict_0511_conf=0.5"
# label_2 folder 
# label_path = "/home/dung/KITTI/data_object_label_2/training/gt_after_change_new"
label_path = "/home/dung/KITTI/data_object_label_2/training/label_2"
# def check_iou(predict_path, label_path):
#     count = 0
#     for predict in os.listdir(predict_path):
#         det_df = pd.read_csv(predict_path,sep=" ",header= 0)
#         print(det_df.shape[0])
#         for gt in os.listdir(label_path):
#             if gt == predict:
#         prin()
#     return def_df
        

# for img_name in os.listdir("/home/dung/KITTI/ALP/yolov7_fake/yolov7/seg"):
#     image_name = img_name[3:9]+'.txt'
#     print(image_name)
#     # print(type(image_name))
#     for gt in os.listdir("/home/dung/KITTI/data_object_label_2/training/label_2"):
#         if gt == image_name:
#             count +=1

#     print(count)
max_iou = []
instance_list = []
iou_count =0 
for predict in os.listdir(predict_path):
    det_path = f"{predict_path}/{predict}"
    det_df = pd.read_csv(det_path,sep=" ",header= None)
    pred_box = np.zeros((1,4))
    pose = np.zeros((len(det_df),1))
    det_df['name_file'] = det_df[6]
    det_df['score'] = det_df[5]
    det_df['bbox_Xtl'] = det_df[1]
    det_df['bbox_Ytl'] = det_df[2]
    det_df['bbox_Xbr'] = det_df[3]
    det_df['bbox_Ybr'] = det_df[4]
    det_df.insert(5,'Localization',pose)
    # det_df = det_df.values.tolist()
    
    
    # print(len(det_df['bbox_Xtl']))
    # print(type(pred_box))

    # print(len(det_df['name_file']))
    
    for gt in os.listdir(label_path):
        label_file_dir = f"{label_path}/{gt}"
        if gt == predict:
            # change name label
            # gt_df = pd.read_csv(label_file_dir,sep=" ",header= None)
            # gt_df['bbox_Xtl'] = gt_df[3]
            # gt_df['bbox_Ytl'] = gt_df[4]
            # gt_df['bbox_Xbr'] = gt_df[5]
            # gt_df['bbox_Ybr'] = gt_df[6]
            # gt_df['location_x'] = gt_df[7]
            # gt_df['location_y'] = gt_df[8]
            # gt_df['location_z'] = gt_df[9]
            # gt_df['class'] = gt_df[0]

            # original label
            gt_df = pd.read_csv(label_file_dir,sep=" ",header= None)
            gt_df['bbox_Xtl'] = gt_df[4]
            gt_df['bbox_Ytl'] = gt_df[5]
            gt_df['bbox_Xbr'] = gt_df[6]
            gt_df['bbox_Ybr'] = gt_df[7]
            gt_df['location_x'] = gt_df[11]
            gt_df['location_y'] = gt_df[12]
            gt_df['location_z'] = gt_df[13]
            gt_df['class'] = gt_df[0]
            
            for i in range(gt_df.shape[0]):
                IoU =[]
                
                save = False
                class_name = gt_df.loc[i,'class']
                distance2cam = 0
                # if class_name == 'Person':
                if class_name == 'Pedestrian':
                    x_l1 = gt_df.loc[i,'bbox_Xtl']
                    y_l1 = gt_df.loc[i,'bbox_Ytl']
                    x_l2 = gt_df.loc[i,'bbox_Xbr']
                    y_l2 = gt_df.loc[i,'bbox_Ybr']
                    x_gt =  gt_df.loc[i,'location_x']
                    y_gt =  gt_df.loc[i,'location_y']
                    z_gt =  gt_df.loc[i,'location_z']
                    gt_box = [x_l1,y_l1,x_l2,y_l2]
                    
                    
                    # print(f'gt_box = {gt_box}')
                    for j in range(det_df.shape[0]):
                        # print(class_name)
                        # print('true')
                        x1 = det_df.loc[j,'bbox_Xtl']
                        y1 = det_df.loc[j,'bbox_Ytl']
                        x2 = det_df.loc[j,'bbox_Xbr']
                        y2 = det_df.loc[j,'bbox_Ybr']
                        pred_box = [x1,y1,x2,y2]
                        print(f'pred_box = {pred_box}')
                        print(f'gt_box = {gt_box}')
                        iou = get_iou(pred_box, gt_box)
                        # print(f"IOU = {iou}")
                        IoU.append(iou)
                        if iou > 0.5:
                            save = True
                            iou_count += 1 
                        # print(max(IoU))
                        

                    if IoU and save:
                        distance2cam = math.sqrt(x_gt**2 + y_gt**2 + z_gt**2)
                        IoU = np.array(IoU)
                        index = IoU.argmax()
                        # max_iou.append(IoU[index])
                        det_df.loc[index,'Localization'] = distance2cam
                        
                        name_file = det_df.loc[index,'name_file']
                        path_txt_gt = f'/home/dung/KITTI/ALP/yolov7_fake/yolov7/predict_with_IOU_pedestrian_conf=0.5/{predict}'
                        # print(f'name file{name_file}')
                        # print(type(name_file))
                        # print(instance_list)
                        if instance_list:
                            print(len(instance_list))
                            if f"{name_file}" in instance_list:
                                # print(f'{name_file}')
                                print("------------Skip_gt-------------")
                                continue
                            
                        f = open(path_txt_gt, "a+")
                        instance_list.append(name_file)
                        class_name = "Pedestrian"
                        max_iou = IoU[index]
                        score = det_df.loc[index,'score']
                        x_min = det_df.loc[index,'bbox_Xtl']
                        y_min = det_df.loc[index,'bbox_Ytl']
                        x_max = det_df.loc[index,'bbox_Xbr']
                        y_max = det_df.loc[index,'bbox_Ybr']
                        # print(f'x_gt = {x_gt} y_gt = {y_gt} z_gt = {z_gt}')
                        # print(f'distance{distance2cam}')
                        
                        # print(f"instance{name_file}")

    
                        f.write("{} {} {} {} {} {} {} {} {} {} {} {}\n".format(class_name, x_min, y_min, x_max, y_max, x_gt, y_gt, z_gt, distance2cam,score,max_iou,name_file))
                print('----------next_gt--------')
print(iou_count)
# print(max_iou)
# print(len(max_iou))
        # print(len(pred_box))
# print(path_txt_gt)
# f = open(path_txt_gt, "a+")
# f.write("{} {} {} {} {} {} {} {} {} {}\n".format(Cla, truncated, occluded, x_min, y_min, x_max, y_max, x, y, z))
# f.close() 
                
# x_min = float((df.loc[df.image_name == name][1]))
# y_min = float(df.loc[df.image_name == name][2])
# x_max = float(df.loc[df.image_name == name][3])
# y_max = float(df.loc[df.image_name == name][4])
# x = float(df.loc[df.image_name == name]['x'])
# y = float(df.loc[df.image_name == name]['y'])
# z = float(df.loc[df.image_name == name]['z'])
# score = float(df.loc[df.image_name == name]['score'])
    # prin()