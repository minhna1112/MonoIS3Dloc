import pandas as pd
import math
import cv2
import numpy as np

def smart_resize(image, target_size):

    ih, iw ,   = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    return image_resized

def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    #shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                             border_mode, value=value)

    #return img, margin
    return img

def resize_and_pad(img, target_shape=(1,398, 224)):
    img = smart_resize(img, target_shape)
    img = pad_image_to_shape(img, target_shape, border_mode=cv2.BORDER_CONSTANT, cval=-1.0)
    #assert img.shape[0] == img.shape[0]
    return img

def zero_centerred(img):
    img = np.asarray(img, dtype=np.float32) -127.5
    img = img / 127.5

    return img

def postprocess(net_output):
    estimation = net_output.numpy()*100
    return estimation

def scale_back(loss):
    return math.sqrt(loss)*100

if __name__ == '__main__':
    print('OK')