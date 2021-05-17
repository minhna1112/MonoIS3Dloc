import numpy as np
import cv2
import tensorflow as tf
import pandas as pd

val_data = pd.read_csv('./data/val.csv')

def postprocess(net_output):
    estimation = net_output.numpy()*100
    return estimation

def err_cal(pred, true):
    x_loss = np.square(pred[:,0] - true[:,0])
    y_loss = np.square(pred[:, 1] - true[:, 1])
    z_loss = np.square(pred[:, 2] - true[:, 2])

    rmse = (x_loss +y_loss +z_loss)**0.5

    return rmse

def estimate(image_path, input_size=224):

    original_image = cv2.imread(image_path, 0)

    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    # image_data = image_data[np.newaxis, ...].astype(np.float32)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    label = np.array(val_data.loc[val_data['img']==image_path.split('/')[-1], ['x', 'y','z']])

    model = tf.keras.models.load_model('./weights/baseline_2204_simp')
    net_output = model(images_data)
    out = postprocess(net_output)
    rmse = err_cal(out, label)
    print(rmse)
if __name__ == '__main__':
    im_path = './data/val/img_0_5_1617359480312880000.png'
    estimate(im_path)