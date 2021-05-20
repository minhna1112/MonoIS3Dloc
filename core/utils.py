import pandas as pd
import math


def postprocess(net_output):
    estimation = net_output.numpy()*100
    return estimation

def scale_back(loss):
    return math.sqrt(loss)*100

if __name__ == '__main__':
    print('OK')