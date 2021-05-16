import matplotlib.pyplot as plt
import pandas as pd

def normalize_output(df):
    df['x'] = df['x'].div(100)
    df['y'] = df['y'].div(100)
    df['z'] = df['z'].div(100)

    return df

def postprocess(net_output):
    estimation = net_output.numpy()*100
    return estimation

def plot_learning_curve(log_path):
    losses = pd.read_csv(log_path, delimiter=',')
    loss = losses['loss']
    val_loss = losses['val_loss']
    val_rel = losses['val_relative_error']
    fig  = plt.figure()
    plt.plot(range(len(loss)), [l for l in loss], 'b')
    plt.plot(range(len(val_loss)), [l for l in val_loss],'r')
    plt.plot(range(len(loss)), [l for l in val_rel], 'g')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    print('OK')