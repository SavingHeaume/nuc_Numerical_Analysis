import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print("学号：" + "2107040641")
print("姓名：" + "孟凡祥")


def lagrange(x_:np.ndarray, y_:np.ndarray, xt:np.ndarray):
    yt = np.zeros(xt.shape)
    n = x_.shape[0]
    n_xt = xt.shape[0]
    for item in range(n_xt):
        for i in range(n):
            temp = 1;
            for j in range(n):
                if i == j:
                    continue
                temp = temp * (xt[item] - x_[j]) / (x_[i] - x_[j])
            yt[item] += temp * y_[i]
    
    return yt

def drow(x_, y_, xt:np.ndarray, yt:np.ndarray):
    plt.rcParams["font.sans-serif"]=["SimHei"] 
    plt.rcParams["axes.unicode_minus"]=False 
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title("原始数据点连线")
    ax[0].plot(x_,  y_, 'o--')
    ax[1].set_title("插值图像")
    ax[1].plot(x_, y_, 'o')
    ax[1].plot(xt, yt)

    plt.savefig("./png/Lagrange_interpolation.png")
    plt.show()

def main():
    x_ = np.array([0, 3, 5, 7, 9, 11, 12, 13, 14, 15])
    y_ = np.array([0, 1.2, 1.7, 2.0, 2.1, 2.0, 1.8, 1.2, 1.0, 1.6])

    xt = np.linspace(0, 15, 151)
    
    yt = lagrange(x_, y_, xt)
    drow(x_, y_, xt, yt)

    df = pd.DataFrame(np.vstack((xt, yt)))
    print(df.head())
    

if __name__ == "__main__":
    main()
