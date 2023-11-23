import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print("学号：" + "2107040641")
print("姓名：" + "孟凡祥")

def get_diff(X, Y):
    n = len(X)
    A = np.zeros([n, n])
    for i in range(0, n):
        A[i][0] = Y[i]
    for j in range(1, n):
        for i in range(j, n):
            A[i][j] = (A[i][j - 1] - A[i - 1][j - 1]) / (X[i] - X[i - j])
    return A

def newton_(X, Y, x):
    sum = Y[0]
    temp = np.zeros((len(X), len(X)))
    # 将第一行赋值
    for i in range(0, len(X)):
        temp[i, 0] = Y[i]
    temp_sum = 1.0
    for i in range(1, len(X)):
        # x的多项式
        temp_sum = temp_sum * (x - X[i - 1])
        # 计算均差
        for j in range(i, len(X)):
            temp[j, i] = (temp[j, i - 1] - temp[j - 1, i - 1]) / (X[j] - X[j - i])
        sum += temp_sum * temp[i, i]
    return sum

def newton(xt:np.ndarray, A, x_, y_):
    n = len(x_)
    result = np.zeros_like(xt, dtype=float)

    for k, x in enumerate(xt):
        sum = y_[0]
        temp_sum = 1.0

        for i in range(1, n):
            temp_sum *= (x - x_[i - 1])
            sum += temp_sum * A[i, i]

        result[k] = sum

    return result

def main():
    x_ = np.array([0, 3, 5, 7, 9, 11, 12, 13, 14, 15])
    y_ = np.array([0, 1.2, 1.7, 2.0, 2.1, 2.0, 1.8, 1.2, 1.0, 1.6])

    xt = np.linspace(0, 15, 151)
    
    #yt = newton(x_, y_, xt)
    #print(xt)
    #print(yt)
    A = get_diff(x_, y_)
    yt = newton(xt, A, x_, y_)
    df = pd.DataFrame(np.vstack((xt, yt)))
    print(df.head(5))

    plt.plot(xt, yt)
    plt.savefig("./.png/Newton_interpolation.png")
    plt.show()
    

if __name__ == "__main__":
    main()
