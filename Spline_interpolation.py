import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
print("学号：" + "2107040641")
print("姓名：" + "孟凡祥")

def natural_cubic_spline(X, Y):
    n = len(X) - 1
    h = np.diff(X)
    alpha = np.diff(Y) / h

    A = np.zeros((n + 1, n + 1))
    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

    b = np.zeros(n + 1)
    b[1:-1] = 3 * (alpha[1:] - alpha[:-1])

    c_prime = np.linalg.solve(A, b)

    c = np.zeros(n)
    d = np.zeros(n)
    b = alpha.copy()

    for i in range(n):
        c[i] = c_prime[i] / 3
        d[i] = (c_prime[i + 1] - c_prime[i]) / (3 * h[i])

    coefficients = []
    for i in range(n):
        coefficients.append([Y[i], b[i], c[i], d[i]])

    return coefficients

def evaluate_cubic_spline(coefficients, xt, X):
    interpolated_values = np.zeros_like(xt, dtype=float)

    for i, x in enumerate(xt):
        for j in range(len(X) - 1):
            if X[j] <= x <= X[j + 1]:
                h = x - X[j]
                a, b, c, d = coefficients[j]
                interpolated_values[i] = a + b * h + c * h**2 + d * h**3
                break

    return interpolated_values

def drow(x_, y_, xt:np.ndarray, yt:np.ndarray):
    plt.rcParams["font.sans-serif"]=["SimHei"] 
    plt.rcParams["axes.unicode_minus"]=False 
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title("原始数据点连线")
    ax[0].plot(x_,  y_, 'o--')
    ax[1].set_title("插值图像")
    ax[1].plot(x_, y_, 'o')
    ax[1].plot(xt, yt)

    plt.savefig("./png/Spline_interpolation.png")
    plt.show()

def cubic_spiline_by_sci(x_, y_, xt):
    cs = CubicSpline(x_, y_)
    interpolated_values = cs(xt)

    df = pd.DataFrame(np.vstack((xt, interpolated_values)))
    print(df.head())

    fig = plt.figure()
    ax:plt.Axes = fig.add_subplot()
    ax.plot(xt, interpolated_values)
    ax.plot(x_, y_, 'o')
    ax.set_title('sicpy中的CubicSpline')
    plt.savefig('./png/sicpy_cubicSpiline.png')
    plt.show()
    


def main():
    x_ = np.array([0, 3, 5, 7, 9, 11, 12, 13, 14, 15])
    y_ = np.array([0, 1.2, 1.7, 2.0, 2.1, 2.0, 1.8, 1.2, 1.0, 1.6])

    xt = np.linspace(0, 15, 151)
    
    coefficients = natural_cubic_spline(x_, y_)
    yt = evaluate_cubic_spline(coefficients, xt, x_)
    drow(x_, y_, xt, yt)

    df = pd.DataFrame(np.vstack((xt, yt)))
    print(df.head())
    cubic_spiline_by_sci(x_, y_, xt)


if __name__ == "__main__":
    main()
