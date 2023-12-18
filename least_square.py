import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

print("学号: " + "2107040641")
print("姓名: " + "孟凡祥")

def normal_equation(x_:np.ndarray, y_:np.ndarray, functions):
    A = np.vstack([func(x_) for func in functions]).T

    G = np.dot(A.T, A)
    b = np.dot(A.T, y_)

    L = cholesky(G, lower=True)

    y1 = np.linalg.solve(L, b)
    
    x_fit = np.linalg.solve(L.T, y1)

    return x_fit

def linear_function(x):
    return np.vstack([x, np.ones_like(x)])

def drow(x_, y_, fit_coefficients):
    fig = plt.figure()
    ax:plt.Axes = fig.add_subplot()

    ax.plot(x_, y_, 'o')
    
    x_new = np.linspace(10, 20, 101)
    y_new = np.vstack([x_new, np.ones_like(x_new)]).T @ fit_coefficients
    ax.plot(x_new, y_new)

    plt.savefig("./png/least_square.png")

    plt.show()

def main():
    x_ = np.array([10, 11, 12, 13, 14, 15, 16])
    y_ = np.array([70, 122, 144, 152, 174, 196, 202])

    fig = plt.figure()
    ax : plt.Axes = fig.add_subplot()

    ax.plot(x_, y_, "o")
    plt.savefig("./png/least_square_original.png")
    plt.show()
    
    print("设函数为 y = a * x + b")
    
    fit_coefficients = normal_equation(x_, y_, [linear_function])
    
    print("参数为", fit_coefficients)
    print("拟合的函数为 y = {} * x + {}".format(fit_coefficients[0], fit_coefficients[1]))

    x_new = np.array([17, 18])
    #x_new = np.vstack([x_new, np.ones_like(x_new)]).T
    X_new = np.vstack((x_new, np.ones_like(x_new))).T
    y_new = X_new @ fit_coefficients

    df = pd.DataFrame(np.vstack((x_new, y_new)))
    print("结果为：\n", df)

    drow(x_, y_, fit_coefficients)
   
if __name__ == "__main__":
    main()
