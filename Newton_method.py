import matplotlib.pyplot as plt
import numpy as np

print("学号: " + "2107040641")
print("姓名: " + "孟凡祥")


def f(x):
    return x ** 3 - 2 * x - 5


def df(x):
    return 3 * x ** 2 - 2


def newton(x_k, th):
    k = 0
    iterations = 0
    roots = list()
    roots.append(x_k)

    while abs(f(x_k)) > th:
        x_k = x_k - f(x_k) / df(x_k)
        roots.append(x_k)
        iterations += 1
    return x_k, iterations, roots


def draw(roots):
    fig, ax = plt.subplots(1, 1)

    x = np.linspace(2, 2.5, 30)
    y = f(x)
    ax.plot(x, y)

    y = f(np.array(roots))
    ax.plot(roots, y, ".", color="r")
    ax.grid()

    fig.show()
    fig.savefig("./Newton_method.png")

def main():
    a = float(2)
    b = float(3)
    th = float(0.000000001)
    print("下限为：{}, 上限为: {}， 阈值为：{}".format(a, b, th))

    x, iterations, roots = newton(a, th)
    print("结果为：{}".format(x))
    print("迭代次数为：{}".format(iterations))
    print(roots)
    draw(roots)


if __name__ == "__main__":
    main()
