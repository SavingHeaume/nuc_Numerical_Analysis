import matplotlib.pyplot as plt
import numpy as np

print("学号: " + "2107040641")
print("姓名: " + "孟凡祥")


def f(x):
    return x ** 3 - 2 * x - 5


def dichotomy(a, b, th):
    iterations = 0
    x = 0
    roots = list()
    while (b - a) > th:
        x = a + (b - a) / 2
        roots.append(x)
        if f(x) == 0:
            return x
        elif f(a) * f(x) > 0:
            a = x
        elif f(a) * f(b) < 0:
            b = x
        iterations += 1
    return x, roots, iterations


def draw(a, b, roots):
    fig, ax = plt.subplots(1, 1)
    # ax = fig.add_axes([0, 0, 1, 1])

    x = np.linspace(a, b, 30)
    y = f(x)

    ax.plot(x, y)
    ax.grid()

    y = f(np.array(roots))
    ax.plot(roots, y, ".", color="r")
    plt.show()
    plt.savefig("./png/dichotomy.png")

def main():
    a = float(2)
    b = float(3)
    th = float(0.00000001)
    print("下限为：{}， 上限为：{}， 阈值为：{}".format(a, b, th))

    x, roots, iterations = dichotomy(a, b, th)
    print("结果为：{}".format(x))
    print("迭代次数为：{}".format(iterations))
    print(roots)
    draw(a, b, roots)


if __name__ == "__main__":
    main()
