import numpy as np

print("学号: " + "2107040641")
print("姓名: " + "孟凡祥")

A_ = np.array([[0.9428, 0.3475, -0.8468],
               [0.3475, 1.8423, 0.4759],
               [-0.8468, 0.4759, 1.2147]])

b_ = np.array([0.4127, 1.7321, -0.8621])


def Gaussian(A, b):
    n = A.shape[1]

    for k in range(n):
        if A[k][k] == 0:
            break
        for i in range(k + 1, n):
            c = - A[i][k] / A[k][k]
            for j in range(n):
                A[i][j] += c * A[k][j]

            b[i] += c * b[k]

    x = np.zeros(b.shape)

    for i in range(n - 1, -1, -1):
        if A[i][i] == 0:
            break
        x[i] = b[i]
        for j in range(n - 1, i, - 1):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]

    return A, b, x


def main():
    print("初始A和b如下")
    print("A_:\n", A_)
    print("b_:\n", b_)
    A, b, x = Gaussian(A_, b_)
    print("\n经过高斯消去后的A和b")
    print("A: \n", A)
    print("b: \n", b)
    print("方程的解为")
    print("x:", x)


if __name__ == "__main__":
    main()
