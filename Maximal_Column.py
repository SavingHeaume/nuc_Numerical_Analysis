from copy import deepcopy
import numpy as np

print("学号: " + "2107040641")
print("姓名: " + "孟凡祥")

A_ = np.array([[0.9428, 0.3475, -0.8468],
               [0.3475, 1.8423, 0.4759],
               [-0.8468, 0.4759, 1.2147]])

b_ = np.array([0.4127, 1.7321, -0.8621])

def get_max_row(a : np.ndarray, j : int):
    max_item = np.abs(a[j][j])
    max_col = j
    n = a.shape[1]
    for i in range(j + 1, n):
        if a[i][j] > max_item:
            max_item = a[i][j]
            max_col = i

    return max_col


def swap_row(i, j, a : np.ndarray, b : np.ndarray):
    if i == j:
        return
    
    temp = deepcopy(a[i])
    a[i] = a[j]
    a[j] = temp
    
    temp = deepcopy(b[i])
    b[i] = b[j]
    b[j] = temp

def Max_col(A : np.ndarray, b : np.ndarray):
    n = A.shape[1]

    for k in range(n):
        max_col = get_max_row(A, k)
        swap_row(k, max_col, A, b)

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
    print("消去前的A和b ")
    print("A_:\n", A_)
    print("b_:\n", b_)
    A, b, x = Max_col(A_, b_)
    print("\n消去后的A和b: ")
    print("A: \n", A)
    print("b: \n", b)
    print("方程得解x: \n", x)


if __name__ == "__main__":
    main()
