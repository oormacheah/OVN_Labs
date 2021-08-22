import numpy as np


def ex1():
    array_a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.uint16)
    print(array_a)
    print('array shape:', array_a.shape)
    print('array item_size:', array_a.itemsize)
    print('array dimension:', array_a.ndim)
    print('dtype:', array_a.dtype)


def ex2():
    arr = np.arange(100, 200, 10)
    arr = arr.reshape(5, 2)
    print(arr)


def ex3():
    arr = np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
    print(arr[:, 2])


def ex4():
    arr = np.array([[3, 6, 9, 12], [15, 18, 21, 24], [27, 30, 33, 36], [39, 42, 45, 48], [51, 54, 57, 60]])
    print(arr[::2, 1::2])


def ex5():
    arr1 = np.array([[5, 6, 9], [21, 18, 27]])
    arr2 = np.array([[15, 33, 24], [4, 7, 1]])

    arr3 = arr1 + arr2
    print(arr3)
    for num in np.nditer(arr3, op_flags=['readwrite']):
        num[...] = np.sqrt(num)  # what the fuck [...] (Ellipsis object)
    print(arr3)


def ex6():
    arr = np.array([[34, 43, 73], [82, 22, 12], [53, 94, 66]])
    new_arr = arr.reshape(1, np.prod(arr.shape))
    new_arr.sort()
    new_arr = new_arr.reshape(arr.shape)
    print(new_arr)


def ex7():
    arr = np.array([[34, 43, 73], [82, 22, 12], [53, 94, 66]])
    print(arr)
    print(np.amax(arr, 0))
    print(np.amin(arr, 1))


def ex8():
    arr = np.array([[34, 43, 73], [82, 22, 12], [53, 94, 66]])
    print(arr)
    new_column = [10, 10, 10]
    arr[:, 1] = new_column
    print(arr)


ex8()
