import numpy as np

def OP(reflectivity2):
    # 指定需要累加的索引
    indices = [26, 27, 28, 29, 30,
               35, 36, 37, 38, 39,
               113, 114, 115, 116, 117, 118,
               122, 123, 124, 125, 126, 127,
               212, 213, 214, 215, 216,
               221, 222, 223, 224, 225, 226, 227]# Replace with your own characteristics
    # 初始化一个空数组，用于存储累加结果
    sum_array = np.zeros(len(reflectivity2))

    # 对每一行的指定索引值进行累加
    for i in range(len(reflectivity2)):
        sum_array[i] = np.sum(reflectivity2[i, indices])

    return sum_array


def ConcentrationInversion(sum_array, target3):
    C = []
    for i in range(0, len(target3)):
        c = 3.01868628e+00 * sum_array[i] - 5.86292652e-04 # Replace with your own concentration formula
        C.append(c)
    return C