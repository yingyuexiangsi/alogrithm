import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

plt.rc('font',family='Times New Roman')
# 设置字体加粗
plt.rcParams['font.weight'] = 'bold'
plt.rcParams.update({'font.size': 15}) # 改变所有字体大小，改变其他性质类似

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RationalQuadratic, Matern, WhiteKernel, ConstantKernel

'''
y = # your collected sample data
X = # ideal sample data
'''


kernel = (
    ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2))  # 线性部分
    + ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5,
                                               length_scale_bounds=(1e-2, 1e2))  # 中等平滑
    + ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(
        length_scale=1.0, alpha=0.5,
        length_scale_bounds=(1e-2, 1e2),
        alpha_bounds=(1e-2, 1e2))  # 多尺度变化
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))  # 噪声
)

model = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-5,  # 可以尝试调整
    n_restarts_optimizer=10,  # 增加优化重启次数以避免局部最优
    normalize_y=True,  # 标准化目标变量
    random_state=42  # 固定随机种子以便复现
)
model.fit(X, y)

print("OK")

# 准备新数据进行预测
X_new = # ideal data

# 进行预测
y_pred, y_std = model.predict(X_new, return_std=True)
# y_pred are data augmentation