import os
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RationalQuadratic, Matern, WhiteKernel, ConstantKernel
# 生成数据
X = TargetDomain # Replace with your own data
y = SourceDomain # Replace with your own data

# 训练模型

# 1. 构造组合核：线性趋势 + 多尺度峰 + 局部粗糙 + 白噪声
kernel = (
    ConstantKernel(1.0, (1e-2, 1e2)) * DotProduct(sigma_0=0.0, sigma_0_bounds=(0, 1e2))   # 基线/线性
    + ConstantKernel(1.0, (1e-2, 1e2)) * RationalQuadratic(length_scale=1.0, alpha=1.0,
                                                             length_scale_bounds=(1e-2, 1e2),
                                                             alpha_bounds=(1e-1, 1e2))      # 多尺度
    + ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=3/2,
                                                length_scale_bounds=(1e-2, 1e2))          # 肩峰/陡降
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))                     # 观测噪声
)

model = GaussianProcessRegressor(kernel=kernel)
model.fit(X, y)