import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Flatten, Dense,
                                     Bidirectional, LSTM, Lambda, Concatenate,
                                     Reshape, GlobalAveragePooling1D, Multiply)
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import math

# -------------------------------------------------
# ECA-Net 1D 通道注意力（零参数版）
# -------------------------------------------------
def eca_block_1d(x, gamma=2., b=1):
    """
    x: (B, T, C)  ——  T 为时间步，C 为通道
    return: (B, T, C) 加权后特征
    """
    print('x.shape',x.shape)
    C = int(x.shape[-1])
    print('C',C)
    # 自适应卷积核大小
    k = int(abs((math.log(C, 2) + b) / gamma))
    k = k if k % 2 else k + 1          # 必须为 odd
    print('k',k)
    # 全局平均 -> (B, C)
    gap = GlobalAveragePooling1D()(x)  # (B, C)
    print('GlobalAveragePooling1D.shape',gap.shape)
#     gap = K.expand_dims(gap, axis=-2)  # (B, 1, C)
    # 2) 扩维  (B, C) -> (B, 1, C)   【用 Lambda 层包装】
    gap = Lambda(lambda z: tf.expand_dims(z, axis=-2))(gap)
    # 1D 局部跨通道交互（权重共享）
    weight = Conv1D(1, kernel_size=k, padding='same', use_bias=False)(gap)
    print('weight.shape',weight.shape)
#     weight = K.sigmoid(weight)         # (B, 1, C)
    # 4) 门控  【用 Lambda 层包装】
    weight = Lambda(tf.nn.sigmoid)(weight)   # (B, 1, C)
    print('weight.shape',weight.shape)
    return Multiply()([x, weight])     # 逐通道加权

# -------------------------------------------------
# 多任务模型（ECA 版）
# -------------------------------------------------
def create_multitask_model(input_shape, output_dim_spec, output_dim_conc):
    inputs = Input(shape=input_shape)          # (None, 256, 1)

    ##########################################################################
    # NO 分支：CNN
    ##########################################################################
    x_no = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x_no = MaxPooling1D(2)(x_no)
    x_no = Conv1D(64, 3, activation='relu', padding='same')(x_no)
    x_no = MaxPooling1D(2)(x_no)
    x_no = Conv1D(128, 3, activation='relu', padding='same')(x_no)
    x_no = MaxPooling1D(2)(x_no)
    x_no = Flatten()(x_no)
    x_no = Dense(128, activation='relu')(x_no)
    spec_output_NO = Dense(output_dim_spec, activation='linear', name='spec_output_NO')(x_no)

    ##########################################################################
    # SO2 分支 1：CNN + ECA
    ##########################################################################
    x_cnn = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = eca_block_1d(x_cnn)                    # <- 替换原 Attention
    x_cnn = Conv1D(64, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = eca_block_1d(x_cnn)                    # <- 替换原 Attention
    x_cnn = Conv1D(128, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = Flatten()(x_cnn)
    x_cnn = Dense(128, activation='relu')(x_cnn)

    ##########################################################################
    # SO2 分支 2：Bi-LSTM + ECA
    ##########################################################################
    x_lstm = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x_lstm = MaxPooling1D(2)(x_lstm)
    x_lstm = Flatten()(x_lstm)
    #  reshape 到 (B, T, C)  满足 LSTM 需求
    x_lstm = Reshape((-1, input_shape[0]))(x_lstm)   # (B, T=64, C=256)
    x_lstm = Bidirectional(LSTM(64, return_sequences=True))(x_lstm)
    x_lstm = eca_block_1d(x_lstm)                    # <- 替换原 Attention
    x_lstm = Bidirectional(LSTM(32, return_sequences=False))(x_lstm)

    ##########################################################################
    # SO2 分支 3：一阶导 + CNN + ECA
    ##########################################################################
    diff = Lambda(lambda x: x[:, 1:] - x[:, :-1])(inputs)
    diff = Reshape((diff.shape[1], 1))(diff)
    x_diff = Conv1D(32, 3, activation='relu', padding='same')(diff)
    x_diff = MaxPooling1D(2)(x_diff)
    x_diff = eca_block_1d(x_diff)                    # <- 替换原 Attention
    x_diff = Flatten()(x_diff)
    x_diff = Dense(64, activation='relu')(x_diff)

    ##########################################################################
    # 融合 & 输出
    ##########################################################################
    combined_SO2 = Concatenate()([x_cnn, x_lstm, x_diff])
    spec_output_SO2 = Dense(output_dim_conc, activation='linear', name='spec_output_SO2')(combined_SO2)

    model = Model(inputs, [spec_output_NO, spec_output_SO2])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss={'spec_output_NO': 'mse', 'spec_output_SO2': 'mse'},
                  loss_weights=[0.7, 0.3],
                  metrics={'spec_output_NO': 'mae', 'spec_output_SO2': 'mae'})
    return model

# ---------------- 构建 & 打印 ----------------
sequence_length = 256
output_dim_spec = 256
output_dim_conc = 256
model = create_multitask_model((sequence_length, 1), output_dim_spec, output_dim_conc)
model.summary()