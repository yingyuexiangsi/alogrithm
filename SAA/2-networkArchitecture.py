import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

import os
plt.rc('font',family='Times New Roman')
# 设置字体加粗
plt.rcParams['font.weight'] = 'bold'
plt.rcParams.update({'font.size': 15}) # 改变所有字体大小，改变其他性质类似

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Flatten, Dense,
                                     Bidirectional, LSTM, Lambda, Concatenate,
                                     Reshape, GlobalAveragePooling1D, Multiply)
from tensorflow.keras import backend as K
import math


def fourier_filter_tf(signal, t, tfStart, tfEnd):
    """TensorFlow 实现的傅里叶带通滤波函数"""
    t = tf.cast(t, tf.float32)
    dt = t[1] - t[0]
    signal_flat = tf.squeeze(signal, axis=-1)
    seq_length = tf.shape(signal_flat)[1]

    fft_result = tf.signal.rfft(signal_flat)
    n = tf.cast(seq_length, tf.float32)
    k = tf.range(0, seq_length // 2 + 1, dtype=tf.int32)
    k_float = tf.cast(k, tf.float32)
    freqs = k_float / (n * dt)

    mask = tf.logical_and(freqs >= tfStart, freqs <= tfEnd)
    mask = tf.cast(mask, dtype=tf.complex64)
    filtered_fft = fft_result * mask
    filtered_signal = tf.signal.irfft(filtered_fft)
    filtered_signal = filtered_signal[:, :seq_length]

    return tf.expand_dims(filtered_signal, axis=-1)


# -------------------------------------------------
# ECA-Net 1D 通道注意力（零参数版）
# -------------------------------------------------
def eca_block_1d(x, gamma=2., b=1):
    """
    x: (B, T, C)  ——  T 为时间步，C 为通道
    return: (B, T, C) 加权后特征
    """
    C = int(x.shape[-1])
    # 自适应卷积核大小
    k = int(abs((math.log(C, 2) + b) / gamma))
    k = k if k % 2 else k + 1  # 必须为 odd
    # 全局平均 -> (B, C)
    gap = GlobalAveragePooling1D()(x)  # (B, C)
    gap = K.expand_dims(gap, axis=-2)  # (B, 1, C)
    # 1D 局部跨通道交互（权重共享）
    weight = Conv1D(1, kernel_size=k, padding='same', use_bias=False)(gap)
    weight = K.sigmoid(weight)  # (B, 1, C)
    return Multiply()([x, weight])  # 逐通道加权


# -------------------------------------------------
# 多任务模型（ECA 版）
# -------------------------------------------------
def create_multitask_model(input_shape, output_dim_spec, output_dim_conc, t):
    inputs = Input(shape=input_shape)  # (None, 256, 1)

    ##########################################################################
    # NO 分支：使用低频滤波后的信号
    ##########################################################################
    filtered_low = Lambda(
        lambda x: fourier_filter_tf(x, t, 0, 0.2),
        output_shape=input_shape
    )(inputs)
    x_no = Conv1D(32, 3, activation='relu', padding='same')(filtered_low)
    x_no = MaxPooling1D(2)(x_no)
    x_no = Conv1D(64, 3, activation='relu', padding='same')(x_no)
    x_no = MaxPooling1D(2)(x_no)
    x_no = Conv1D(128, 3, activation='relu', padding='same')(x_no)
    x_no = MaxPooling1D(2)(x_no)
    x_no = Flatten()(x_no)
    x_no = Dense(128, activation='relu')(x_no)
    spec_output_NO = Dense(output_dim_spec, activation='linear', name='spec_output_NO')(x_no)

    ##########################################################################
    # SO2 分支 1：使用高频滤波后的信号
    ##########################################################################
    filtered_high = Lambda(
        lambda x: fourier_filter_tf(x, t, 0.5, 0.7),
        output_shape=input_shape
    )(inputs)
    x_cnn = Conv1D(32, 3, activation='relu', padding='same')(filtered_high)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = eca_block_1d(x_cnn)  # <- 替换原 Attention
    x_cnn = Conv1D(64, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = eca_block_1d(x_cnn)  # <- 替换原 Attention
    x_cnn = Conv1D(128, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = Flatten()(x_cnn)
    x_cnn = Dense(128, activation='relu')(x_cnn)

    ##########################################################################
    # SO2  分支 2：Bi-LSTM + ECA
    ##########################################################################
    x_lstm = Conv1D(64, 3, activation='relu', padding='same')(filtered_high)
    x_lstm = MaxPooling1D(2)(x_lstm)
    x_lstm = Flatten()(x_lstm)
    #  reshape 到 (B, T, C)  满足 LSTM 需求
    x_lstm = Reshape((-1, input_shape[0]))(x_lstm)  # (B, T=64, C=256)
    x_lstm = Bidirectional(LSTM(64, return_sequences=True))(x_lstm)
    x_lstm = eca_block_1d(x_lstm)  # <- 替换原 Attention
    x_lstm = Bidirectional(LSTM(32, return_sequences=False))(x_lstm)

    ##########################################################################
    # SO2  分支 3：一阶导 + CNN + ECA
    ##########################################################################
    diff = Lambda(lambda x: x[:, 1:] - x[:, :-1])(filtered_high)
    diff = Reshape((diff.shape[1], 1))(diff)
    x_diff = Conv1D(32, 3, activation='relu', padding='same')(diff)
    x_diff = MaxPooling1D(2)(x_diff)
    x_diff = eca_block_1d(x_diff)  # <- 替换原 Attention
    x_diff = Flatten()(x_diff)
    x_diff = Dense(64, activation='relu')(x_diff)

    ##########################################################################
    # 融合 & 输出
    ##########################################################################
    combined_SO2 = Concatenate()([x_cnn, x_lstm])
    spec_output_SO2 = Dense(output_dim_conc, activation='linear', name='spec_output_SO2')(combined_SO2)

    model = Model(inputs, [spec_output_SO2, spec_output_NO])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss={'spec_output_SO2': 'mse', 'spec_output_NO': 'mse'},
                  loss_weights=[0.3, 0.7],
                  metrics={'spec_output_SO2': 'mae', 'spec_output_NO': 'mae'})
    return model


# ---------------- 构建 & 打印 ----------------
sequence_length = 256
output_dim_spec = 1
output_dim_conc = 1
t = np.linspace(200, 230, sequence_length)
model = create_multitask_model((sequence_length, 1), output_dim_spec, output_dim_conc, t)
model.summary()
