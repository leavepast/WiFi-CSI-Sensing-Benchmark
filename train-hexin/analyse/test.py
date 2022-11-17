# -*- coding: utf-8 -*-

import pywt
import numpy as np
import matplotlib.pyplot as plt


def cwt(x, fs, totalscal, wavelet='cgau8'):
    if wavelet not in pywt.wavelist():
        print('小波函数名错误')
    else:
        wfc = pywt.central_frequency(wavelet=wavelet)
        a = 2 * wfc * totalscal / (np.arange(totalscal, 0, -1))
        period = 1.0 / fs
        [cwtmar, fre] = pywt.cwt(x, a, wavelet, period)
        amp = abs(cwtmar)
        return amp, fre


def dwt(x, wavelet='db3'):
    cA, cD = pywt.dwt(x, wavelet, mode='symmetric')
    ya = pywt.idwt(cA, None, wavelet, mode='symmetric')
    yd = pywt.idwt(None, cD, wavelet, mode='symmetric')
    return ya, yd, cA, cD


if __name__ == '__main__':
    w = 5
    z = 30
    fs = 1024
    fsw = 5
    time = 10
    f = w * z
    t = np.linspace(0, time - 1 / fs, int(time * fs))
    x = (1 + 1 * np.sin(2 * np.pi * 20 * t)) * np.sin(2 * np.pi * f * t)
    amp, fre = cwt(x, fs, 512, 'morl')
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.ylabel('Amplitude')
    plt.xlabel('time')
    plt.subplot(2, 1, 2)
    plt.contourf(t, fre, amp)
    plt.ylabel('Frequency')
    plt.xlabel('time')
    # 离散小波分析
    ya, yd, _, _ = dwt(x, 'db3')
    plt.figure(2)
    plt.plot(t, ya)
    plt.xlabel('time')
    plt.ylabel('近似系数')
    plt.show()
