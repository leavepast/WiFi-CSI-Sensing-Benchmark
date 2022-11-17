import numpy as np
import matplotlib.pyplot as plt
import pywt
from CSIKit.tools.batch_graph import BatchGraph
from scipy import signal
from wifiread import read_pcap
from sklearn.decomposition import PCA



def reduce_noise(sub):
    sub_length = len(sub)
    #wavelet analysis
    wavelet = pywt.Wavelet("sym5")
    max_level = pywt.dwt_max_level(len(sub), wavelet.dec_len)

    #decompose
    coefficients = pywt.wavedec(sub, wavelet, level=max_level)

    #filtering
    for i in range(len(coefficients)):
        #see coefficients 5% smaller than max coef as noise
        coefficients[i] = pywt.threshold(coefficients[i], 0.05*max(coefficients[i]))

    #reconstruct
    data = pywt.waverec(coefficients, wavelet)
    return data[:sub_length]

def reduce_noise_batch(subs):
    new_datas = np.zeros((np.size(subs, 0), np.size(subs, 1)))
    for i in range(0, np.size(subs, 1)):
        new_datas[:,i]=reduce_noise(subs[:,i])
    return new_datas

def reduce_dim(subs,n_PCA_components,size,patch :int =1):
    pca = PCA(n_PCA_components)
    new_datas = np.zeros((len(subs), n_PCA_components * patch), np.float64)
    for i in range(0, patch):
        new_datas[:, n_PCA_components * i:n_PCA_components * (i + 1)] = pca.fit_transform(subs[:, size * i:size * (i + 1)])
    return new_datas


if __name__ == '__main__':
    subs=read_pcap(file="/data/wifi/hexin/data_zu1/s1_高勤标/pcap/t1.pcap")
    plt.subplot(5, 1, 1)
    for i in range(np.size(subs,1)):
        plt.plot(np.arange(np.size(subs, 0)), np.abs(subs[:, i]))
    plt.subplot(5, 1, 2)
    for i in range(np.size(subs,1)):
        sub_amp=np.abs(subs[:, i])
        plt.plot(np.arange(np.size(subs, 0)),reduce_noise(sub_amp))
    sub_index=60
    plt.subplot(5, 1, 3)
    plt.plot(np.arange(np.size(subs, 0)), np.abs(subs[:, sub_index]))
    plt.subplot(5, 1, 4)
    plt.plot(np.arange(np.size(subs, 0)), reduce_noise(np.abs(subs[:, sub_index])))
    plt.subplot(5, 1, 5)
    subs_low=reduce_noise_batch(np.abs(subs))
    #subs_low=reduce_dim( np.abs(subs),4)
    for i in range(np.size(subs_low, 1)):
        plt.plot(np.arange(np.size(subs_low, 0)), np.abs(subs_low[:, i]))
    plt.show()
    pass
