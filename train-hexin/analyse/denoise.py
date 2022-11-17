import numpy as np
import matplotlib.pyplot as plt
from CSIKit.tools.batch_graph import BatchGraph
from scipy import signal
from wifiread import read_pcap

def lowpass(csi_vec: np.array, cutoff, fs: float=1000, order: int =6) -> np.array:
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    csi = np.zeros([np.size(csi_vec, 0), np.size(csi_vec, 1)], dtype=complex)
    for i in range(np.size(csi_vec,1)):
        csi[:, i]=signal.filtfilt(b, a, csi_vec[:,i])
    return csi

def highpass(csi_vec: np.array, cutoff, fs: float=1000, order: int =6) -> np.array:
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    csi = np.zeros([np.size(csi_vec, 0), np.size(csi_vec, 1)], dtype=complex)
    for i in range(np.size(csi_vec, 1)):
        csi[:, i] = signal.filtfilt(b, a, csi_vec[:, i])
    return csi


def bandpass(csi_vec: np.array, low_cut: float, high_cut: float, fs: float=1000, order: int=6) -> np.array:
    nyq = 0.5*fs
    b, a = signal.butter(order, [low_cut/nyq, high_cut/nyq], btype="band", analog=False)
    csi = np.zeros([np.size(csi_vec, 0), np.size(csi_vec, 1)], dtype=complex)
    for i in range(np.size(csi_vec, 1)):
        csi[:, i] = signal.filtfilt(b, a, csi_vec[:, i])
    return csi

if __name__ == '__main__':
    data=read_pcap(file="/data/wifi/hexin/data_zu1/s1_高勤标/pcap/t1.pcap")
    csi_l=lowpass(data,cutoff=2,fs=10,order=10)
    csi_h=highpass(data,cutoff=2,fs=10,order=10)
    plt.subplot(3, 1, 1)
    for i in range(np.size(data,1)):
        plt.plot(np.arange(np.size(data, 0)), np.abs(data[:, i]))
    plt.subplot(3, 1, 2)
    for i in range(np.size(data, 1)):
        plt.plot(np.arange(np.size(data, 0)), np.abs(csi_l[:, i]))
    plt.subplot(3, 1, 3)
    for i in range(np.size(data, 1)):
        plt.plot(np.arange(np.size(data, 0)), np.abs(csi_h[:, i]))
    plt.show()
    # data=np.expand_dims(data,data.ndim)
    # data = np.expand_dims(data, data.ndim)
    # get_doppler_spectrum_stream(data)