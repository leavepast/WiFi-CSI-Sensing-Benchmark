import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from wifiread import read_pcap

def get_doppler_spectrum_stream(csi, rx_cnt=1, rx_acnt=1, num_tones=256, freq_bin_len=256, method='stft'):
    # 设置参数
    sample_rate = 1000
    half_rate = sample_rate / 2
    upper_order = 6
    upper_stop = 40
    lower_order = 6
    lower_stop = 2
    lu, ld = signal.butter(upper_order, upper_stop / sample_rate, 'low')
    hu, hd = signal.butter(lower_order, lower_stop / sample_rate, 'high')

    ii = 0
    # Select Antenna Pair[WiDance]
    csi_mean = np.mean(np.abs(csi), axis=0)
    csi_var = np.sqrt(np.var(np.abs(csi), axis=0))
    csi_mean_var_ratio = np.divide(csi_mean, csi_var)
    csi_mean_var_ratio_mean = np.mean(np.transpose(np.reshape(csi_mean_var_ratio, [rx_acnt, num_tones])), axis=0)
    if rx_cnt==1 and rx_acnt==1:
        idx=0
    else:
        idx = int(np.where(csi_mean_var_ratio_mean == np.max(csi_mean_var_ratio_mean))[0])
    csi_data_ref = np.tile(csi[:, idx * num_tones: (idx + 1) * num_tones], [1, rx_acnt])
    # Amp Adjust[IndoTrack]
    csi_data_adj = np.zeros((np.shape(csi)), dtype=complex)
    csi_data_ref_adj = np.zeros(np.shape(csi_data_ref), dtype=complex)
    alpha_sum = 0

    for jj in range(num_tones * rx_acnt):
        amp = np.abs(csi[:, jj])
        try:
            alpha = np.min(amp[amp != 0])
        except:
            pass
        alpha_sum = alpha_sum + alpha
        csi_data_adj[:, jj] = np.multiply(np.abs(csi[:, jj]) - alpha, np.exp(1j * np.angle(csi[:, jj])))

    beta = 1000 * alpha_sum / (num_tones * rx_acnt)
    for jj in range(num_tones * rx_acnt):
        csi_data_ref_adj[:, jj] = np.multiply(np.abs(csi[:, jj]) + beta, np.exp(1j * np.angle(csi_data_ref[:, jj])))

    # Conj Mult
    conj_mult = np.multiply(csi_data_adj, np.conj(csi_data_ref_adj))
    conj_mult = np.concatenate([conj_mult[:, : int(num_tones * idx)], conj_mult[:, int(num_tones * (idx + 1)): num_tones * 3]], axis=1)
    # Filter Out Static Component & High Frequency Component
    for jj in range(np.size(conj_mult, 1)):
        conj_mult[:, jj] = signal.lfilter(lu, ld, conj_mult[:, jj])
        conj_mult[:, jj] = signal.lfilter(hu, hd, conj_mult[:, jj])

    # PCA analysis
    # pca = PCA(n_components=60)
    # conj_mult_pca_real = pca.fit_transform(conj_mult.real)  # 不支持复数，暂时使用这种方法，和MATLAB不同
    # conj_mult_pca_imag = pca.fit_transform(conj_mult.imag)  # 不支持复数，暂时使用这种方法，和MATLAB不同
    # conj_mult_pca = np.zeros(np.shape(conj_mult_pca_real), dtype=complex)
    # for kk in range(np.size(conj_mult_pca_real, 0)):
    #     for qq in range(np.size(conj_mult_pca_real, 1)):
    #         conj_mult_pca[kk, qq] = complex(conj_mult_pca_real[kk, qq], conj_mult_pca_imag[kk, qq])

    # % TFA With CWT or STFT
    if method == 'stft':
        # window_size = int(np.round(sample_rate / 4 + 1))
        window_size = sample_rate - 1
        if np.mod(window_size, 2) == 0:
            window_size = window_size + 1
        # f, t, freq_time_prof = signal.spectrogram(conj_mult[:,0],window=signal.get_window('hamming',256),fs=1000,nperseg=256, noverlap=255,return_onesided=False,axis=0)  # 和MATLAB的spectragram()对应
        f, t, freq_time_prof = signal.stft(conj_mult[:, 0], window=signal.get_window('hann', window_size), fs=sample_rate, nperseg=window_size, noverlap=window_size - 1, return_onesided=False)  # 和MATLAB的tfrsp()对应
    # plt.figure(ii + 1)
    # plt.pcolormesh(np.arange(np.size(freq_time_prof, 1)), np.arange(np.size(freq_time_prof, 0)),
    #                circshift1D(np.abs(freq_time_prof), int(np.size(freq_time_prof, 0) / 2)))
    # plt.pause(0.1)
    # Select Concerned Freq

    # Spectrum Normalization By Sum For Each Snapshot
    freq_time_prof = np.divide(np.abs(freq_time_prof), np.tile(np.sum(np.abs(freq_time_prof), 0), np.size(freq_time_prof, 0)).reshape([np.size(freq_time_prof, 0), np.size(freq_time_prof, 1)]))

    # Frequency Bin(Corresponding to FFT Results)

    # Store Doppler Velocity Spectrum
    if ii == 0:
        doppler_spectrum = np.zeros([rx_cnt, np.size(freq_time_prof, 0), np.size(freq_time_prof, 1)])
    if np.size(freq_time_prof, 1) >= np.size(doppler_spectrum, 2):
        doppler_spectrum[ii, :, :] = freq_time_prof[:, : np.size(doppler_spectrum, 2)]
    else:
        doppler_spectrum[ii, :, :] = np.concatenate([freq_time_prof, np.zeros([np.size(doppler_spectrum, 1), int(np.size(doppler_spectrum, 2) - np.size(freq_time_prof, 1))])], axis=1)

    doppler_spectrum = doppler_spectrum.squeeze()
    plt.pcolormesh(np.arange(np.size(doppler_spectrum, 1)), np.arange(np.size(doppler_spectrum, 0)),
                   circshift1D(np.abs(doppler_spectrum), int(np.size(doppler_spectrum, 0) / 2)), cmap='jet')
    plt.ylim([420, 550])
    plt.pause(0.1)

    doppler_spectrum_center = int(np.where(f == np.max(f))[0])
    return doppler_spectrum, circshift1D(f, doppler_spectrum_center)[doppler_spectrum_center - freq_bin_len: doppler_spectrum_center + freq_bin_len + 1]

def circshift1D(lst, k):
    # k是右移
    return np.concatenate([lst[-k:], lst[: -k]])

if __name__ == '__main__':
    data=read_pcap(file="/data/wifi/hexin/data_zu1/s1_高勤标/pcap/t1.pcap")
    # data=np.expand_dims(data,data.ndim)
    # data = np.expand_dims(data, data.ndim)
    # get_doppler_spectrum_stream(data)
