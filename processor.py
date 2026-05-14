import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

class Signal:
    def __init__(self, data, sampling_freq, name="Signal"):
        self.data = np.array(data, dtype=float)
        self.fs = sampling_freq
        self.name = name
        self.mean_values = None  # Tu trafią dane po packet_averaging

    def filter_signal(self, lowcut: float = 0.02, highcut: float = 10.0, order: int = 4):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        
        # filtfilt zapobiega przesunięciu fazowemu
        self.data = filtfilt(b, a, self.data)
        print(f"[{self.name}] Sygnał przefiltrowany.")

    def remove_noise(self, min_val=None, max_val=None, diff_threshold=50):
        # Usuwanie po wartościach bezwzględnych
        if min_val is not None:
            self.data[self.data < min_val] = np.nan
        if max_val is not None:
            self.data[self.data > max_val] = np.nan
        
        # Usuwanie nagłych skoków (różnicowe)
        samples_in_05s = int(0.5 * self.fs)
        # Przesunięcie i obliczenie różnicy
        diff = np.abs(np.append(np.zeros(samples_in_05s), self.data[samples_in_05s:] - self.data[:-samples_in_05s]))
        self.data[diff > diff_threshold] = np.nan
        print(f"[{self.name}] Szumy usunięte.")

    def packet_averaging(self, x_seconds):
        samples_per_packet = int(x_seconds * self.fs)
        # Używamy pandas do łatwego liczenia średniej z pominięciem NaN
        series = pd.Series(self.data)
        self.mean_values = series.groupby(np.arange(len(series)) // samples_per_packet).mean().values
        return self.mean_values

class ICP_ABP_Processor:
    def __init__(self, icp_data, abp_data, sampling_freq):
        # Tworzymy osobne obiekty dla ICP i ABP
        self.icp = Signal(icp_data, sampling_freq, name="ICP")
        self.abp = Signal(abp_data, sampling_freq, name="ABP")
        self.fs = sampling_freq

    def process_all(self):
        # Możesz wywoływać metody z różnymi parametrami dla każdego sygnału
        self.icp.remove_noise(min_val=-1, max_val=100)
        self.abp.remove_noise(min_val=20)
        
        self.icp.filter_signal()
        self.abp.filter_signal()

    def get_windowed_data(self, x_seconds_avg, window_size):
        # Średnie kroczące / pakietowe
        mean_icp = self.icp.packet_averaging(x_seconds_avg)
        mean_abp = self.abp.packet_averaging(x_seconds_avg)
        
        # Okienkowanie (sliding window)
        okna_icp = np.lib.stride_tricks.sliding_window_view(mean_icp, window_shape=window_size)
        okna_abp = np.lib.stride_tricks.sliding_window_view(mean_abp, window_shape=window_size)
        
        return okna_icp, okna_abp