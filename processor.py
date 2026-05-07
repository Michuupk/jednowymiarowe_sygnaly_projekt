import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

class SignalProcessor:
    def __init__(self, icp_data, abp_data, sampling_freq):
        self.icp = np.array(icp_data)
        self.abp = np.array(abp_data)
        self.fs = sampling_freq

    def filtracja_sygnalow(self, lowcut: float = 0.02, highcut: float = 10.0):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        
        #self.icp = filtfilt(b, a, self.icp)
        #self.abp = filtfilt(b, a, self.abp)
        print("Sygnały przefiltrowane.")

    def usun_szumy(self):
        icp_mask_bad = (self.icp < -1) | (self.icp > 100)
        self.icp[icp_mask_bad] = np.nan 
        self.abp[self.abp < 20] = np.nan
        
        samples_in_05s = int(0.5 * self.fs)
        icp_diff = np.abs(np.append(np.zeros(samples_in_05s), self.icp[samples_in_05s:] - self.icp[:-samples_in_05s]))
        abp_diff = np.abs(np.append(np.zeros(samples_in_05s), self.abp[samples_in_05s:] - self.abp[:-samples_in_05s]))
        
        self.icp[icp_diff > 50] = np.nan
        self.abp[abp_diff > 50] = np.nan
        print("Szumy usunięte.")

    def packet_averaging(self, x_seconds):
        samples_per_packet = int(x_seconds * self.fs)
        df = pd.DataFrame({'ICP': self.icp, 'ABP': self.abp})
        averaged = df.groupby(np.arange(len(df)) // samples_per_packet).mean()
        
        self.mean_icp = averaged['ICP'].values
        self.map_abp = averaged['ABP'].values
        return self.mean_icp, self.map_abp

    def windowing(self, par_x):
        okna_icp = np.lib.stride_tricks.sliding_window_view(self.mean_icp, window_shape=par_x)
        okna_abp = np.lib.stride_tricks.sliding_window_view(self.map_abp, window_shape=par_x)
        return okna_icp, okna_abp