import numpy as np
from scipy.signal import coherence, stft
import pywt # pamiętaj o pip install PyWavelets
import pandas as pd

# --- 3. WSKAŹNIK PRx (Analiza z artykułu) ---
def calculate_PRx(okna_icp, okna_abp):
    """
    icp i abp to macierze z funkcji sliding_window_view
    kształt: (liczba_okien, punkty_w_oknie)
    """
    prx_values = []

    # Iterujemy po każdym oknie czasowym
    for i in range(len(okna_icp)):
        # Wyciągamy pojedyncze okno dla obu sygnałów
        icp_window = okna_icp[i]
        abp_window = okna_abp[i]

        # Tworzymy DataFrame tylko dla tego jednego okna (opcjonalnie)
        # lub używamy bezpośrednio numpy dla szybkości:
        
        # Usuwamy NaN z obu sygnałów jednocześnie (jeśli są)
        mask = ~np.isnan(icp_window) & ~np.isnan(abp_window)
        if np.sum(mask) > 2:  # Potrzebujemy min. 3 punktów do korelacji
            r = np.corrcoef(icp_window[mask], abp_window[mask])[0, 1]
            prx_values.append(r)
        else:
            prx_values.append(np.nan)

    return np.array(prx_values)

def analiza_falkowa(sygnal, fs, skala_min=5, skala_max=20, krok=1):
    skale = np.arange(skala_min, skala_max + krok, krok)
    wspolczynniki, czestotliwosci = pywt.cwt(sygnal, skale, 'morl')
    return wspolczynniki, czestotliwosci

def analiza_czasowo_czestotliwosciowa(sygnal, fs, rozmiar_okna=1024, overlap_percent=0.9):
    noverlap = int(rozmiar_okna * overlap_percent)
    f, t, Zxx = stft(sygnal, fs=fs, window='hann', nperseg=rozmiar_okna, noverlap=noverlap)
    
    pasmo_idx = np.argmin(np.abs(f - 0.4)) 
    wartosci_dla_04Hz = np.abs(Zxx[pasmo_idx, :])
    return t, wartosci_dla_04Hz

def analiza_koherencji(sygnal1, sygnal2, fs):
    f, Cxy = coherence(sygnal1, sygnal2, fs=fs)
    mask = (f >= 0.05) & (f <= 0.2)
    f_zainteresowania = f[mask]
    koherencja_w_pasmie = Cxy[mask]
    powiazane = koherencja_w_pasmie > 0.5
    return f_zainteresowania, koherencja_w_pasmie, powiazane
