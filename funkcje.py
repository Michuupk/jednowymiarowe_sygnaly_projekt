import numpy as np
from scipy.signal import coherence, stft
import pywt # pamiętaj o pip install PyWavelets
import pandas as pd
from scipy.stats import ks_2samp
import numpy as np

# --- 3. WSKAŹNIK PRx (Analiza z artykułu) ---
def calculate_PRx(okna_icp, okna_abp, x_seconds_avg):
    """
    okna_icp, okna_abp: macierze z sliding_window_view po packet_averaging
    x_seconds_avg: czas (w sekundach), z jakiego wyciągnięto jedną średnią (np. 5)
    
    Zwraca: np.ndarray o kształcie (liczba_okien, 2)
            Kolumna 0: Czas środka okna w GODZINACH
            Kolumna 1: Wartość PRx
    """
    liczba_okien, window_size = okna_icp.shape
    prx_values = []
    czasy_okien_godziny = []

    for i in range(liczba_okien):
        icp_window = okna_icp[i]
        abp_window = okna_abp[i]

        # --- OBLICZANIE RZECZYWISTEGO CZASU ---
        # Początek okna 'i' w sekundach to indeks * czas trwania jednego pakietu
        # Środek okna to początek + połowa szerokości okna (również w pakietach)
        srodek_sekundy = (i + (window_size / 2.0)) * x_seconds_avg
        
        # Przeliczenie na godziny (do późniejszego filtrowania stref czasowych)
        czas_h = srodek_sekundy / 3600.0
        czasy_okien_godziny.append(czas_h)

        # --- OBLICZANIE KORELACJI PRx ---
        mask = ~np.isnan(icp_window) & ~np.isnan(abp_window)
        if np.sum(mask) > 2:
            r = np.corrcoef(icp_window[mask], abp_window[mask])[0, 1]
            prx_values.append(r)
        else:
            prx_values.append(np.nan)

    return np.column_stack((czasy_okien_godziny, prx_values))

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

def defineTimeZones(PRx: np.ndarray, start: float, end: float, pair_time_sectors: list):
    """
    Filtruje macierz wyników PRx na podstawie globalnego zakresu czasu oraz zdefiniowanych sektorów zdarzeń.
    
    Argumenty:
    PRx: macierz NumPy o kształcie (N, 2), gdzie kolumna 0 to czas (h), a kolumna 1 to wartość.
    start: globalny czas początkowy w godzinach (np. 2.5)
    end: globalny czas końcowy w godzinach (np. 17.75)
    pair_time_sectors: lista przedziałów zdarzeń w godzinach, np. [[3, 4.25], [4.5, 5]]
    
    Zwraca:
    eventPRx: macierz z danymi wewnątrz przedziałów zdarzeń (z uwzględnieniem globalnego zakresu)
    normalPRx: macierz z danymi poza zdarzeniami (ale nadal wewnątrz globalnego zakresu)
    """
    # Wyciągamy całą kolumnę czasu dla wygody
    czas = PRx[:, 0]
    
    # 1. Tworzymy maskę globalną (czy dany punkt mieści się między 'start' a 'end')
    global_mask = (czas >= start) & (czas <= end)
    
    # 2. Tworzymy maskę zdarzeń (domyślnie wszystko ustawiamy na False)
    event_mask = np.zeros(len(PRx), dtype=bool)
    
    # Dodajemy kolejne przedziały do maski zdarzeń (używając operatora logicznego LUB: '|')
    for s_start, s_end in pair_time_sectors:
        event_mask |= (czas >= s_start) & (czas <= s_end)
        
    # 3. Łączymy maski
    # eventPRx: musi być w przedziale globalnym ORAZ w przedziale zdarzenia
    is_event = global_mask & event_mask
    
    # normalPRx: musi być w przedziale globalnym ORAZ NIE BYĆ w przedziale zdarzenia (~ to negacja)
    is_normal = global_mask & ~event_mask
    
    # 4. Wycinamy odpowiednie wiersze z macierzy
    eventPRx = PRx[is_event]
    normalPRx = PRx[is_normal]

    return eventPRx, normalPRx

def kolmogorovSmirnov(event_vals, normal_vals):

    # 1. Wyciągamy TYLKO wartości PRx (indeks 1, bo indeks 0 to czas)
    event_vals = event_vals[:, 1]
    normal_vals = normal_vals[:, 1]

    # 2. Filtrujemy dane z wartości NaN (bardzo ważne dla testu K-S!)
    event_vals_clean = event_vals[~np.isnan(event_vals)]
    normal_vals_clean = normal_vals[~np.isnan(normal_vals)]

    # 3. Wykonujemy test Kołmogorowa-Smirnowa dla dwóch prób
    ks_stat, p_value = stats.ks_2samp(event_vals_clean, normal_vals_clean)

    print(f"Statystyka K-S: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4e}")

    # Prosta interpretacja
    alpha = 0.05
    if p_value < alpha:
        print("Odrzucamy hipotezę zerową: Rozkłady PRx w strefach zdarzeń i normalnych SĄ istotnie różne.")
    else:
        print("Brak podstaw do odrzucenia hipotezy zerowej: Rozkłady PRx są do siebie podobne.")