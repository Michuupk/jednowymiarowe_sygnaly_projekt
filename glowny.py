import numpy as np

# IMPORTUJEMY KLASĘ Z PLIKU processor.py
from processor import SignalProcessor

# IMPORTUJEMY FUNKCJE Z PLIKU analysis.py
from funkcje import analiza_falkowa, analiza_koherencji

fs = 100 # np. 100 Hz
dane_icp = np.random.normal(15, 2, 1000) # 1000 losowych pomiarów
dane_abp = np.random.normal(80, 10, 1000)

# 2. Używamy zaimportowanej KLASY (wnętrze)
print("--- URUCHAMIAMY KLASĘ ---")
moj_procesor = SignalProcessor(dane_icp, dane_abp, sampling_freq=fs)

moj_procesor.filtracja_sygnalow()
moj_procesor.usun_szumy()
srednie_icp, srednie_abp = moj_procesor.packet_averaging(x_seconds=1)

# 3. Używamy zaimportowanych FUNKCJI (zewnętrzne) FIX
print("\n--- URUCHAMIAMY ANALIZY ---")
# Wysyłamy wyczyszczone dane (np. uśrednione ICP) do funkcji falkowej
wynik_falkowy = analiza_falkowa(srednie_icp, fs)
print("Analiza falkowa zakończona.")

f, koherencja, powiazanie = analiza_koherencji(srednie_icp, srednie_abp, fs)
print("Analiza koherencji zakończona.")