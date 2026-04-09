# Instalacja i setup PyDrive (jednorazowo)
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os


# KONFIGURACJA - ZMIEŃ NA SWOJE!
data_dir = '/content/data'  # Lokalny folder w Colab
record_name = 'charis4'     # Nazwa rekordu (.hea i .dat)

# TERAZ TWÓJ KOD BEZ ZMIAN
local_path = os.path.join(data_dir, record_name)

# Odczyt (automatycznie znajdzie .hea i .dat)
record = wfdb.rdrecord(local_path)

# Dane sygnałów
signals = record.p_signal  # NxM tablica (próbki x kanały)
fs = record.fs  # Częstotliwość próbkowania
time = np.arange(len(signals)) / fs

print(f"Rekord: {record_name}")
print(f"Długość: {len(signals)} próbek ({len(signals)/fs/3600:.1f} h)")
print(f"fs: {fs} Hz, Kanały: {record.n_sig}")
print(f"Nazwy kanałów: {record.sig_name}")  # np. ['MCG', 'ABP', 'ICP']

# EKG/ECG (dostosuj indeks kanału)
ecg_channel = 0  # Lub 1 dla ECG w CHARIS
ecg = signals[:, ecg_channel]

# Wizualizacja pierwszych 60s
plt.figure(figsize=(15, 6))
plt.plot(time[:int(60*fs)], ecg[:int(60*fs)], linewidth=0.8)
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.title(f'EKG z {record_name} (z Google Drive)')
plt.grid(True, alpha=0.3)
plt.show()

# Zapis do CSV (opcjonalnie)
# np.savetxt('ecg_data.csv', np.column_stack([time, ecg]), delimiter=',', 
#            header='time_s,ecg_mv', comments='')
