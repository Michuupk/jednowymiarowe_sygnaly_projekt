import numpy as np
from scipy.signal import coherence, stft
import pywt # pamiętaj o pip install PyWavelets
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def defineTimeZones(PRx: np.ndarray, event_time_sectors: list, skip_time_sectors: list = None):
    """
    Filtruje macierz wyników PRx.
    
    Zwraca:
    - eventPRx: wyłącznie z przedziałów [event_start_min, event_end_min]
    - normalPRx: wyłącznie z przedziałów [pre_event_start_min, event_start_min)
    - wholePRx: zaczyna się 15 minut przed pierwszym pre_event_start_min 
                (lub od 0, jeśli zabraknie czasu na początku), 
                a kończy równo 60 minut (1 godzinę) po ostatnim event_end_min.
    """
    if not event_time_sectors:
        raise ValueError("Lista event_time_sectors nie może być pusta.")

    czas = PRx[:, 0]
    
    # --- 1. Wyliczanie ram czasowych dla wholePRx ---
    # Koniec ostatniego eventu
    last_event_end_min = event_time_sectors[-1]['event_end_min']
    
    # Dodajemy 60 minut (1 godzinę) do globalnego końca
    global_end_min = last_event_end_min + 60.0
    
    # Pobieramy początek najwcześniejszego pre-eventu
    first_pre_event_start_min = event_time_sectors[0]['pre_event_start_min']
    
    # Globalny start cofnięty o 15 minut. 
    # Używamy max(0, ...), by nie wejść w czas ujemny w przypadku wczesnych eventów.
    global_start_min = max(0, first_pre_event_start_min - 15.0)
    
    # Konwersja na godziny dla wholePRx
    global_start_h = global_start_min / 60.0
    global_end_h = global_end_min / 60.0
    
    # Maska dla całego ustalonego okna (wholePRx)
    whole_mask = (czas >= global_start_h) & (czas <= global_end_h)

    # --- 2. Tworzymy maskę do pominięcia artefaktów (skip_mask) ---
    skip_mask = np.zeros(len(PRx), dtype=bool)
    if skip_time_sectors is not None:
        for sk_start, sk_end in skip_time_sectors:
            skip_mask |= (czas >= sk_start) & (czas <= sk_end)

    # --- 3. Inicjalizacja masek dla event i normal ---
    event_mask = np.zeros(len(PRx), dtype=bool)
    normal_mask = np.zeros(len(PRx), dtype=bool)

    # Budowanie masek z dokładnych przedziałów dla każdego zdarzenia
    for seg in event_time_sectors:
        e_start = seg['event_start_min'] / 60.0
        e_end = seg['event_end_min'] / 60.0
        p_start = seg['pre_event_start_min'] / 60.0
        
        # Event: od event_start_min do event_end_min
        event_mask |= (czas >= e_start) & (czas <= e_end)
        
        # Normal/Pre-event: od pre_event_start_min do event_start_min
        normal_mask |= (czas >= p_start) & (czas < e_start)

    # --- 4. Łączenie warunków z priorytetami ---
    # Wykluczamy skip_mask ze wszystkich stref
    is_event = event_mask & ~skip_mask
    
    # Zabezpieczenie, by normal nie nachodził na żaden event
    is_normal = normal_mask & ~event_mask & ~skip_mask 
    
    # whole obejmuje wyliczony zakres z uwzględnieniem dodatkowej godziny na końcu i 15 min na początku
    is_whole = whole_mask & ~skip_mask

    # --- 5. Wycinanie odpowiednich wierszy z macierzy ---
    eventPRx = PRx[is_event]
    normalPRx = PRx[is_normal]
    wholePRx = PRx[is_whole]

    return eventPRx, normalPRx, wholePRx

def kolmogorovSmirnov(event_vals, normal_vals):
    # 1. Wyciągamy TYLKO wartości PRx (indeks 1, bo indeks 0 to czas)
    event_vals = event_vals[:, 1]
    normal_vals = normal_vals[:, 1]

    # 2. Filtrujemy dane z wartości NaN
    event_vals_clean = event_vals[~np.isnan(event_vals)]
    normal_vals_clean = normal_vals[~np.isnan(normal_vals)]

    # Zabezpieczenie przed pustymi tablicami
    if len(event_vals_clean) == 0 or len(normal_vals_clean) == 0:
        print("Błąd: Jedna z prób jest pusta. Nie można wykonać testu K-S.")
        return None, None

    # 3. Wykonujemy test Kołmogorowa-Smirnowa
    ks_stat, p_value = stats.ks_2samp(event_vals_clean, normal_vals_clean)

    print(f"Statystyka K-S: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4e}")

    # Prosta interpretacja w konsoli
    alpha = 0.05
    if p_value < alpha:
        print("Odrzucamy hipotezę zerową: Rozkłady PRx SĄ istotnie różne.")
    else:
        print("Brak podstaw do odrzucenia hipotezy zerowej: Rozkłady PRx są do siebie podobne.")

    # ==========================================
    # 4. TWORZENIE WYKRESÓW (eCDF oraz PDF)
    # ==========================================
    
    # --- Dane do eCDF ---
    x_event = np.sort(event_vals_clean)
    y_event = np.arange(1, len(x_event) + 1) / len(x_event)
    
    x_normal = np.sort(normal_vals_clean)
    y_normal = np.arange(1, len(x_normal) + 1) / len(x_normal)

    # --- Dane do PDF (Kernel Density Estimation) ---
    x_pdf = np.linspace(-1.1, 1.1, 500) # Zakres osi X dla PRx
    
    # KDE wymaga, aby próba miała przynajmniej 2 punkty
    if len(normal_vals_clean) > 1:
        kde_normal = stats.gaussian_kde(normal_vals_clean)
        y_pdf_normal = kde_normal(x_pdf)
    else:
        y_pdf_normal = np.zeros_like(x_pdf)
        
    if len(event_vals_clean) > 1:
        kde_event = stats.gaussian_kde(event_vals_clean)
        y_pdf_event = kde_event(x_pdf)
    else:
        y_pdf_event = np.zeros_like(x_pdf)

    # --- Inicjalizacja układu subplots (1 wiersz, 2 kolumny) ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Dystrybuanta Empiryczna (eCDF)", 
            "Gęstość Prawdopodobieństwa (PDF)"
        ),
        horizontal_spacing=0.1
    )

    # -- WYKRES 1: eCDF (Lewa strona) --
    fig.add_trace(go.Scatter(
        x=x_normal, y=y_normal, mode='lines', 
        name='Strefy normalne (Normal)',
        line=dict(color='limegreen', width=2),
        legendgroup='normal'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_event, y=y_event, mode='lines', 
        name='Strefy zdarzeń (Event)',
        line=dict(color='crimson', width=2),
        legendgroup='event'
    ), row=1, col=1)

    # -- WYKRES 2: PDF (Prawa strona) --
    # Dodajemy cieniowanie obszaru pod krzywą gęstości (fill='tozeroy')
    fig.add_trace(go.Scatter(
        x=x_pdf, y=y_pdf_normal, mode='lines', 
        name='Strefy normalne (Normal)',
        line=dict(color='limegreen', width=2),
        fill='tozeroy', fillcolor='rgba(50, 205, 50, 0.2)',
        legendgroup='normal', showlegend=False # Ukrywamy duplikaty w legendzie
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x_pdf, y=y_pdf_event, mode='lines', 
        name='Strefy zdarzeń (Event)',
        line=dict(color='crimson', width=2),
        fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.2)',
        legendgroup='event', showlegend=False
    ), row=1, col=2)

    # --- Konfiguracja układu ---
    tytul_wykresu = (
        f"Wizualizacja testu Kołmogorowa-Smirnowa<br>"
        f"<sup>Statystyka K-S: <b>{ks_stat:.4f}</b> | P-value: <b>{p_value:.4e}</b></sup>"
    )

    fig.update_layout(
        title=dict(text=tytul_wykresu, x=0.5, y=0.95),
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="top", 
            y=-0.1,  # Zmienione: podniesiono lekko wyżej, żeby na pewno weszło w kadr
            xanchor="center", 
            x=0.5
        ),
        margin=dict(t=120, b=120),  # Zmienione: znacznie większy margines na dole na legendę
        width=1200,
        height=650  # Zmienione: lekko podbita całkowita wysokość obszaru roboczego
    )

    # Formatowanie osi dla pierwszego wykresu (eCDF)
    fig.update_xaxes(title_text="Wartość PRx", range=[-1.1, 1.1], gridcolor='rgba(200, 200, 200, 0.3)', row=1, col=1)
    fig.update_yaxes(title_text="Skumulowane Prawdopodobieństwo", range=[0, 1.05], gridcolor='rgba(200, 200, 200, 0.3)', row=1, col=1)

    # Formatowanie osi dla drugiego wykresu (PDF)
    fig.update_xaxes(title_text="Wartość PRx", range=[-1.1, 1.1], gridcolor='rgba(200, 200, 200, 0.3)', row=1, col=2)
    fig.update_yaxes(title_text="Gęstość", gridcolor='rgba(200, 200, 200, 0.3)', row=1, col=2)

    fig.show()

    return ks_stat, p_value