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

def defineTimeZones(PRx: np.ndarray, start: float, end: float, event_time_sectors: list, skip_time_sectors: list = None):
    """
    Filtruje macierz wyników PRx na podstawie globalnego zakresu czasu, 
    listy zdefiniowanych sektorów zdarzeń (jako słowniki w minutach) oraz sektorów odrzuconych.
    
    Argumenty:
    PRx: macierz NumPy o kształcie (N, 2), gdzie kolumna 0 to czas (h), a kolumna 1 to wartość PRx.
    start: globalny czas początkowy w godzinach (np. 2.5).
    end: globalny czas końcowy w godzinach (np. 17.75).
    event_time_sectors: lista słowników z czasem w minutach, 
                        np. [{'event_start_min': 75, 'event_end_min': 100}, {'event_start_min': 200, 'event_end_min': 250}]
    skip_time_sectors: lista przedziałów do całkowitego pominięcia w godzinach, 
                       np. [[3.5, 3.6], [10.0, 10.2]] (domyślnie None).
    
    Zwraca:
    eventPRx: macierz z danymi wewnątrz zdarzeń (z wykluczeniem skip_time_sectors).
    normalPRx: macierz z danymi poza zdarzeniami (z wykluczeniem skip_time_sectors).
    wholePRx: macierz z całymi danymi wewnątrz globalnego czasu (z wykluczeniem skip_time_sectors).
    """
    # Wyciągamy całą kolumnę czasu (w godzinach) dla wygody
    czas = PRx[:, 0]
    
    # 1. Tworzymy maskę globalną (czy dany punkt mieści się między 'start' a 'end')
    global_mask = (czas >= start) & (czas <= end)
    
    # 2. Tworzymy maskę do pominięcia (skip_mask) - na podstawie godzin
    skip_mask = np.zeros(len(PRx), dtype=bool)
    if skip_time_sectors is not None:
        for sk_start, sk_end in skip_time_sectors:
            skip_mask |= (czas >= sk_start) & (czas <= sk_end)
            
    # 3. Tworzymy maskę zdarzeń na podstawie listy słowników
    event_mask = np.zeros(len(PRx), dtype=bool)
    for seg in event_time_sectors:
        # Konwersja z minut na godziny
        e_start = seg['event_start_min'] / 60.0
        e_end = seg['event_end_min'] / 60.0
        # Dodajemy zdarzenie do ogólnej maski zdarzeń (operator logiczny LUB: '|')
        event_mask |= (czas >= e_start) & (czas <= e_end)
        
    # 4. Łączymy maski w ostateczne warunki
    # eventPRx: w przedziale globalnym ORAZ w którymkolwiek zdarzeniu ORAZ NIE pomijane
    is_event = global_mask & event_mask & ~skip_mask
    
    # normalPRx: w przedziale globalnym ORAZ NIE w zdarzeniach ORAZ NIE pomijane
    is_normal = global_mask & ~event_mask & ~skip_mask
    
    # wholePRx: w przedziale globalnym ORAZ NIE pomijane (zawiera w sobie event i normal)
    is_whole = global_mask & ~skip_mask
    
    # 5. Wycinamy odpowiednie wiersze z macierzy
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
        title=dict(text=tytul_wykresu, x=0.5),
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.05,
            xanchor="center", x=0.5
        ),
        width=1200,
        height=600
    )

    # Formatowanie osi dla pierwszego wykresu (eCDF)
    fig.update_xaxes(title_text="Wartość PRx", range=[-1.1, 1.1], gridcolor='rgba(200, 200, 200, 0.3)', row=1, col=1)
    fig.update_yaxes(title_text="Skumulowane Prawdopodobieństwo", range=[0, 1.05], gridcolor='rgba(200, 200, 200, 0.3)', row=1, col=1)

    # Formatowanie osi dla drugiego wykresu (PDF)
    fig.update_xaxes(title_text="Wartość PRx", range=[-1.1, 1.1], gridcolor='rgba(200, 200, 200, 0.3)', row=1, col=2)
    fig.update_yaxes(title_text="Gęstość", gridcolor='rgba(200, 200, 200, 0.3)', row=1, col=2)

    fig.show()

    return ks_stat, p_value