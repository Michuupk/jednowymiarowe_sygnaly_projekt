import sys
import os
import numpy as np
from scipy import signal
import wfdb
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QPushButton, QHBoxLayout, QFileDialog, QComboBox, 
                               QLabel, QMessageBox, QSlider, QDoubleSpinBox)

class SignalAnalysisWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analiza Sygnałów 1D - PyQtGraph & WFDB")
        self.resize(1100, 750)

        # Konfiguracja globalna PyQtGraph (ustawienie jasnego motywu, podobnego do Matplotlib)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 1. Wykres PyQtGraph
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', "Czas", units='s')
        self.plot_widget.setLabel('left', "Amplituda")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Wyłączamy domyślne przesuwanie osi X myszką, aby slider miał pełną kontrolę
        self.plot_widget.setMouseEnabled(x=False, y=True)
        
        # Inicjalizacja krzywej (PlotDataItem), którą będziemy tylko aktualizować (to daje wydajność)
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen(color='#1f77b4', width=1.5))
        
        layout.addWidget(self.plot_widget)

        # 2. Panel okna czasowego (Slider)
        slider_layout = QHBoxLayout()
        
        self.label_window = QLabel("Szerokość okna [s]:")
        self.spin_window = QDoubleSpinBox()
        self.spin_window.setRange(0.1, 10000.0)
        self.spin_window.setValue(2.0)
        self.spin_window.setSingleStep(0.5)
        self.spin_window.valueChanged.connect(self.update_slider_range)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_move)
        self.slider_scale = 100.0 

        slider_layout.addWidget(self.label_window)
        slider_layout.addWidget(self.spin_window)
        slider_layout.addWidget(self.slider)
        
        layout.addLayout(slider_layout)
        layout.addWidget(self._create_separator())

        # 3. Panel kontrolny (WFDB)
        self.current_db_path = ""
        wfdb_layout = QHBoxLayout()
        
        self.btn_select_folder = QPushButton("Wybierz folder bazy WFDB")
        self.btn_select_folder.clicked.connect(self.select_folder)
        
        self.combo_records = QComboBox()
        self.combo_records.setMinimumWidth(150)
        
        self.btn_load_wfdb = QPushButton("Wczytaj sygnał")
        self.btn_load_wfdb.clicked.connect(self.load_wfdb_signal)
        self.btn_load_wfdb.setEnabled(False) 

        wfdb_layout.addWidget(self.btn_select_folder)
        wfdb_layout.addWidget(QLabel("Rekord:"))
        wfdb_layout.addWidget(self.combo_records)
        wfdb_layout.addWidget(self.btn_load_wfdb)
        wfdb_layout.addStretch() 
        
        layout.addLayout(wfdb_layout)
        layout.addWidget(self._create_separator())

        # 4. ROZSZERZALNY PANEL PRZETWARZANIA SYGNAŁÓW
        processing_layout = QHBoxLayout()
        
        self.btn_reset = QPushButton("Sygnał testowy (10s)")
        self.btn_reset.clicked.connect(self.generate_test_signal)
        
        self.combo_methods = QComboBox()
        # --- REJESTR METOD PRZETWARZANIA ---
        # Aby dodać nową metodę, po prostu dopisz ją do tego słownika!
        self.processing_methods = {
            "Oryginał (Bez filtru)": self.process_none,
            "Filtr dolnoprzepustowy (SciPy - Butterworth 3Hz)": self.process_lowpass,
            "Wygładzanie (Średnia krocząca - 5 próbek)": self.process_moving_average
        }
        self.combo_methods.addItems(self.processing_methods.keys())
        
        self.btn_apply_method = QPushButton("Zastosuj metodę")
        self.btn_apply_method.clicked.connect(self.apply_processing)

        processing_layout.addWidget(self.btn_reset)
        processing_layout.addWidget(QLabel("Metoda przetwarzania:"))
        processing_layout.addWidget(self.combo_methods)
        processing_layout.addWidget(self.btn_apply_method)
        processing_layout.addStretch()
        
        layout.addLayout(processing_layout)

        # Zmienne przechowujące dane
        self.t = np.array([])
        self.y_raw = np.array([])       # Oryginalny, nietknięty sygnał
        self.y_processed = np.array([]) # Sygnał po analizie/filtracji
        self.fs = 100

        self.generate_test_signal()

    def _create_separator(self):
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #cccccc;")
        return line

    # --- LOGIKA SLIDERA I PRZEWIJANIA (PyQtGraph) ---

    def update_slider_range(self):
        if len(self.t) == 0: return

        total_time = self.t[-1]
        window_size = self.spin_window.value()

        if window_size >= total_time:
            self.slider.setEnabled(False)
            self.plot_widget.setXRange(0, total_time, padding=0)
        else:
            self.slider.setEnabled(True)
            max_slider_val = int((total_time - window_size) * self.slider_scale)
            self.slider.setRange(0, max_slider_val)
            self.on_slider_move(self.slider.value())

    def on_slider_move(self, value):
        if len(self.t) == 0: return
        start_time = value / self.slider_scale
        end_time = start_time + self.spin_window.value()
        
        # Błyskawiczna zmiana zakresu widoku w PyQtGraph
        self.plot_widget.setXRange(start_time, end_time, padding=0)

    # --- LOGIKA WFDB ---

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz folder z danymi WFDB")
        if folder_path:
            self.current_db_path = folder_path
            self.combo_records.clear()
            records = sorted([os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('.hea')])
            
            if records:
                self.combo_records.addItems(records)
                self.btn_load_wfdb.setEnabled(True)
                self.btn_select_folder.setText(f"Folder: {os.path.basename(folder_path)}")
            else:
                self.btn_load_wfdb.setEnabled(False)
                QMessageBox.warning(self, "Brak plików", "Nie znaleziono plików .hea.")

    def load_wfdb_signal(self):
        record_name = self.combo_records.currentText()
        if not record_name or not self.current_db_path: return
        try:
            record_path = os.path.join(self.current_db_path, record_name)
            record = wfdb.rdrecord(record_path)
            
            self.y_raw = record.p_signal[:, 0]
            self.fs = record.fs
            self.t = np.linspace(0, len(self.y_raw) / self.fs, len(self.y_raw), endpoint=False)
            
            self.apply_processing() # Aplikuje domyślną metodę (Oryginał)
            self.plot_widget.setTitle(f"[{record_name}] Fs: {self.fs} Hz")
            
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać sygnału:\n{str(e)}")

    def generate_test_signal(self):
        self.fs = 100 
        self.t = np.linspace(0, 10, 10 * self.fs, endpoint=False)
        clean = np.sin(2 * np.pi * 1 * self.t)
        self.y_raw = clean + np.random.normal(0, 0.5, self.t.shape)
        
        self.combo_methods.setCurrentIndex(0) # Reset comboboxa do oryginału
        self.apply_processing()
        self.plot_widget.setTitle("Sygnał testowy")

    # --- ROZSZERZALNE PRZETWARZANIE SYGNAŁÓW ---

    def apply_processing(self):
        if len(self.y_raw) == 0: return
        
        # Pobieranie nazwy wybranej metody i odpowiadającej jej funkcji ze słownika
        selected_method_name = self.combo_methods.currentText()
        processing_function = self.processing_methods[selected_method_name]
        
        try:
            # Wywołanie wybranej funkcji na surowym sygnale
            self.y_processed = processing_function(self.y_raw)
            self.update_plot()
        except Exception as e:
            QMessageBox.critical(self, "Błąd przetwarzania", f"Wystąpił błąd metody:\n{str(e)}")

    # 1. Metoda: Brak zmian
    def process_none(self, y):
        return y.copy()

    # 2. Metoda: Filtr dolnoprzepustowy
    def process_lowpass(self, y):
        cutoff = 3.0 / (0.5 * self.fs)
        b, a = signal.butter(4, cutoff, btype='low')
        return signal.filtfilt(b, a, y)

    # 3. Metoda: Średnia krocząca (przykład innej techniki)
    def process_moving_average(self, y):
        window = 5
        weights = np.repeat(1.0, window) / window
        return np.convolve(y, weights, 'same')

    # --- RYSOWANIE ---

    def update_plot(self):
        """Aktualizuje dane na wykresie bez niszczenia samego obiektu wykresu (bardzo szybkie)."""
        # Kolor zależy od tego, czy używamy filtru
        color = '#1f77b4' if self.combo_methods.currentIndex() == 0 else '#d62728'
        self.plot_curve.setPen(pg.mkPen(color=color, width=1.5))
        
        # Błyskawiczne wrzucenie nowych danych
        self.plot_curve.setData(self.t, self.y_processed)
        
        # Resetujemy auto-skalowanie osi Y dla nowych danych
        self.plot_widget.autoRange(padding=0.1)
        
        # Ustawiamy suwak na początek i przeliczamy zakres
        self.slider.setValue(0)
        self.update_slider_range()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalAnalysisWindow()
    window.show()
    sys.exit(app.exec())