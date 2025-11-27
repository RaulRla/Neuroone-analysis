import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

class EEGPreprocessor:
    def __init__(self, file_path, sample_rate=250, num_channels=4):
        """
        Inicializa o preprocessador de EEG

        Args:
            file_path: caminho do arquivo .txt
            sample_rate: taxa de amostragem (Hz)
            num_channels: número de canais
        """
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.raw_data = None
        self.filtered_data = None
        self.load_data(file_path)

    def load_data(self, file_path):
        """Carrega os dados do arquivo .txt"""
        try:
            with open(file_path, 'r') as f:
                values = f.read().strip().split()
                values = [float(v) for v in values]

            # Organizar em matriz (canais x amostras)
            num_samples = len(values) // self.num_channels
            self.raw_data = np.array(values[:num_samples * self.num_channels])
            self.raw_data = self.raw_data.reshape(self.num_channels, num_samples)

            print(f"✓ Dados carregados: {self.num_channels} canais, {num_samples} amostras")
            print(f"✓ Duração: {num_samples/self.sample_rate:.2f} segundos")
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            raise

    def apply_bandpass_filter(self, lowcut=0.5, highcut=50, order=4):
        """
        Aplica filtro passa-banda Butterworth

        Args:
            lowcut: frequência de corte inferior (Hz)
            highcut: frequência de corte superior (Hz)
            order: ordem do filtro
        """
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = signal.butter(order, [low, high], btype='band')

        self.filtered_data = np.zeros_like(self.raw_data)
        for ch in range(self.num_channels):
            self.filtered_data[ch] = signal.filtfilt(b, a, self.raw_data[ch])

        print(f"✓ Filtro passa-banda aplicado: {lowcut}-{highcut} Hz")

    def apply_notch_filter(self, freq=60, Q=30):
        """
        Aplica filtro notch para remover interferência de linha de força

        Args:
            freq: frequência a ser removida (Hz)
            Q: fator de qualidade
        """
        if self.filtered_data is None:
            data = self.raw_data
        else:
            data = self.filtered_data

        b, a = signal.iirnotch(freq, Q, self.sample_rate)

        for ch in range(self.num_channels):
            data[ch] = signal.filtfilt(b, a, data[ch])

        self.filtered_data = data
        print(f"✓ Filtro notch aplicado: {freq} Hz")

    def remove_artifacts(self, threshold=3):
        """
        Remove artefatos baseado em desvio padrão

        Args:
            threshold: número de desvios padrão para considerar artefato
        """
        if self.filtered_data is None:
            data = self.raw_data.copy()
        else:
            data = self.filtered_data.copy()

        for ch in range(self.num_channels):
            mean = np.mean(data[ch])
            std = np.std(data[ch])

            # Identificar outliers
            z_scores = np.abs((data[ch] - mean) / std)
            outliers = z_scores > threshold

            # Substituir outliers pela média
            data[ch][outliers] = mean

            print(f"  Canal {ch+1}: {np.sum(outliers)} artefatos removidos")

        self.filtered_data = data

    def normalize(self):
        """Normaliza os dados (z-score)"""
        if self.filtered_data is None:
            data = self.raw_data.copy()
        else:
            data = self.filtered_data.copy()

        for ch in range(self.num_channels):
            mean = np.mean(data[ch])
            std = np.std(data[ch])
            data[ch] = (data[ch] - mean) / std

        self.filtered_data = data
        print("✓ Dados normalizados (z-score)")

    def calculate_power_spectrum(self, channel):
        """
        Calcula o espectro de potência para um canal

        Args:
            channel: índice do canal (0-3)
        """
        if self.filtered_data is None:
            data = self.raw_data[channel]
        else:
            data = self.filtered_data[channel]

        # Calcular FFT
        N = len(data)
        yf = fft(data)
        xf = fftfreq(N, 1/self.sample_rate)

        # Apenas frequências positivas
        mask = xf >= 0
        xf = xf[mask]
        power = np.abs(yf[mask])**2 / N

        return xf, power

    def extract_frequency_bands(self, channel):
        """
        Extrai potência das bandas de frequência

        Args:
            channel: índice do canal (0-3)
        """
        xf, power = self.calculate_power_spectrum(channel)

        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'LowAlpha': (8, 10),
            'HighAlpha': (10, 13),
            'LowBeta': (13, 16),
            'HighBeta': (16, 30),
            'LowGamma': (30, 40),
            'MiddleGamma': (40, 50)
        }

        band_powers = {}
        for band_name, (low, high) in bands.items():
            mask = (xf >= low) & (xf <= high)
            band_powers[band_name] = np.sum(power[mask])

        return band_powers

    def calculate_attention_meditation(self, band_powers):
        """
        Calcula valores reais de Atenção (Att) e Meditação (Med) baseado nas bandas de frequência
        
        Baseado em pesquisa neurocientífica:
        - Atenção: correlacionada com atividade em Beta e Gamma (frequências de concentração)
        - Meditação: correlacionada com atividade em Alpha e Theta (frequências de relaxamento)
        
        Args:
            band_powers: dicionário com potências das bandas
        
        Returns:
            tuple: (att_value, med_value) - valores entre 0 e 100
        """
        # Extrair potências das bandas
        delta = band_powers.get('Delta', 0)
        theta = band_powers.get('Theta', 0)
        low_alpha = band_powers.get('LowAlpha', 0)
        high_alpha = band_powers.get('HighAlpha', 0)
        low_beta = band_powers.get('LowBeta', 0)
        high_beta = band_powers.get('HighBeta', 0)
        low_gamma = band_powers.get('LowGamma', 0)
        middle_gamma = band_powers.get('MiddleGamma', 0)
        
        # Agrupar bandas por função cognitiva
        alpha_total = low_alpha + high_alpha
        beta_total = low_beta + high_beta
        gamma_total = low_gamma + middle_gamma
        theta_total = theta
        
        # Potência total
        total_power = delta + theta_total + alpha_total + beta_total + gamma_total
        
        if total_power == 0:
            return 0, 0
        
        # Normalizar potências (0-1)
        alpha_norm = alpha_total / total_power
        beta_norm = beta_total / total_power
        gamma_norm = gamma_total / total_power
        theta_norm = theta_total / total_power
        delta_norm = delta / total_power
        
        # Calcular Atenção (Att)
        # Baseado em: Beta + Gamma (frequências de concentração)
        # Inversamente correlacionado com: Alpha + Theta (frequências de relaxamento)
        attention_index = (beta_norm + gamma_norm * 0.8) - (alpha_norm * 0.3 + theta_norm * 0.2)
        att_value = np.clip(attention_index * 100, 0, 100)
        
        # Calcular Meditação (Med)
        # Baseado em: Alpha + Theta (frequências de relaxamento/meditação)
        # Inversamente correlacionado com: Beta + Gamma (frequências de atenção)
        meditation_index = (alpha_norm + theta_norm * 0.7) - (beta_norm * 0.3 + gamma_norm * 0.2)
        med_value = np.clip(meditation_index * 100, 0, 100)
        
        return att_value, med_value

    def process_to_dataframe(self):
        """
        Processa os dados e retorna um DataFrame no formato compatível com o app Streamlit
        """
        # Aplicar pipeline de processamento completo
        self.apply_bandpass_filter(lowcut=0.5, highcut=50, order=4)
        self.apply_notch_filter(freq=60, Q=30)
        self.remove_artifacts(threshold=3)
        self.normalize()

        # Calcular potências médias das bandas de frequência para cada amostra
        num_samples = self.filtered_data.shape[1]
        
        data_rows = []
        base_time = pd.Timestamp.now()
        
        # Janela deslizante para calcular bandas de frequência em segmentos
        window_size = min(256, num_samples // 10)  # Janela adaptável
        
        for i in range(0, num_samples, window_size):
            # Extrair segmento de dados
            end_idx = min(i + window_size, num_samples)
            segment_data = self.filtered_data[:, i:end_idx]
            
            # Calcular potências das bandas para este segmento
            segment_band_powers = {}
            for band in ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma']:
                segment_band_powers[band] = 0.0
            
            for channel in range(self.num_channels):
                # Calcular FFT para este segmento
                N = segment_data.shape[1]
                if N < 2:
                    continue
                    
                yf = fft(segment_data[channel])
                xf = fftfreq(N, 1/self.sample_rate)
                
                # Apenas frequências positivas
                mask = xf >= 0
                xf = xf[mask]
                power = np.abs(yf[mask])**2 / N
                
                # Extrair potências por banda
                bands = {
                    'Delta': (0.5, 4),
                    'Theta': (4, 8),
                    'LowAlpha': (8, 10),
                    'HighAlpha': (10, 13),
                    'LowBeta': (13, 16),
                    'HighBeta': (16, 30),
                    'LowGamma': (30, 40),
                    'MiddleGamma': (40, 50)
                }
                
                for band_name, (low, high) in bands.items():
                    band_mask = (xf >= low) & (xf <= high)
                    segment_band_powers[band_name] += np.sum(power[band_mask])
            
            # Calcular médias entre canais
            for band in segment_band_powers:
                segment_band_powers[band] /= self.num_channels
            
            # Calcular Att e Med baseado nas potências reais
            att_value, med_value = self.calculate_attention_meditation(segment_band_powers)
            
            # Criar linha de dados
            row_data = segment_band_powers.copy()
            row_data['Att'] = att_value
            row_data['Med'] = med_value
            row_data['Datetime'] = base_time + pd.Timedelta(seconds=i/self.sample_rate)
            
            data_rows.append(row_data)
        
        # Criar DataFrame
        df = pd.DataFrame(data_rows)
        
        # Reordenar colunas para compatibilidade
        column_order = ['Datetime', 'Att', 'Med', 'Delta', 'Theta', 'LowAlpha', 'HighAlpha', 
                       'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma']
        df = df[[col for col in column_order if col in df.columns]]
        
        return df

def process_txt_to_dataframe(file_path, sample_rate=250, num_channels=4):
    """
    Função conveniente para processar arquivo .txt e retornar DataFrame compatível
    
    Args:
        file_path: caminho do arquivo .txt
        sample_rate: taxa de amostragem (Hz)
        num_channels: número de canais
    
    Returns:
        pd.DataFrame: DataFrame no formato compatível com o app Streamlit
    """
    try:
        preprocessor = EEGPreprocessor(file_path, sample_rate, num_channels)
        df = preprocessor.process_to_dataframe()
        print(f"✅ TXT processado com sucesso: {len(df)} amostras geradas")
        return df
    except Exception as e:
        print(f"❌ Erro ao processar TXT: {e}")
        # Retornar DataFrame vazio em caso de erro
        return pd.DataFrame()
