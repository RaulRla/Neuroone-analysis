#!/usr/bin/env python3
# processador_eeg_minimal.py
# Versão em Português (PT-BR) do processador minimalista - gera relatórios PDF

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_theme(style='darkgrid')
plt.rcParams.update({
    'figure.figsize': (11, 8.5),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

REQUIRED_COLS = [
    'Delta', 'Theta', 'LowAlpha', 'HighAlpha',
    'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma',
    'Att', 'Med'
]

class ProcessadorEEG:
    """Processador EEG - versão minimalista (PT-BR)"""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.filename = self.csv_path.stem
        self.df = None
        self.df_clean = None
        self.bandas = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha',
                       'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma']
        self.nomes_pt = {
            'Delta': 'Delta', 'Theta': 'Theta', 'LowAlpha': 'Alpha Baixo', 'HighAlpha': 'Alpha Alto',
            'LowBeta': 'Beta Baixo', 'HighBeta': 'Beta Alto', 'LowGamma': 'Gamma Baixo', 'MiddleGamma': 'Gamma Médio'
        }

    def carregar_dados(self):
        print(f"  Carregando: {self.filename} ...")
        self.df = pd.read_csv(self.csv_path)
        faltantes = [c for c in REQUIRED_COLS if c not in self.df.columns]
        if faltantes:
            raise ValueError(f"Colunas obrigatórias ausentes: {faltantes}")
        return True

    def preprocessar_dados(self):
        print("  Pré-processando dados...")
        self.df_clean = self.df.dropna().reset_index(drop=True)

        # Criar coluna Datetime se Date/Time existirem
        if 'Date' in self.df_clean.columns and 'Time' in self.df_clean.columns:
            try:
                self.df_clean['Datetime'] = pd.to_datetime(
                    self.df_clean['Date'].astype(str).str.strip() + ' ' + self.df_clean['Time'].astype(str).str.strip(),
                    dayfirst=False, errors='coerce'
                )
            except Exception:
                self.df_clean['Datetime'] = pd.NaT
        else:
            self.df_clean['Datetime'] = pd.NaT

        # Índices e razões (proteções contra divisão por zero)
        alpha = self.df_clean['LowAlpha'] + self.df_clean['HighAlpha']
        beta = self.df_clean['LowBeta'] + self.df_clean['HighBeta']
        gamma = self.df_clean['LowGamma'] + self.df_clean['MiddleGamma']
        theta = self.df_clean['Theta'].replace(0, np.nan)

        self.df_clean['Razao_AlphaTheta'] = (alpha / theta).replace([np.inf, -np.inf], np.nan)
        self.df_clean['Razao_BetaAlpha'] = (beta / alpha).replace([np.inf, -np.inf], np.nan)
        self.df_clean['Razao_GammaBeta'] = (gamma / beta).replace([np.inf, -np.inf], np.nan)
        self.df_clean['Indice_Relaxamento'] = (alpha / beta.replace(0, np.nan)).fillna(0) * 100

        # Médias móveis - janela adaptável
        janela = min(30, max(1, len(self.df_clean)//10))
        self.df_clean['Att_MM'] = self.df_clean['Att'].rolling(window=janela, min_periods=1).mean()
        self.df_clean['Med_MM'] = self.df_clean['Med'].rolling(window=janela, min_periods=1).mean()

        # Picos por percentil
        self.df_clean['Pico_Att'] = self.df_clean['Att'] >= self.df_clean['Att'].quantile(0.85)
        self.df_clean['Pico_Med'] = self.df_clean['Med'] >= self.df_clean['Med'].quantile(0.85)

        # Classificação simples (thresholds adaptativos)
        att_thr = self.df_clean['Att'].quantile(0.66)
        med_thr = self.df_clean['Med'].quantile(0.66)

        def classificar(row):
            if row['Att'] >= att_thr and row['Med'] >= med_thr:
                return 'Meditação Focada'
            if row['Att'] >= att_thr:
                return 'Alta Concentração'
            if row['Med'] >= med_thr:
                return 'Meditação Profunda'
            if row['Razao_AlphaTheta'] > 1.5:
                return 'Alerta Relaxado'
            return 'Estado Normal'

        self.df_clean['Estado_Cerebral'] = self.df_clean.apply(classificar, axis=1)
        return True

    def gerar_relatorio(self, output_dir):
        print("  Gerando relatório PDF...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.filename}_relatorio_minimal.pdf"

        with PdfPages(output_path) as pdf:
            self._pagina_capa(pdf)
            self._pagina_ondas(pdf)
            self._pagina_att_med(pdf)
            self._pagina_correlacao_resumo(pdf)

        print(f" [OK] Relatório salvo: {output_path}")
        return str(output_path)

    def _pagina_capa(self, pdf):
        fig = plt.figure(figsize=(11,8.5))
        fig.text(0.5, 0.86, 'Relatório de Análise EEG', ha='center', fontsize=22, weight='bold')
        fig.text(0.5, 0.80, f'Participante: {self.filename}', ha='center', fontsize=14)
        fig.text(0.5, 0.76, f'Gerado em: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ha='center', fontsize=10)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _pagina_ondas(self, pdf):
        fig, ax = plt.subplots()
        t = np.arange(len(self.df_clean))
        palette = sns.color_palette('tab10', n_colors=len(self.bandas))
        for i, banda in enumerate(self.bandas):
            ax.plot(t, self.df_clean[banda], label=self.nomes_pt[banda], linewidth=1)
        ax.set_title('Ondas Cerebrais (visão compacta)')
        ax.set_xlabel('Amostras')
        ax.set_ylabel('Amplitude / Potência')
        ax.legend(ncol=2, fontsize=8)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _pagina_att_med(self, pdf):
        """Página com ATT e MED juntos: RAW (cores claras) + MA 30s (cores fortes)"""
        fig, ax = plt.subplots(figsize=(11, 8.5))

        # ATT
        ax.plot(
            self.df_clean.index,
            self.df_clean['Att'],
            color='#7fb3ff',       # azul claro
            alpha=0.4,
            label='Att (bruto)'
        )
        ax.plot(
            self.df_clean.index,
            self.df_clean['Att_MM'],
            color='#0056d6',       # azul escuro
            linewidth=2,
            label='Att (média móvel)'
        )

        # MED
        ax.plot(
            self.df_clean.index,
            self.df_clean['Med'],
            color='#93ff9c',       # verde claro
            alpha=0.4,
            label='Med (bruto)'
        )
        ax.plot(
            self.df_clean.index,
            self.df_clean['Med_MM'],
            color='#0f8a24',       # verde escuro
            linewidth=2,
            label='Med (média móvel)'
        )

        ax.set_title('Atenção e Meditação — Bruto vs Média Móvel 30s', fontsize=14)
        ax.set_xlabel('Amostras')
        ax.set_ylabel('Nível (0–100)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _pagina_correlacao_resumo(self, pdf):
        fig = plt.figure(figsize=(11,8.5))
        cols = self.bandas + ['Att','Med']
        corr = self.df_clean[cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0)
        plt.title('Matriz de Correlação')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def processar_pasta(pasta_entrada, pasta_saida=None):
    caminho = Path(pasta_entrada)
    if pasta_saida is None:
        pasta_saida = caminho / 'relatorios_minimal'
    caminho_saida = Path(pasta_saida)
    caminho_saida.mkdir(parents=True, exist_ok=True)

    arquivos = list(caminho.glob('*.csv'))
    if not arquivos:
        print(f"Nenhum CSV encontrado em {pasta_entrada}")
        return

    for f in arquivos:
        try:
            proc = ProcessadorEEG(str(f))
            proc.carregar_dados()
            proc.preprocessar_dados()
            proc.gerar_relatorio(caminho_saida)
        except Exception as e:
            print(f"Erro ao processar {f.name}: {e}")

if __name__ == '__main__':
    import sys
    pasta = sys.argv[1] if len(sys.argv) > 1 else '.'
    processar_pasta(pasta)
