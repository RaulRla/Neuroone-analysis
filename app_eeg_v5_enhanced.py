# app_eeg_v5_enhanced.py (PT-BR) — Análise Completa com Jupyter Notebook Features
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from processador_eeg_minimal import ProcessadorEEG
from eeg_preprocessor import process_txt_to_dataframe
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time
import os
from scipy import stats, signal

try:
    from PyPDF2 import PdfMerger
except Exception:
    PdfMerger = None

st.set_page_config(page_title="Analisador Avançado", layout="wide", initial_sidebar_state="expanded")
st.title("🧠 Analisador — Análise Completa & Inteligente")
st.markdown("*Análise avançada com classificação de estados cerebrais, análise espectral e insights estatísticos*")

# ---------- Configurações ----------
DOWNSAMPLE_THRESHOLD = 2000
DOWNSAMPLE_FACTOR = 5
MAX_PLOT_POINTS = 4000
MA_WINDOW = "30s"
GRAPH_HEIGHT_MAIN = 700
GRAPH_HEIGHT_SECONDARY = 600
GRAPH_HEIGHT_SMALL = 450
MARGIN_LARGE = dict(t=80, b=80, l=80, r=80)
MARGIN_SMALL = dict(t=60, b=60, l=70, r=70)

# ---------- Helper Functions ----------
def carregar_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

def parse_datetime(df):
    df2 = df.copy()
    if 'Date' in df2.columns and 'Time' in df2.columns:
        df2['Datetime'] = pd.to_datetime(
            df2['Date'].astype(str).str.strip() + ' ' + df2['Time'].astype(str).str.strip(),
            dayfirst=False, errors='coerce'
        )
        if df2['Datetime'].isna().all():
            df2['Datetime'] = pd.to_datetime(
                df2['Date'].astype(str).str.strip() + ' ' + df2['Time'].astype(str).str.strip(),
                dayfirst=True, errors='coerce'
            )
    else:
        df2['Datetime'] = pd.NaT
    return df2

def dias_disponiveis(df):
    if 'Datetime' in df.columns and not df['Datetime'].isna().all():
        return sorted(df['Datetime'].dt.date.dropna().unique().tolist())
    return []

def get_plot_df(df):
    n = len(df)
    if n <= DOWNSAMPLE_THRESHOLD:
        return df.copy()
    factor = max(DOWNSAMPLE_FACTOR, int(np.ceil(n / MAX_PLOT_POINTS)))
    return df.iloc[::factor].reset_index(drop=True)

def segs_desde_meianoite(dt_series):
    return dt_series.dt.hour * 3600 + dt_series.dt.minute * 60 + dt_series.dt.second

def segundos_para_hms(s):
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{int(h):02d}:{int(m):02d}:{int(sec):02d}"

def agregar_bandas_e_ma(df, window_size=30):
    """Agrupa bandas em 5 camadas e calcula MA baseada em tempo."""
    df2 = df.copy()
    if 'Datetime' in df2.columns and pd.api.types.is_datetime64_any_dtype(df2['Datetime']):
        tmp = df2.set_index('Datetime')
        def col_sum(cols):
            available = [c for c in cols if c in tmp.columns]
            if not available:
                return pd.Series(0, index=tmp.index)
            return tmp[available].sum(axis=1)

        delta = col_sum(['Delta'])
        theta = col_sum(['Theta'])
        alpha = col_sum(['LowAlpha','HighAlpha'])
        beta = col_sum(['LowBeta','HighBeta'])
        gamma = col_sum(['LowGamma','MiddleGamma'])

        window_str = f"{window_size}s"
        df2['Delta_MA'] = delta.rolling(window_str, min_periods=1).mean().values
        df2['Theta_MA'] = theta.rolling(window_str, min_periods=1).mean().values
        df2['Alpha_MA'] = alpha.rolling(window_str, min_periods=1).mean().values
        df2['Beta_MA'] = beta.rolling(window_str, min_periods=1).mean().values
        df2['Gamma_MA'] = gamma.rolling(window_str, min_periods=1).mean().values
    else:
        def col_sum_samples(cols):
            available = [c for c in cols if c in df2.columns]
            if not available:
                return pd.Series(0, index=df2.index)
            return df2[available].sum(axis=1)
        delta = col_sum_samples(['Delta'])
        theta = col_sum_samples(['Theta'])
        alpha = col_sum_samples(['LowAlpha','HighAlpha'])
        beta = col_sum_samples(['LowBeta','HighBeta'])
        gamma = col_sum_samples(['LowGamma','MiddleGamma'])

        df2['Delta_MA'] = delta.rolling(window_size, min_periods=1).mean()
        df2['Theta_MA'] = theta.rolling(window_size, min_periods=1).mean()
        df2['Alpha_MA'] = alpha.rolling(window_size, min_periods=1).mean()
        df2['Beta_MA'] = beta.rolling(window_size, min_periods=1).mean()
        df2['Gamma_MA'] = gamma.rolling(window_size, min_periods=1).mean()
    return df2

def calcular_razoes_cerebrais(df):
    """Calcula razões entre bandas de frequência."""
    df2 = df.copy()
    
    # Calcular totais de bandas
    alpha_total = 0
    theta_total = 0
    beta_total = 0
    gamma_total = 0
    
    if 'LowAlpha' in df2.columns and 'HighAlpha' in df2.columns:
        alpha_total = df2['LowAlpha'] + df2['HighAlpha']
    if 'Theta' in df2.columns:
        theta_total = df2['Theta']
    if 'LowBeta' in df2.columns and 'HighBeta' in df2.columns:
        beta_total = df2['LowBeta'] + df2['HighBeta']
    if 'LowGamma' in df2.columns and 'MiddleGamma' in df2.columns:
        gamma_total = df2['LowGamma'] + df2['MiddleGamma']
    
    # Calcular razões (evitar divisão por zero)
    df2['AlphaTheta_Ratio'] = np.where(theta_total > 0, alpha_total / theta_total, 0)
    df2['BetaAlpha_Ratio'] = np.where(alpha_total > 0, beta_total / alpha_total, 0)
    df2['GammaBeta_Ratio'] = np.where(beta_total > 0, gamma_total / beta_total, 0)
    
    return df2

def classificar_estado_cerebral(row):
    """Classifica o estado mental baseado em Att/Med e razões de frequência."""
    try:
        if row['Att'] > 60 and row['Med'] > 60:
            return 'Meditação Focada'
        elif row['Att'] > 70:
            return 'Alto Foco'
        elif row['Med'] > 70:
            return 'Meditação Profunda'
        elif row.get('AlphaTheta_Ratio', 0) > 1.5:
            return 'Alerta Relaxado'
        elif row.get('BetaAlpha_Ratio', 0) > 2.0:
            return 'Pensamento Ativo'
        else:
            return 'Estado Normal'
    except:
        return 'Estado Normal'

def agregacao_semanal(df):
    if 'Datetime' not in df.columns or df['Datetime'].isna().all():
        return pd.DataFrame()
    agg = df.groupby(df['Datetime'].dt.date).agg(
        Att_mean = ('Att','mean') if 'Att' in df.columns else (lambda x: np.nan),
        Med_mean = ('Med','mean') if 'Med' in df.columns else (lambda x: np.nan),
        Att_max = ('Att','max') if 'Att' in df.columns else (lambda x: np.nan),
        Med_max = ('Med','max') if 'Med' in df.columns else (lambda x: np.nan),
        Att_min = ('Att','min') if 'Att' in df.columns else (lambda x: np.nan),
        Med_min = ('Med','min') if 'Med' in df.columns else (lambda x: np.nan),
        Att_std = ('Att','std') if 'Att' in df.columns else (lambda x: np.nan),
        Med_std = ('Med','std') if 'Med' in df.columns else (lambda x: np.nan),
        samples = ('Datetime','count')
    )
    agg = agg.reset_index().rename(columns={'Datetime':'date'})
    return agg

def calcular_metricas_avancadas(df):
    """Calcula métricas relevantes para análise EEG."""
    metricas = {}
    
    # Estatísticas de Atenção
    if 'Att' in df.columns:
        metricas['Att_media'] = df['Att'].mean()
        metricas['Att_mediana'] = df['Att'].median()
        metricas['Att_desvio'] = df['Att'].std()
        metricas['Att_max'] = df['Att'].max()
        metricas['Att_min'] = df['Att'].min()
        metricas['Att_q1'] = df['Att'].quantile(0.25)
        metricas['Att_q3'] = df['Att'].quantile(0.75)
        
    # Estatísticas de Meditação
    if 'Med' in df.columns:
        metricas['Med_media'] = df['Med'].mean()
        metricas['Med_mediana'] = df['Med'].median()
        metricas['Med_desvio'] = df['Med'].std()
        metricas['Med_max'] = df['Med'].max()
        metricas['Med_min'] = df['Med'].min()
        metricas['Med_q1'] = df['Med'].quantile(0.25)
        metricas['Med_q3'] = df['Med'].quantile(0.75)
        
    # Razões e índices
    alpha = 0
    theta = 0
    beta = 0
    gamma = 0
    
    if 'LowAlpha' in df.columns and 'HighAlpha' in df.columns:
        alpha = df['LowAlpha'].mean() + df['HighAlpha'].mean()
    if 'Theta' in df.columns:
        theta = df['Theta'].mean()
    if 'LowBeta' in df.columns and 'HighBeta' in df.columns:
        beta = df['LowBeta'].mean() + df['HighBeta'].mean()
    if 'LowGamma' in df.columns and 'MiddleGamma' in df.columns:
        gamma = df['LowGamma'].mean() + df['MiddleGamma'].mean()
        
    if theta > 0:
        metricas['Razao_Alpha_Theta'] = alpha / theta
    if alpha > 0:
        metricas['Razao_Beta_Alpha'] = beta / alpha
    if beta > 0:
        metricas['Razao_Gamma_Beta'] = gamma / beta
        
    return metricas

def calcular_potencia_media_bandas(df):
    """Calcula a potência média de cada banda de frequência."""
    brainwaves = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 
                  'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma']
    
    potencias = {}
    for wave in brainwaves:
        if wave in df.columns:
            potencias[wave] = df[wave].mean()
    
    return potencias

# ---------- Upload ----------
uploaded_files = st.file_uploader("📁 Carregue um ou mais arquivos (CSV ou TXT) com dados EEG", type=['csv', 'txt'], accept_multiple_files=True)
if not uploaded_files:
    st.info("⚙️ Carregue um ou mais arquivos CSV ou TXT para começar a análise completa dos seus dados EEG.")
    st.stop()

# Armazenar todos os arquivos carregados
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {}

# Processar arquivos carregados
tmpdir = tempfile.mkdtemp()
for uploaded_file in uploaded_files:
    if uploaded_file.name not in st.session_state.uploaded_data:
        tmp_file = Path(tmpdir) / uploaded_file.name
        with open(tmp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"⏳ Processando {uploaded_file.name}..."):
            if uploaded_file.name.lower().endswith('.csv'):
                # Processar arquivo CSV normalmente
                df_raw = carregar_csv(str(tmp_file))
                df_parsed = parse_datetime(df_raw)
                st.session_state.uploaded_data[uploaded_file.name] = df_parsed
                st.success(f"✅ CSV carregado: {uploaded_file.name}")
            elif uploaded_file.name.lower().endswith('.txt'):
                # Processar arquivo TXT usando o EEGPreprocessor
                try:
                    df_processed = process_txt_to_dataframe(str(tmp_file))
                    st.session_state.uploaded_data[uploaded_file.name] = df_processed
                    st.success(f"✅ TXT processado: {uploaded_file.name} - {len(df_processed)} amostras")
                    st.info(f"📊 Colunas disponíveis: {', '.join(df_processed.columns.tolist())}")
                except Exception as e:
                    st.error(f"❌ Erro ao processar arquivo TXT {uploaded_file.name}: {str(e)}")
                    continue
            else:
                st.warning(f"⚠️ Tipo de arquivo não suportado: {uploaded_file.name}")
                continue

# Sidebar - Seleção de arquivo
st.sidebar.header("📂 Seleção de Arquivo")
file_names = list(st.session_state.uploaded_data.keys())
selected_file = st.sidebar.selectbox("Escolha o arquivo para análise", options=file_names, index=0)

# Obter dataframe selecionado
df = st.session_state.uploaded_data[selected_file].copy()

# Mostrar informações do arquivo selecionado
st.sidebar.info(f"**Arquivo atual:** {selected_file}\n\n**Total de amostras:** {len(df)}")

# Sidebar - Configurações
st.sidebar.header("⚙️ Configurações de Análise")
smoothing_window = st.sidebar.slider("Janela de Suavização (segundos)", 10, 120, 30, 5)

# Calcular MA para Att/Med
if 'Att' in df.columns or 'Med' in df.columns:
    if 'Datetime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Datetime']):
        tmp = df.set_index('Datetime')
        window_str = f"{smoothing_window}s"
        if 'Att' in tmp.columns:
            df['Att_MA'] = tmp['Att'].rolling(window_str, min_periods=1).mean().values
        if 'Med' in tmp.columns:
            df['Med_MA'] = tmp['Med'].rolling(window_str, min_periods=1).mean().values
    else:
        if 'Att' in df.columns:
            df['Att_MA'] = df['Att'].rolling(smoothing_window, min_periods=1).mean()
        if 'Med' in df.columns:
            df['Med_MA'] = df['Med'].rolling(smoothing_window, min_periods=1).mean()

# Agregar bandas + MA
df = agregar_bandas_e_ma(df, smoothing_window)

# Calcular razões cerebrais
df = calcular_razoes_cerebrais(df)

# Classificar estados cerebrais
if 'Att' in df.columns and 'Med' in df.columns:
    df['Brain_State'] = df.apply(classificar_estado_cerebral, axis=1)

dias = dias_disponiveis(df)

# Sidebar - Seleção de dia
st.sidebar.header("📅 Seleção de Período")
if dias:
    dia_escolhido = st.sidebar.selectbox("Escolha um dia para análise", options=dias, index=0)
    dias_overlay = st.sidebar.multiselect("Sobrepor outros dias (comparação)", options=dias, default=[])
else:
    dia_escolhido = None
    dias_overlay = []

# Sidebar - Visualizações
st.sidebar.markdown("---")
st.sidebar.header("📈 Opções de Visualização")
show_raw_data = st.sidebar.checkbox("Mostrar dados brutos", value=False)
show_brain_states = st.sidebar.checkbox("Análise de Estados Cerebrais", value=True)
show_spectral = st.sidebar.checkbox("Análise Espectral", value=True)
show_smoothed_waves = st.sidebar.checkbox("Ondas Suavizadas", value=True)

# Sidebar - Filtros
st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtros")
if 'Att' in df.columns:
    att_min, att_max = float(df['Att'].min()), float(df['Att'].max())
    # Verificar se min e max são diferentes para evitar erro no slider
    if att_min == att_max:
        att_min = max(0, att_min - 5)  # Adicionar margem se valores forem iguais
        att_max = att_max + 5
    att_range = st.sidebar.slider("Intervalo Att", att_min, att_max, (att_min, att_max))
else:
    att_range = None
if 'Med' in df.columns:
    med_min, med_max = float(df['Med'].min()), float(df['Med'].max())
    # Verificar se min e max são diferentes para evitar erro no slider
    if med_min == med_max:
        med_min = max(0, med_min - 5)  # Adicionar margem se valores forem iguais
        med_max = med_max + 5
    med_range = st.sidebar.slider("Intervalo Med", med_min, med_max, (med_min, med_max))
else:
    med_range = None

df_filtrado = df.copy()
if att_range is not None:
    df_filtrado = df_filtrado[(df_filtrado['Att'] >= att_range[0]) & (df_filtrado['Att'] <= att_range[1])]
if med_range is not None:
    df_filtrado = df_filtrado[(df_filtrado['Med'] >= med_range[0]) & (df_filtrado['Med'] <= med_range[1])]

agg = agregacao_semanal(df_filtrado)

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral", 
    "🌊 Análise de Ondas Cerebrais", 
    "🧠 Estados Cerebrais", 
    "📈 Insights Estatísticos",
    "📅 Análise Semanal",
    "🌍 Análise Geral (Todas as Sessões)"
])

# ========== TAB 1: VISÃO GERAL ==========
with tab1:
    st.header("📊 Visão Geral e Séries Temporais")
    
    if dias:
        df_dia = df_filtrado[df_filtrado['Datetime'].dt.date == pd.to_datetime(dia_escolhido).date()].copy()
        if df_dia.empty:
            st.warning("❌ Nenhum dado para o dia selecionado após os filtros.")
        else:
            df_dia['sec_day'] = segs_desde_meianoite(df_dia['Datetime'])
            plot_df = get_plot_df(df_dia)

            # Métricas do dia
            st.subheader("📊 Resumo do Dia")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Att Média", f"{df_dia['Att'].mean():.1f}", 
                         delta=f"±{df_dia['Att'].std():.1f}")
            with col2:
                st.metric("Med Média", f"{df_dia['Med'].mean():.1f}", 
                         delta=f"±{df_dia['Med'].std():.1f}")
            with col3:
                st.metric("Att Máxima", f"{df_dia['Att'].max():.0f}")
            with col4:
                st.metric("Med Máxima", f"{df_dia['Med'].max():.0f}")

            st.markdown("---")

            # Gráfico principal
            st.subheader("💡 Atenção e Meditação — Evolução Temporal")
            
            fig_main = go.Figure()
            
            if show_raw_data:
                if 'Att' in plot_df.columns:
                    fig_main.add_trace(go.Scattergl(
                        x=plot_df['sec_day'], y=plot_df['Att'], 
                        mode='lines', name='Att (bruto)', 
                        line=dict(color='rgba(183, 214, 255, 0.3)', width=1),
                        hovertemplate='Att: %{y:.1f}<extra></extra>'
                    ))
                if 'Med' in plot_df.columns:
                    fig_main.add_trace(go.Scattergl(
                        x=plot_df['sec_day'], y=plot_df['Med'], 
                        mode='lines', name='Med (bruto)', 
                        line=dict(color='rgba(201, 243, 201, 0.3)', width=1),
                        hovertemplate='Med: %{y:.1f}<extra></extra>'
                    ))

            if 'Att_MA' in plot_df.columns:
                fig_main.add_trace(go.Scattergl(
                    x=plot_df['sec_day'], y=plot_df['Att_MA'], 
                    mode='lines', name=f'Atenção (MA {smoothing_window}s)', 
                    line=dict(color='#0056d6', width=4),
                    hovertemplate='Atenção: %{y:.1f}<extra></extra>'
                ))
            if 'Med_MA' in plot_df.columns:
                fig_main.add_trace(go.Scattergl(
                    x=plot_df['sec_day'], y=plot_df['Med_MA'], 
                    mode='lines', name=f'Meditação (MA {smoothing_window}s)', 
                    line=dict(color='#0f8a24', width=4),
                    hovertemplate='Meditação: %{y:.1f}<extra></extra>'
                ))

            fig_main.update_layout(
                height=GRAPH_HEIGHT_MAIN,
                title=dict(text="Evolução ao longo do dia", font=dict(size=20)),
                xaxis=dict(
                    title=dict(text='Horário do Dia', font=dict(size=16)),
                    tickfont=dict(size=14),
                    tickmode='array',
                    tickvals=np.linspace(plot_df['sec_day'].min(), plot_df['sec_day'].max(), 10),
                    ticktext=[segundos_para_hms(int(s)) for s in np.linspace(plot_df['sec_day'].min(), plot_df['sec_day'].max(), 10)],
                    gridcolor='rgba(200,200,200,0.3)'
                ),
                yaxis=dict(
                    title=dict(text='Nível (0-100)', font=dict(size=16)),
                    tickfont=dict(size=14),
                    gridcolor='rgba(200,200,200,0.3)'
                ),
                legend=dict(
                    orientation='h', 
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5,
                    font=dict(size=14)
                ),
                margin=MARGIN_LARGE,
                hovermode='x unified',
                template='plotly_white',
                plot_bgcolor='rgba(250,250,250,1)'
            )
            st.plotly_chart(fig_main, use_container_width=True)

            st.markdown("---")

            # Distribuições
            st.subheader("📊 Distribuições")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Att' in df_dia.columns:
                    fig_hist_att = go.Figure()
                    fig_hist_att.add_trace(go.Histogram(
                        x=df_dia['Att'],
                        nbinsx=30,
                        name='Atenção',
                        marker_color='#0056d6',
                        opacity=0.75
                    ))
                    fig_hist_att.update_layout(
                        title=dict(text='Distribuição de Atenção', font=dict(size=16)),
                        xaxis_title='Nível de Atenção',
                        yaxis_title='Frequência',
                        height=GRAPH_HEIGHT_SMALL,
                        margin=MARGIN_SMALL,
                        template='plotly_white',
                        showlegend=False
                    )
                    st.plotly_chart(fig_hist_att, use_container_width=True)
            
            with col2:
                if 'Med' in df_dia.columns:
                    fig_hist_med = go.Figure()
                    fig_hist_med.add_trace(go.Histogram(
                        x=df_dia['Med'],
                        nbinsx=30,
                        name='Meditação',
                        marker_color='#0f8a24',
                        opacity=0.75
                    ))
                    fig_hist_med.update_layout(
                        title=dict(text='Distribuição de Meditação', font=dict(size=16)),
                        xaxis_title='Nível de Meditação',
                        yaxis_title='Frequência',
                        height=GRAPH_HEIGHT_SMALL,
                        margin=MARGIN_SMALL,
                        template='plotly_white',
                        showlegend=False
                    )
                    st.plotly_chart(fig_hist_med, use_container_width=True)
    else:
        st.info("ℹ️ Nenhuma data detectada no CSV.")

# ========== TAB 2: ANÁLISE DE ONDAS CEREBRAIS ==========
with tab2:
    st.header("🌊 Análise de Ondas Cerebrais")
    
    if dias and not df_filtrado.empty:
        df_dia = df_filtrado[df_filtrado['Datetime'].dt.date == pd.to_datetime(dia_escolhido).date()].copy()
        
        if not df_dia.empty:
            df_dia['sec_day'] = segs_desde_meianoite(df_dia['Datetime'])
            plot_df = get_plot_df(df_dia)
            
            # Análise Espectral - Radar Chart
            if show_spectral:
                st.subheader("📡 Análise Espectral - Distribuição de Potência")
                
                potencias = calcular_potencia_media_bandas(df_dia)
                
                if potencias:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Radar chart
                        fig_radar = go.Figure(data=go.Scatterpolar(
                            r=list(potencias.values()),
                            theta=list(potencias.keys()),
                            fill='toself',
                            line_color='#0056d6',
                            fillcolor='rgba(0, 86, 214, 0.3)',
                            opacity=0.8
                        ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, max(potencias.values()) * 1.1]
                                )
                            ),
                            title=dict(text="Distribuição de Potência por Banda de Frequência", font=dict(size=16)),
                            height=GRAPH_HEIGHT_SECONDARY,
                            margin=MARGIN_SMALL,
                            showlegend=False,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📊 Potências Médias")
                        for wave, power in sorted(potencias.items(), key=lambda x: x[1], reverse=True):
                            st.metric(wave, f"{power:.2f} μV²")
                        
                        # Banda dominante
                        banda_dominante = max(potencias, key=potencias.get)
                        st.markdown("---")
                        st.success(f"**Banda Dominante:** {banda_dominante}")
                
            st.markdown("---")
            
            # Ondas suavizadas individuais
            if show_smoothed_waves:
                st.subheader(f"🎵 Padrões de Ondas Suavizadas (Janela: {smoothing_window}s)")
                
                brainwaves = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 
                              'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma']
                colors = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', 
                          '#4169E1', '#9370DB', '#FF69B4', '#8B4513']
                
                fig_smooth = go.Figure()
                
                for i, wave in enumerate(brainwaves):
                    if wave in df_dia.columns:
                        # Calcular média móvel
                        if 'Datetime' in df_dia.columns and pd.api.types.is_datetime64_any_dtype(df_dia['Datetime']):
                            # Reset index para evitar duplicatas
                            df_temp = df_dia.reset_index(drop=True).copy()
                            df_temp_indexed = df_temp.set_index('Datetime')
                            ma_data = df_temp_indexed[wave].rolling(f"{smoothing_window}s", min_periods=1, center=True).mean()
                            ma_values = ma_data.values
                        else:
                            ma_values = df_dia[wave].rolling(smoothing_window, min_periods=1, center=True).mean().values
                        
                        # Downsample para plotagem
                        plot_indices = np.linspace(0, len(df_dia)-1, min(len(df_dia), MAX_PLOT_POINTS), dtype=int)
                        
                        fig_smooth.add_trace(go.Scatter(
                            x=df_dia['sec_day'].iloc[plot_indices],
                            y=ma_values[plot_indices],
                            name=wave,
                            line=dict(color=colors[i], width=2),
                            mode='lines'
                        ))
                
                fig_smooth.update_layout(
                    height=GRAPH_HEIGHT_SECONDARY,
                    title=dict(text=f"Ondas Cerebrais Suavizadas ({smoothing_window}s)", font=dict(size=18)),
                    xaxis=dict(
                        title=dict(text='Horário', font=dict(size=16)),
                        tickfont=dict(size=14),
                        tickmode='array',
                        tickvals=np.linspace(df_dia['sec_day'].min(), df_dia['sec_day'].max(), 10),
                        ticktext=[segundos_para_hms(int(s)) for s in np.linspace(df_dia['sec_day'].min(), df_dia['sec_day'].max(), 10)],
                        gridcolor='rgba(200,200,200,0.3)'
                    ),
                    yaxis=dict(
                        title=dict(text='Amplitude (μV)', font=dict(size=16)),
                        tickfont=dict(size=14),
                        gridcolor='rgba(200,200,200,0.3)'
                    ),
                    margin=MARGIN_LARGE,
                    legend=dict(
                        orientation='v',
                        yanchor='top',
                        y=0.98,
                        xanchor='right',
                        x=0.98,
                        font=dict(size=12),
                        bgcolor='rgba(255,255,255,0.8)'
                    ),
                    template='plotly_white',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_smooth, use_container_width=True)
    else:
        st.info("ℹ️ Nenhuma data detectada no CSV.")

# ========== TAB 3: ESTADOS CEREBRAIS ==========
with tab3:
    st.header("🧠 Análise de Estados Cerebrais")
    
    if 'Brain_State' in df_filtrado.columns and show_brain_states:
        df_dia = df_filtrado[df_filtrado['Datetime'].dt.date == pd.to_datetime(dia_escolhido).date()].copy()
        
        if not df_dia.empty:
            # Contagem de estados
            state_counts = df_dia['Brain_State'].value_counts()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Gráfico de pizza
                fig_pie = px.pie(
                    values=state_counts.values,
                    names=state_counts.index,
                    title="Distribuição de Estados Cerebrais",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_layout(
                    height=GRAPH_HEIGHT_SECONDARY,
                    margin=MARGIN_SMALL,
                    font=dict(size=14)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Estatísticas de Estados")
                total_samples = len(df_dia)
                for state, count in state_counts.items():
                    percentage = (count / total_samples) * 100
                    st.metric(state, f"{count} amostras", delta=f"{percentage:.1f}%")
            
            st.markdown("---")
            
            # Timeline de estados
            st.subheader("⏱️ Linha do Tempo dos Estados Cerebrais")
            
            df_dia['sec_day'] = segs_desde_meianoite(df_dia['Datetime'])
            plot_df_states = get_plot_df(df_dia)
            
            # Mapear estados para números
            state_map = {state: i for i, state in enumerate(df_dia['Brain_State'].unique())}
            plot_df_states['State_Num'] = plot_df_states['Brain_State'].map(state_map)
            
            fig_timeline = go.Figure()
            
            for state, num in state_map.items():
                mask = plot_df_states['Brain_State'] == state
                fig_timeline.add_trace(go.Scatter(
                    x=plot_df_states.loc[mask, 'sec_day'],
                    y=plot_df_states.loc[mask, 'State_Num'],
                    mode='markers',
                    name=state,
                    marker=dict(size=8)
                ))
            
            fig_timeline.update_layout(
                height=GRAPH_HEIGHT_SMALL,
                title=dict(text="Estados Cerebrais ao Longo do Dia", font=dict(size=18)),
                xaxis=dict(
                    title=dict(text='Horário', font=dict(size=16)),
                    tickfont=dict(size=14),
                    tickmode='array',
                    tickvals=np.linspace(plot_df_states['sec_day'].min(), plot_df_states['sec_day'].max(), 10),
                    ticktext=[segundos_para_hms(int(s)) for s in np.linspace(plot_df_states['sec_day'].min(), plot_df_states['sec_day'].max(), 10)]
                ),
                yaxis=dict(
                    title=dict(text='Estado Cerebral', font=dict(size=16)),
                    tickfont=dict(size=14),
                    tickmode='array',
                    tickvals=list(state_map.values()),
                    ticktext=list(state_map.keys())
                ),
                margin=MARGIN_LARGE,
                template='plotly_white',
                hovermode='closest'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            st.markdown("---")
            
            # Razões cerebrais
            st.subheader("🧮 Razões de Frequência Cerebral")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'AlphaTheta_Ratio' in df_dia.columns:
                    avg_ratio = df_dia['AlphaTheta_Ratio'].mean()
                    st.metric("Razão α/θ", f"{avg_ratio:.2f}")
                    st.caption("Relaxamento consciente")
            
            with col2:
                if 'BetaAlpha_Ratio' in df_dia.columns:
                    avg_ratio = df_dia['BetaAlpha_Ratio'].mean()
                    st.metric("Razão β/α", f"{avg_ratio:.2f}")
                    st.caption("Atividade vs. relaxamento")
            
            with col3:
                if 'GammaBeta_Ratio' in df_dia.columns:
                    avg_ratio = df_dia['GammaBeta_Ratio'].mean()
                    st.metric("Razão γ/β", f"{avg_ratio:.2f}")
                    st.caption("Processamento cognitivo")
    else:
        st.info("ℹ️ Análise de estados cerebrais não disponível.")

# ========== TAB 4: INSIGHTS ESTATÍSTICOS ==========
with tab4:
    st.header("📈 Insights Estatísticos Avançados")
    
    metricas = calcular_metricas_avancadas(df_filtrado)
    
    # Métricas principais
    st.subheader("📌 Métricas Estatísticas Detalhadas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📈 Atenção")
        st.metric("Média", f"{metricas.get('Att_media', 0):.1f}")
        st.metric("Mediana", f"{metricas.get('Att_mediana', 0):.1f}")
        st.metric("Desvio Padrão", f"{metricas.get('Att_desvio', 0):.1f}")
        st.metric("Máximo", f"{metricas.get('Att_max', 0):.0f}")
        st.metric("Mínimo", f"{metricas.get('Att_min', 0):.0f}")
    
    with col2:
        st.markdown("### 🧘 Meditação")
        st.metric("Média", f"{metricas.get('Med_media', 0):.1f}")
        st.metric("Mediana", f"{metricas.get('Med_mediana', 0):.1f}")
        st.metric("Desvio Padrão", f"{metricas.get('Med_desvio', 0):.1f}")
        st.metric("Máximo", f"{metricas.get('Med_max', 0):.0f}")
        st.metric("Mínimo", f"{metricas.get('Med_min', 0):.0f}")
    
    with col3:
        st.markdown("### 🧠 Índices Cerebrais")
        st.metric("Razão α/θ", f"{metricas.get('Razao_Alpha_Theta', 0):.2f}")
        st.metric("Razão β/α", f"{metricas.get('Razao_Beta_Alpha', 0):.2f}")
        st.metric("Razão γ/β", f"{metricas.get('Razao_Gamma_Beta', 0):.2f}")
        
        st.markdown("---")
        st.markdown("**Interpretação:**")
        st.caption("• α/θ alto: Maior relaxamento consciente")
        st.caption("• β/α alto: Estado mais ativo vs. relaxado")
        st.caption("• γ/β alto: Maior processamento cognitivo")
    
    st.markdown("---")
    
    # Matriz de correlação
    st.subheader("🔗 Matriz de Correlação")
    
    cols_corr = ['Att', 'Med', 'Delta', 'Theta', 'LowAlpha', 'HighAlpha', 
                 'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma']
    cols_disponiveis = [c for c in cols_corr if c in df_filtrado.columns]
    
    if len(cols_disponiveis) > 1:
        corr_matrix = df_filtrado[cols_disponiveis].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlação")
        ))
        
        fig_corr.update_layout(
            title=dict(text="Correlação entre Variáveis EEG", font=dict(size=18)),
            height=GRAPH_HEIGHT_SECONDARY,
            margin=MARGIN_LARGE,
            template='plotly_white',
            xaxis=dict(tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=11))
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💡 Principais Correlações")
        
        # Encontrar correlações mais fortes
        corr_flat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_pairs = corr_flat.stack().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Correlações Positivas Fortes:**")
            count = 0
            for (var1, var2), corr_val in corr_pairs.items():
                if corr_val > 0.3 and count < 5:
                    st.write(f"• {var1} ↔ {var2}: **{corr_val:.2f}**")
                    count += 1
        
        with col2:
            st.markdown("**🔵 Correlações Negativas Fortes:**")
            count = 0
            for (var1, var2), corr_val in corr_pairs.iloc[::-1].items():
                if corr_val < -0.3 and count < 5:
                    st.write(f"• {var1} ↔ {var2}: **{corr_val:.2f}**")
                    count += 1
    else:
        st.info("Dados insuficientes para matriz de correlação.")
    
    st.markdown("---")
    
    # Padrões temporais
    st.subheader("⏰ Padrões Temporais por Hora")
    
    if 'Datetime' in df_filtrado.columns and not df_filtrado['Datetime'].isna().all():
        df_filtrado['hour'] = df_filtrado['Datetime'].dt.hour
        
        hourly_stats = df_filtrado.groupby('hour').agg({
            'Att': ['mean', 'std'],
            'Med': ['mean', 'std']
        }).reset_index()
        
        hourly_stats.columns = ['hour', 'Att_mean', 'Att_std', 'Med_mean', 'Med_std']
        
        fig_hourly = go.Figure()
        
        fig_hourly.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['Att_mean'],
            mode='lines+markers',
            name='Atenção Média',
            line=dict(color='#0056d6', width=3),
            error_y=dict(type='data', array=hourly_stats['Att_std'], visible=True)
        ))
        
        fig_hourly.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['Med_mean'],
            mode='lines+markers',
            name='Meditação Média',
            line=dict(color='#0f8a24', width=3),
            error_y=dict(type='data', array=hourly_stats['Med_std'], visible=True)
        ))
        
        fig_hourly.update_layout(
            title=dict(text="Padrões por Hora do Dia", font=dict(size=18)),
            xaxis=dict(title=dict(text='Hora do Dia', font=dict(size=16)), tickfont=dict(size=14)),
            yaxis=dict(title=dict(text='Nível Médio', font=dict(size=16)), tickfont=dict(size=14)),
            height=GRAPH_HEIGHT_SECONDARY,
            margin=MARGIN_LARGE,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(font=dict(size=13))
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Insights sobre horários
        st.markdown("### 🕐 Insights de Horários")
        
        best_att_hour = hourly_stats.loc[hourly_stats['Att_mean'].idxmax()]
        best_med_hour = hourly_stats.loc[hourly_stats['Med_mean'].idxmax()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"🎯 **Melhor hora para Atenção:** {int(best_att_hour['hour'])}:00 (média {best_att_hour['Att_mean']:.1f})")
        
        with col2:
            st.success(f"🧘 **Melhor hora para Meditação:** {int(best_med_hour['hour'])}:00 (média {best_med_hour['Med_mean']:.1f})")

# ========== TAB 5: ANÁLISE SEMANAL ==========
with tab5:
    st.header("📅 Análise Semanal & Tendências")
    
    if agg.empty:
        st.info("❌ Dados insuficientes para análise semanal.")
    else:
        # Tabela resumida
        st.subheader("📋 Resumo Diário")
        display_cols = ['date', 'Att_mean', 'Att_std', 'Med_mean', 'Med_std', 'samples']
        st.dataframe(
            agg[display_cols].style.format({
                'Att_mean': '{:.2f}', 'Att_std': '{:.2f}', 
                'Med_mean': '{:.2f}', 'Med_std': '{:.2f}',
                'samples': '{:.0f}'
            }),
            use_container_width=True,
            height=300
        )

        st.markdown("---")

        # Gráficos lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            fig_mean = go.Figure()
            fig_mean.add_trace(go.Bar(
                x=agg['date'].astype(str), 
                y=agg['Att_mean'], 
                name='Atenção (média)', 
                marker_color='#0056d6'
            ))
            fig_mean.add_trace(go.Bar(
                x=agg['date'].astype(str), 
                y=agg['Med_mean'], 
                name='Meditação (média)', 
                marker_color='#0f8a24'
            ))
            fig_mean.update_layout(
                title=dict(text="Média Diária", font=dict(size=18)),
                barmode='group',
                height=GRAPH_HEIGHT_SECONDARY,
                margin=MARGIN_LARGE,
                template='plotly_white',
                xaxis=dict(tickfont=dict(size=12)),
                yaxis=dict(title='Nível Médio', tickfont=dict(size=12)),
                legend=dict(font=dict(size=13))
            )
            st.plotly_chart(fig_mean, use_container_width=True)
        
        with col2:
            fig_range = go.Figure()
            fig_range.add_trace(go.Box(
                y=df_filtrado['Att'], 
                name='Atenção', 
                marker_color='#0056d6',
                boxmean='sd'
            ))
            fig_range.add_trace(go.Box(
                y=df_filtrado['Med'], 
                name='Meditação', 
                marker_color='#0f8a24',
                boxmean='sd'
            ))
            fig_range.update_layout(
                title=dict(text="Dispersão de Valores", font=dict(size=18)),
                height=GRAPH_HEIGHT_SECONDARY,
                margin=MARGIN_LARGE,
                template='plotly_white',
                yaxis=dict(title='Nível', tickfont=dict(size=12)),
                legend=dict(font=dict(size=13))
            )
            st.plotly_chart(fig_range, use_container_width=True)

        st.markdown("---")

        # Insights automáticos
        st.subheader("🎯 Insights Automáticos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Atenção")
            if 'Att_mean' in agg.columns:
                best_att_idx = agg['Att_mean'].idxmax()
                worst_att_idx = agg['Att_mean'].idxmin()
                best_att = agg.loc[best_att_idx]
                worst_att = agg.loc[worst_att_idx]
                
                st.success(f"✅ **Melhor dia:** {best_att['date']} (média {best_att['Att_mean']:.1f})")
                st.warning(f"⚠️ **Pior dia:** {worst_att['date']} (média {worst_att['Att_mean']:.1f})")
                st.info(f"📊 **Variabilidade:** ±{agg['Att_mean'].std():.2f}")
        
        with col2:
            st.markdown("### 🧘 Meditação")
            if 'Med_mean' in agg.columns:
                best_med_idx = agg['Med_mean'].idxmax()
                worst_med_idx = agg['Med_mean'].idxmin()
                best_med = agg.loc[best_med_idx]
                worst_med = agg.loc[worst_med_idx]
                
                st.success(f"✅ **Melhor dia:** {best_med['date']} (média {best_med['Med_mean']:.1f})")
                st.warning(f"⚠️ **Pior dia:** {worst_med['date']} (média {worst_med['Med_mean']:.1f})")
                st.info(f"📊 **Variabilidade:** ±{agg['Med_mean'].std():.2f}")

# ===== EXPORTAR PDF =====
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Exportar Relatório")

if st.sidebar.button("📄 Gerar Relatório PDF", type="primary", use_container_width=True):
    with st.spinner("📊 Gerando relatório PDF completo com gráficos..."):
        try:
            import io
            from reportlab.lib.pagesizes import A4, letter
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from io import BytesIO
            
            # Criar buffer para PDF
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20, leftMargin=20, topMargin=25, bottomMargin=25)
            
            # Estilos
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#0056d6'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#0056d6'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            story = []
            
            # Título
            story.append(Paragraph("🧠 Relatório de Análise EEG Avançada {uploaded_file.name}", title_style))
            story.append((Spacer(1, 0.1*inch))
)
            
            # Informações gerais
            metricas = calcular_metricas_avancadas(df_filtrado)
            
            story.append(Paragraph("📊 Resumo Estatístico", heading_style))
            
            data = [
                ['Métrica', 'Atenção', 'Meditação'],
                ['Média', f"{metricas.get('Att_media', 0):.1f}", f"{metricas.get('Med_media', 0):.1f}"],
                ['Mediana', f"{metricas.get('Att_mediana', 0):.1f}", f"{metricas.get('Med_mediana', 0):.1f}"],
                ['Desvio Padrão', f"{metricas.get('Att_desvio', 0):.1f}", f"{metricas.get('Med_desvio', 0):.1f}"],
                ['Máximo', f"{metricas.get('Att_max', 0):.0f}", f"{metricas.get('Med_max', 0):.0f}"],
                ['Mínimo', f"{metricas.get('Att_min', 0):.0f}", f"{metricas.get('Med_min', 0):.0f}"],
            ]
            
            t = Table(data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0056d6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t)
            story.append(Spacer(1, 0.1*inch))
            
            # Índices cerebrais
            story.append(Paragraph("🧠 Índices Cerebrais", heading_style))
            data2 = [
                ['Índice', 'Valor', 'Interpretação'],
                ['Razão α/θ', f"{metricas.get('Razao_Alpha_Theta', 0):.2f}", 'Relaxamento consciente'],
                ['Razão β/α', f"{metricas.get('Razao_Beta_Alpha', 0):.2f}", 'Estado ativo vs. relaxado'],
                ['Razão γ/β', f"{metricas.get('Razao_Gamma_Beta', 0):.2f}", 'Processamento cognitivo'],
            ]
            
            t2 = Table(data2, colWidths=[1.5*inch, 1.5*inch, 2.5*inch])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f8a24')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t2)
            story.append(Spacer(1, 0.1*inch))
            
            # Estados cerebrais
            if 'Brain_State' in df_filtrado.columns:
                story.append(Paragraph("🧠 Análise de Estados Cerebrais", heading_style))
                
                state_counts = df_filtrado['Brain_State'].value_counts()
                total = len(df_filtrado)
                
                state_data = [['Estado Cerebral', 'Amostras', 'Percentual']]
                for state, count in state_counts.items():
                    pct = (count / total) * 100
                    state_data.append([state, f"{count}", f"{pct:.1f}%"])
                
                t3 = Table(state_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
                t3.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9370DB')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                story.append(t3)
                story.append(Spacer(1, 0.1*inch))
            
            # Análise semanal
            if not agg.empty:
                story.append(Paragraph("📈 Análise Semanal", heading_style))
                
                agg_data = [['Data', 'Att Média', 'Med Média', 'Amostras']]
                for _, row in agg.head(10).iterrows():
                    agg_data.append([
                        str(row['date']),
                        f"{row['Att_mean']:.1f}",
                        f"{row['Med_mean']:.1f}",
                        f"{int(row['samples'])}"
                    ])
                
                t4 = Table(agg_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1*inch])
                t4.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0056d6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                ]))
                story.append(t4)
                
                # Insights
                story.append(PageBreak())
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph("🎯 Insights Principais", heading_style))
                
                if 'Att_mean' in agg.columns:
                    best_att = agg.loc[agg['Att_mean'].idxmax()]
                    worst_att = agg.loc[agg['Att_mean'].idxmin()]
                    
                    insights_text = f"""
                    <b>Atenção:</b><br/>
                    • Melhor dia: {best_att['date']} (média {best_att['Att_mean']:.1f})<br/>
                    • Pior dia: {worst_att['date']} (média {worst_att['Att_mean']:.1f})<br/>
                    • Variabilidade: ±{agg['Att_mean'].std():.2f}<br/>
                    <br/>
                    """
                    story.append(Paragraph(insights_text, styles['Normal']))
                
                if 'Med_mean' in agg.columns:
                    best_med = agg.loc[agg['Med_mean'].idxmax()]
                    worst_med = agg.loc[agg['Med_mean'].idxmin()]
                    
                    insights_text2 = f"""
                    <b>Meditação:</b><br/>
                    • Melhor dia: {best_med['date']} (média {best_med['Med_mean']:.1f})<br/>
                    • Pior dia: {worst_med['date']} (média {worst_med['Med_mean']:.1f})<br/>
                    • Variabilidade: ±{agg['Med_mean'].std():.2f}<br/>
                    """
                    story.append(Paragraph(insights_text2, styles['Normal']))
            
            # ===== GRÁFICOS COM MATPLOTLIB =====
            
            # Gráfico 1: Distribuições de Att e Med (SEM PageBreak - continua na mesma página)
            story.append(Paragraph("📊 Distribuições Estatísticas", heading_style))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
            
            if 'Att' in df_filtrado.columns:
                ax1.hist(df_filtrado['Att'].dropna(), bins=30, color='#0056d6', alpha=0.7, edgecolor='black')
                ax1.set_title('Distribuição de Atenção', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Nível de Atenção', fontsize=11)
                ax1.set_ylabel('Frequência', fontsize=11)
                ax1.grid(True, alpha=0.3)
                ax1.axvline(metricas.get('Att_media', 0), color='red', linestyle='--', linewidth=2, label=f'Média: {metricas.get("Att_media", 0):.1f}')
                ax1.legend()
            
            if 'Med' in df_filtrado.columns:
                ax2.hist(df_filtrado['Med'].dropna(), bins=30, color='#0f8a24', alpha=0.7, edgecolor='black')
                ax2.set_title('Distribuição de Meditação', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Nível de Meditação', fontsize=11)
                ax2.set_ylabel('Frequência', fontsize=11)
                ax2.grid(True, alpha=0.3)
                ax2.axvline(metricas.get('Med_media', 0), color='red', linestyle='--', linewidth=2, label=f'Média: {metricas.get("Med_media", 0):.1f}')
                ax2.legend()
            
            plt.tight_layout()
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            img = Image(img_buffer, width=6.5*inch, height=2.3*inch)
            story.append(img)
            story.append(Spacer(1, 0.15*inch))
            
            # Gráfico 2: Box plots comparativos (MESMA PÁGINA)
            story.append(Paragraph("📦 Análise de Dispersão", heading_style))
            
            fig, ax = plt.subplots(figsize=(8, 3.5))
            
            data_to_plot = []
            labels = []
            if 'Att' in df_filtrado.columns:
                data_to_plot.append(df_filtrado['Att'].dropna())
                labels.append('Atenção')
            if 'Med' in df_filtrado.columns:
                data_to_plot.append(df_filtrado['Med'].dropna())
                labels.append('Meditação')
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                               notch=True, showmeans=True)
                
                colors_box = ['#0056d6', '#0f8a24']
                for patch, color in zip(bp['boxes'], colors_box[:len(data_to_plot)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                
                ax.set_title('Dispersão de Valores (Box Plot)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Nível', fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            img_buffer2 = BytesIO()
            plt.savefig(img_buffer2, format='png', dpi=150, bbox_inches='tight')
            img_buffer2.seek(0)
            plt.close()
            
            img2 = Image(img_buffer2, width=5.5*inch, height=3.4*inch)
            story.append(img2)
            story.append(Spacer(1, 0.3*inch))
            
            # Gráfico 3: Análise Espectral
            potencias = calcular_potencia_media_bandas(df_filtrado)
            if potencias:
                story.append(PageBreak())
                story.append(Paragraph("📡 Análise Espectral", heading_style))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Bar chart
                waves = list(potencias.keys())
                powers = list(potencias.values())
                colors_spec = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', 
                              '#4169E1', '#9370DB', '#FF69B4', '#8B4513']
                
                ax1.bar(range(len(waves)), powers, color=colors_spec[:len(waves)], alpha=0.7, edgecolor='black')
                ax1.set_xticks(range(len(waves)))
                ax1.set_xticklabels(waves, rotation=45, ha='right')
                ax1.set_title('Potência por Banda de Frequência', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Potência Média (μV²)', fontsize=10)
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Pie chart
                ax2.pie(powers, labels=waves, autopct='%1.1f%%', colors=colors_spec[:len(waves)],
                       startangle=90, textprops={'fontsize': 8})
                ax2.set_title('Distribuição Percentual', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                img_buffer3 = BytesIO()
                plt.savefig(img_buffer3, format='png', dpi=150, bbox_inches='tight')
                img_buffer3.seek(0)
                plt.close()
                
                img3 = Image(img_buffer3, width=6.5*inch, height=2.6*inch)
                story.append(img3)
                story.append(Spacer(1, 0.1*inch))
                
                banda_dominante = max(potencias, key=potencias.get)
                story.append(Paragraph(f"<b>Banda Dominante:</b> {banda_dominante} ({potencias[banda_dominante]:.2f} μV²)", styles['Normal']))
            
            # Gráfico 4: Análise Espectral - Radar Chart
            potencias_pdf = calcular_potencia_media_bandas(df_filtrado)
            if potencias_pdf:
                story.append(Paragraph("📡 Análise Espectral - Distribuição de Potência", heading_style))
                
                # Criar radar chart com matplotlib
                waves = list(potencias_pdf.keys())
                powers = list(potencias_pdf.values())
                
                # Número de variáveis
                num_vars = len(waves)
                
                # Calcular ângulos para cada eixo
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                powers_plot = powers + [powers[0]]  # Fechar o polígono
                angles += angles[:1]  # Fechar o polígono
                
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                
                # Plot
                ax.plot(angles, powers_plot, 'o-', linewidth=2, color='#0056d6', label='Potência')
                ax.fill(angles, powers_plot, alpha=0.25, color='#0056d6')
                
                # Labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(waves, size=10)
                ax.set_ylim(0, max(powers) * 1.1)
                ax.set_title('Distribuição de Potência por Banda de Frequência', 
                           size=14, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                img_buffer4 = BytesIO()
                plt.savefig(img_buffer4, format='png', dpi=150, bbox_inches='tight')
                img_buffer4.seek(0)
                plt.close()
                
                img4 = Image(img_buffer4, width=6*inch, height=6*inch)
                story.append(img4)
                story.append(Spacer(1, 0.1*inch))
                
                # Banda dominante
                banda_dominante_pdf = max(potencias_pdf, key=potencias_pdf.get)
                insights_spectral = f"""
                <b>Análise Espectral:</b><br/>
                • Banda Dominante: {banda_dominante_pdf} ({potencias_pdf[banda_dominante_pdf]:.2f} μV²)<br/>
                • Total de Bandas Analisadas: {len(potencias_pdf)}<br/>
                """
                story.append(Paragraph(insights_spectral, styles['Normal']))
            
            # Gráfico 5: Estados Cerebrais (se disponível)
            if 'Brain_State' in df_filtrado.columns:
                story.append(Paragraph("🧠 Distribuição de Estados Cerebrais", heading_style))
                
                state_counts = df_filtrado['Brain_State'].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                
                colors_states = plt.cm.Set3(range(len(state_counts)))
                wedges, texts, autotexts = ax.pie(state_counts.values, labels=state_counts.index, 
                                                   autopct='%1.1f%%', colors=colors_states,
                                                   startangle=90, textprops={'fontsize': 9})
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.set_title('Distribuição de Estados Cerebrais', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                img_buffer5 = BytesIO()
                plt.savefig(img_buffer5, format='png', dpi=150, bbox_inches='tight')
                img_buffer5.seek(0)
                plt.close()
                
                img5 = Image(img_buffer5, width=5.5*inch, height=3.4*inch)
                story.append(img5)
            
            # ===== KEY INSIGHTS SUMMARY =====
            story.append(Paragraph("🔑 PRINCIPAIS INSIGHTS DOS DADOS EEG", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Calcular insights
            total_time = len(df_filtrado)
            if 'Datetime' in df_filtrado.columns and not df_filtrado['Datetime'].isna().all():
                time_diff = (df_filtrado['Datetime'].max() - df_filtrado['Datetime'].min()).total_seconds()
                minutes = int(time_diff // 60)
                seconds = int(time_diff % 60)
                time_str = f"{int(time_diff)} segundos ({minutes} minutos {seconds} segundos)"
            else:
                time_str = f"{total_time} amostras"
            
            # 1. Overall Performance
            insights_1 = f"""
            <b>1. Desempenho Geral:</b><br/>
            • Atenção Média: {metricas.get('Att_media', 0):.2f}<br/>
            • Meditação Média: {metricas.get('Med_media', 0):.2f}<br/>
            • Tempo Total de Gravação: {time_str}<br/>
            • Total de Amostras: {total_time}<br/>
            """
            story.append(Paragraph(insights_1, styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
            
            # 2. Peak Performance
            if 'Att' in df_filtrado.columns and 'Med' in df_filtrado.columns:
                max_att_idx = df_filtrado['Att'].idxmax()
                max_med_idx = df_filtrado['Med'].idxmax()
                
                if 'Datetime' in df_filtrado.columns and not df_filtrado['Datetime'].isna().all():
                    max_att_time = df_filtrado.loc[max_att_idx, 'Datetime'].strftime('%H:%M:%S')
                    max_med_time = df_filtrado.loc[max_med_idx, 'Datetime'].strftime('%H:%M:%S')
                else:
                    max_att_time = f"amostra {max_att_idx}"
                    max_med_time = f"amostra {max_med_idx}"
                
                insights_2 = f"""
                <b>2. Pico de Desempenho:</b><br/>
                • Atenção Máxima: {metricas.get('Att_max', 0):.0f} (alcançado em {max_att_time})<br/>
                • Meditação Máxima: {metricas.get('Med_max', 0):.0f} (alcançado em {max_med_time})<br/>
                """
                story.append(Paragraph(insights_2, styles['Normal']))
                story.append(Spacer(1, 0.15*inch))
            
            # 3. Dominant Frequency Band
            if potencias:
                banda_dominante = max(potencias, key=potencias.get)
                poder_dominante = potencias[banda_dominante]
                
                insights_3 = f"""
                <b>3. Banda de Frequência Dominante:</b><br/>
                • Banda: {banda_dominante}<br/>
                • Potência Média: {poder_dominante:.2f} μV²<br/>
                """
                story.append(Paragraph(insights_3, styles['Normal']))
                story.append(Spacer(1, 0.15*inch))
            
            # 4. Variability Analysis
            banda_mais_estavel = None
            if potencias:
                # Calcular desvio padrão de cada banda
                band_stds = {}
                for wave in potencias.keys():
                    if wave in df_filtrado.columns:
                        band_stds[wave] = df_filtrado[wave].std()
                if band_stds:
                    banda_mais_estavel = min(band_stds, key=band_stds.get)
                    poder_estavel = potencias.get(banda_mais_estavel, 0)
            
            insights_4 = f"""
            <b>4. Análise de Variabilidade:</b><br/>
            • Desvio Padrão da Atenção: {metricas.get('Att_desvio', 0):.2f}<br/>
            • Desvio Padrão da Meditação: {metricas.get('Med_desvio', 0):.2f}<br/>
            """
            if banda_mais_estavel:
                insights_4 += f"• Banda Mais Estável: {banda_mais_estavel} ({poder_estavel:.2f} potência média)<br/>"
            
            story.append(Paragraph(insights_4, styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
            
            # 5. Strongest Correlations
            cols_corr = ['Att', 'Med', 'Delta', 'Theta', 'LowAlpha', 'HighAlpha', 
                         'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma']
            cols_disponiveis = [c for c in cols_corr if c in df_filtrado.columns]
            
            if len(cols_disponiveis) > 1:
                corr_matrix = df_filtrado[cols_disponiveis].corr()
                corr_flat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                corr_pairs = corr_flat.stack().sort_values(ascending=False)
                
                insights_5 = "<b>5. Correlações Mais Fortes:</b><br/>"
                count = 0
                for (var1, var2), corr_val in corr_pairs.items():
                    if count < 3:
                        insights_5 += f"• {var1} & {var2}: {corr_val:.3f}<br/>"
                        count += 1
                
                story.append(Paragraph(insights_5, styles['Normal']))
                story.append(Spacer(1, 0.15*inch))
            
            # 6. Brain State Summary (if available)
            if 'Brain_State' in df_filtrado.columns:
                state_counts = df_filtrado['Brain_State'].value_counts()
                estado_dominante = state_counts.index[0]
                pct_dominante = (state_counts.iloc[0] / len(df_filtrado)) * 100
                
                insights_6 = f"""
                <b>6. Resumo de Estados Cerebrais:</b><br/>
                • Estado Dominante: {estado_dominante} ({pct_dominante:.1f}% do tempo)<br/>
                • Total de Estados Detectados: {len(state_counts)}<br/>
                """
                story.append(Paragraph(insights_6, styles['Normal']))
                story.append(Spacer(1, 0.15*inch))
            
            # 7. Temporal Patterns (if available)
            if 'Datetime' in df_filtrado.columns and not df_filtrado['Datetime'].isna().all():
                df_filtrado['hour'] = df_filtrado['Datetime'].dt.hour
                hourly_stats = df_filtrado.groupby('hour').agg({
                    'Att': 'mean',
                    'Med': 'mean'
                }).reset_index()
                
                best_att_hour = hourly_stats.loc[hourly_stats['Att'].idxmax()]
                best_med_hour = hourly_stats.loc[hourly_stats['Med'].idxmax()]
                
                insights_7 = f"""
                <b>7. Padrões Temporais:</b><br/>
                • Melhor Hora para Atenção: {int(best_att_hour['hour'])}:00 (média {best_att_hour['Att']:.1f})<br/>
                • Melhor Hora para Meditação: {int(best_med_hour['hour'])}:00 (média {best_med_hour['Med']:.1f})<br/>
                """
                story.append(Paragraph(insights_7, styles['Normal']))
            
            # Rodapé
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph("=" * 80, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            footer_text = f"<i>Relatório gerado em {pd.Timestamp.now().strftime('%d/%m/%Y às %H:%M')}</i>"
            story.append(Paragraph(footer_text, styles['Normal']))
            story.append(Paragraph("<i>Análise EEG v5.0 Enhanced - Relatório Completo com Visualizações e Insights</i>", styles['Normal']))
            
            # Construir PDF
            doc.build(story)
            buffer.seek(0)
            
            # Botão de download
            st.sidebar.download_button(
                label="⬇️ Baixar Relatório PDF",
                data=buffer,
                file_name=f"relatorio_eeg_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
            st.sidebar.success("✅ Relatório gerado com sucesso!")
            
        except ImportError:
            st.sidebar.error("❌ Biblioteca reportlab não instalada. Execute: pip install reportlab")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao gerar PDF: {str(e)}")

# ========== TAB 6: ANÁLISE GERAL (TODAS AS SESSÕES) ==========
with tab6:
    st.header("🌍 Análise Geral - Todas as Sessões")
    
    if len(st.session_state.uploaded_data) > 1:
        st.info(f"📊 Analisando {len(st.session_state.uploaded_data)} sessões combinadas")
        
        # Combinar todos os dataframes
        all_dfs = []
        for filename, df_session in st.session_state.uploaded_data.items():
            df_temp = df_session.copy()
            df_temp['session_file'] = filename
            all_dfs.append(df_temp)
        
        df_combined = pd.concat(all_dfs, ignore_index=True)
        
        # Ordenar por datetime se disponível (necessário para rolling window)
        if 'Datetime' in df_combined.columns and not df_combined['Datetime'].isna().all():
            df_combined = df_combined.sort_values('Datetime').reset_index(drop=True)
        
        # Processar dados combinados
        df_combined = agregar_bandas_e_ma(df_combined, smoothing_window)
        df_combined = calcular_razoes_cerebrais(df_combined)
        if 'Att' in df_combined.columns and 'Med' in df_combined.columns:
            df_combined['Brain_State'] = df_combined.apply(classificar_estado_cerebral, axis=1)
        
        # Métricas gerais
        st.subheader("📈 Métricas Gerais de Todas as Sessões")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Sessões", len(st.session_state.uploaded_data))
        with col2:
            st.metric("Total de Amostras", f"{len(df_combined):,}")
        with col3:
            if 'Att' in df_combined.columns:
                st.metric("Att Média Geral", f"{df_combined['Att'].mean():.1f}")
        with col4:
            if 'Med' in df_combined.columns:
                st.metric("Med Média Geral", f"{df_combined['Med'].mean():.1f}")
        
        st.markdown("---")
        
        # Análise por sessão (arquivo)
        st.subheader("📊 Comparação Entre Sessões")
        
        session_stats = []
        for filename in st.session_state.uploaded_data.keys():
            df_session = df_combined[df_combined['session_file'] == filename]
            
            if 'Datetime' in df_session.columns and not df_session['Datetime'].isna().all():
                session_date = df_session['Datetime'].dt.date.mode()[0] if len(df_session['Datetime'].dt.date.mode()) > 0 else 'N/A'
            else:
                session_date = 'N/A'
            
            stats = {
                'Sessão': filename,
                'Data': str(session_date),
                'Amostras': len(df_session),
                'Att Média': df_session['Att'].mean() if 'Att' in df_session.columns else 0,
                'Med Média': df_session['Med'].mean() if 'Med' in df_session.columns else 0,
                'Att Máx': df_session['Att'].max() if 'Att' in df_session.columns else 0,
                'Med Máx': df_session['Med'].max() if 'Med' in df_session.columns else 0
            }
            session_stats.append(stats)
        
        df_sessions = pd.DataFrame(session_stats)
        
        # Tabela de comparação
        st.dataframe(
            df_sessions.style.format({
                'Amostras': '{:,.0f}',
                'Att Média': '{:.1f}',
                'Med Média': '{:.1f}',
                'Att Máx': '{:.0f}',
                'Med Máx': '{:.0f}'
            }).background_gradient(subset=['Att Média', 'Med Média'], cmap='Blues'),
            use_container_width=True,
            height=300
        )
        
        st.markdown("---")
        
        # Gráficos comparativos
        st.subheader("📈 Evolução ao Longo das Sessões")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras - Att por sessão
            fig_att = go.Figure()
            fig_att.add_trace(go.Bar(
                x=df_sessions['Sessão'],
                y=df_sessions['Att Média'],
                name='Atenção Média',
                marker_color='#0056d6',
                text=df_sessions['Att Média'].round(1),
                textposition='auto'
            ))
            fig_att.update_layout(
                title=dict(text="Atenção Média por Sessão", font=dict(size=16)),
                xaxis_title="Sessão",
                yaxis_title="Nível Médio",
                height=GRAPH_HEIGHT_SMALL,
                margin=MARGIN_SMALL,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig_att, use_container_width=True)
        
        with col2:
            # Gráfico de barras - Med por sessão
            fig_med = go.Figure()
            fig_med.add_trace(go.Bar(
                x=df_sessions['Sessão'],
                y=df_sessions['Med Média'],
                name='Meditação Média',
                marker_color='#0f8a24',
                text=df_sessions['Med Média'].round(1),
                textposition='auto'
            ))
            fig_med.update_layout(
                title=dict(text="Meditação Média por Sessão", font=dict(size=16)),
                xaxis_title="Sessão",
                yaxis_title="Nível Médio",
                height=GRAPH_HEIGHT_SMALL,
                margin=MARGIN_SMALL,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig_med, use_container_width=True)
        
        st.markdown("---")
        
        # Análise por dia (agregando todas as sessões)
        st.subheader("📅 Análise Agregada por Dia")
        
        if 'Datetime' in df_combined.columns and not df_combined['Datetime'].isna().all():
            df_combined['date'] = df_combined['Datetime'].dt.date
            
            daily_agg = df_combined.groupby('date').agg({
                'Att': ['mean', 'std', 'max', 'min', 'count'],
                'Med': ['mean', 'std', 'max', 'min']
            }).reset_index()
            
            daily_agg.columns = ['Data', 'Att_mean', 'Att_std', 'Att_max', 'Att_min', 'samples', 
                                'Med_mean', 'Med_std', 'Med_max', 'Med_min']
            
            # Gráfico de linha - Evolução temporal
            fig_evolution = go.Figure()
            
            fig_evolution.add_trace(go.Scatter(
                x=daily_agg['Data'],
                y=daily_agg['Att_mean'],
                mode='lines+markers',
                name='Atenção Média',
                line=dict(color='#0056d6', width=3),
                marker=dict(size=8),
                error_y=dict(type='data', array=daily_agg['Att_std'], visible=True)
            ))
            
            fig_evolution.add_trace(go.Scatter(
                x=daily_agg['Data'],
                y=daily_agg['Med_mean'],
                mode='lines+markers',
                name='Meditação Média',
                line=dict(color='#0f8a24', width=3),
                marker=dict(size=8),
                error_y=dict(type='data', array=daily_agg['Med_std'], visible=True)
            ))
            
            fig_evolution.update_layout(
                title=dict(text="Evolução Temporal - Todas as Sessões", font=dict(size=18)),
                xaxis_title="Data",
                yaxis_title="Nível Médio",
                height=GRAPH_HEIGHT_SECONDARY,
                margin=MARGIN_LARGE,
                template='plotly_white',
                hovermode='x unified',
                legend=dict(font=dict(size=13))
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            st.markdown("---")
            
            # Tabela de resumo diário
            st.subheader("📋 Resumo Diário Consolidado")
            st.dataframe(
                daily_agg.style.format({
                    'Att_mean': '{:.1f}',
                    'Att_std': '{:.1f}',
                    'Att_max': '{:.0f}',
                    'Att_min': '{:.0f}',
                    'samples': '{:.0f}',
                    'Med_mean': '{:.1f}',
                    'Med_std': '{:.1f}',
                    'Med_max': '{:.0f}',
                    'Med_min': '{:.0f}'
                }),
                use_container_width=True,
                height=300
            )
        
        st.markdown("---")
        
        # Análise de estados cerebrais consolidada
        if 'Brain_State' in df_combined.columns:
            st.subheader("🧠 Estados Cerebrais - Visão Geral")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                state_counts_all = df_combined['Brain_State'].value_counts()
                
                fig_states = px.pie(
                    values=state_counts_all.values,
                    names=state_counts_all.index,
                    title="Distribuição de Estados - Todas as Sessões",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_states.update_layout(
                    height=GRAPH_HEIGHT_SECONDARY,
                    margin=MARGIN_SMALL,
                    font=dict(size=14)
                )
                st.plotly_chart(fig_states, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Estatísticas Gerais")
                total_all = len(df_combined)
                for state, count in state_counts_all.items():
                    pct = (count / total_all) * 100
                    st.metric(state, f"{count:,}", delta=f"{pct:.1f}%")
        
        st.markdown("---")
        
        # Insights finais
        st.subheader("💡 Insights Consolidados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📈 Progresso")
            if len(df_sessions) >= 2:
                first_att = df_sessions.iloc[0]['Att Média']
                last_att = df_sessions.iloc[-1]['Att Média']
                diff_att = last_att - first_att
                
                if diff_att > 0:
                    st.success(f"✅ Atenção melhorou {diff_att:.1f} pontos")
                elif diff_att < 0:
                    st.warning(f"⚠️ Atenção reduziu {abs(diff_att):.1f} pontos")
                else:
                    st.info("➡️ Atenção manteve-se estável")
        
        with col2:
            st.markdown("### 🧘 Meditação")
            if len(df_sessions) >= 2:
                first_med = df_sessions.iloc[0]['Med Média']
                last_med = df_sessions.iloc[-1]['Med Média']
                diff_med = last_med - first_med
                
                if diff_med > 0:
                    st.success(f"✅ Meditação melhorou {diff_med:.1f} pontos")
                elif diff_med < 0:
                    st.warning(f"⚠️ Meditação reduziu {abs(diff_med):.1f} pontos")
                else:
                    st.info("➡️ Meditação manteve-se estável")
        
        with col3:
            st.markdown("### 🎯 Melhor Sessão")
            best_session = df_sessions.loc[df_sessions['Att Média'].idxmax()]
            st.success(f"**{best_session['Sessão']}**")
            st.caption(f"Att: {best_session['Att Média']:.1f}")
            st.caption(f"Med: {best_session['Med Média']:.1f}")
    
    else:
        st.info("📊 Carregue mais de um arquivo CSV para ver a análise consolidada de todas as sessões.")
        st.markdown("---")
        st.markdown("""
        ### 💡 Dica
        
        A análise geral permite:
        - Comparar múltiplas sessões de gravação
        - Ver evolução ao longo do tempo
        - Identificar padrões entre diferentes dias
        - Acompanhar progresso geral
        
        **Para usar esta funcionalidade:**
        1. Carregue múltiplos arquivos CSV usando o uploader
        2. Os dados serão automaticamente combinados e analisados
        3. Você poderá ver comparações e tendências entre todas as sessões
        """)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Sobre a Análise")
st.sidebar.info("""
**Análises Integradas:**
- Estados cerebrais classificados
- Análise espectral com radar chart
- Ondas suavizadas interativas
- Insights estatísticos avançados
- Padrões temporais por hora
- Correlações entre variáveis
- **Análise consolidada multi-sessão**

**Baseado em:**
- Jupyter Notebook EEG_Data_Processing
- Métricas científicas de EEG
- Visualizações interativas Plotly
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*v5.0 Enhanced - Análise Completa + Multi-Sessão*")

