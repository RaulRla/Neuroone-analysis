# 🧠 Analisador EEG Avançado - Streamlit App

Aplicação web para análise completa de dados EEG com classificação de estados cerebrais, análise espectral e insights estatísticos.

## 📋 Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

## 🚀 Instalação Local

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Executar a Aplicação

```bash
streamlit run app_eeg_v5_enhanced.py
```

A aplicação será aberta automaticamente no seu navegador em `http://localhost:8501`

## ☁️ Deploy no Streamlit Cloud

### Passo 1: Preparar o Repositório

1. Crie um repositório no GitHub
2. Faça upload dos seguintes arquivos:
   - `app_eeg_v5_enhanced.py`
   - `processador_eeg_minimal.py`
   - `requirements.txt`
   - `.streamlit/config.toml`

### Passo 2: Deploy

1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Faça login com sua conta GitHub
3. Clique em "New app"
4. Selecione:
   - **Repository**: seu repositório
   - **Branch**: main (ou master)
   - **Main file path**: `app_eeg_v5_enhanced.py`
5. Clique em "Deploy!"

### Passo 3: Configurações Avançadas (Opcional)

Se necessário, você pode ajustar as configurações em "Advanced settings":
- **Python version**: 3.9 ou superior
- **Secrets**: não necessário para esta aplicação

## 📁 Estrutura de Arquivos

```
streamlit/
├── app_eeg_v5_enhanced.py          # Aplicação principal
├── processador_eeg_minimal.py      # Processador de dados EEG
├── requirements.txt                # Dependências Python
├── .streamlit/
│   └── config.toml                 # Configurações do Streamlit
└── README.md                       # Este arquivo
```

## 🎯 Funcionalidades

### 📊 Visão Geral
- Séries temporais de Atenção e Meditação
- Métricas estatísticas em tempo real
- Distribuições e histogramas

### 🌊 Análise de Ondas Cerebrais
- Análise espectral com radar chart
- Ondas suavizadas (Delta, Theta, Alpha, Beta, Gamma)
- Potência média por banda de frequência

### 🧠 Estados Cerebrais
- Classificação automática de estados mentais
- Timeline de estados ao longo do dia
- Razões de frequência cerebral (α/θ, β/α, γ/β)

### 📈 Insights Estatísticos
- Métricas detalhadas (média, mediana, desvio padrão)
- Matriz de correlação entre variáveis
- Padrões temporais por hora do dia

### 📅 Análise Semanal
- Resumo diário consolidado
- Gráficos de evolução temporal
- Insights automáticos de progresso

### 🌍 Análise Geral (Multi-Sessão)
- Comparação entre múltiplas sessões
- Evolução ao longo do tempo
- Métricas consolidadas

### 📄 Exportação de Relatórios
- Geração de PDF completo com gráficos
- Insights consolidados
- Visualizações estatísticas

## 📊 Formato dos Dados

A aplicação espera arquivos CSV com as seguintes colunas:

**Obrigatórias:**
- `Date`, `Time` (ou `Datetime`)
- `Delta`, `Theta`
- `LowAlpha`, `HighAlpha`
- `LowBeta`, `HighBeta`
- `LowGamma`, `MiddleGamma`
- `Att` (Atenção)
- `Med` (Meditação)

## 🔧 Configurações

### Ajustar Porta (Local)

Edite `.streamlit/config.toml`:

```toml
[server]
port = 8502  # Altere para a porta desejada
```

### Personalizar Tema

Edite `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#0056d6"      # Cor principal
backgroundColor = "#FFFFFF"    # Cor de fundo
secondaryBackgroundColor = "#F0F2F6"  # Cor de fundo secundária
textColor = "#262730"         # Cor do texto
```

## 🐛 Solução de Problemas

### Erro: "ModuleNotFoundError"

```bash
pip install -r requirements.txt --upgrade
```

### Erro: "Port already in use"

```bash
streamlit run app_eeg_v5_enhanced.py --server.port 8502
```

### Erro ao gerar PDF

Certifique-se de que o reportlab está instalado:

```bash
pip install reportlab PyPDF2
```

## 📝 Notas de Versão

### v5.0 Enhanced
- ✅ Análise multi-sessão consolidada
- ✅ Exportação de relatórios PDF com gráficos
- ✅ Classificação de estados cerebrais
- ✅ Análise espectral avançada
- ✅ Insights estatísticos automáticos
- ✅ Padrões temporais por hora
- ✅ Matriz de correlação interativa

## 🤝 Suporte

Para problemas ou sugestões:
1. Verifique a documentação do Streamlit: [docs.streamlit.io](https://docs.streamlit.io)
2. Revise os logs de erro no terminal
3. Certifique-se de que todas as dependências estão instaladas

## 📄 Licença

Este projeto é fornecido como está, para fins educacionais e de pesquisa.

---

**Desenvolvido com ❤️ usando Streamlit**
