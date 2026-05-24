# Copyright (C) 2026 Bruno Proença de Souza
# Licenciado sob GNU AGPL v3 - veja o arquivo LICENSE

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import re
from scipy import stats

# ==========================================
# FUNÇÃO AUXILIAR DE FORMATAÇÃO PT-BR
# ==========================================
def formatar_numero(valor, prefixo='', sufixo='', decimais=0):
    """Formata números para o padrão brasileiro (1.000,00)."""
    if pd.isna(valor):
        return "-"
    if decimais > 0:
        s = f"{valor:,.{decimais}f}"
    else:
        s = f"{valor:,.0f}"
    s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"{prefixo}{s}{sufixo}".strip()

# Configuração da página
st.set_page_config(
    page_title="Dashboard Soja Paraná",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
.main {
    padding: 0rem 1rem;
}

h1 {
    color: #2c5f2d;
    text-align: center;
    padding: 20px;
    margin-bottom: 10px;
}

/* GRADIENTE AJUSTADO */
.stMetric {
    background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #0f172a 100%);
    padding: 15px;
    border-radius: 12px;

    /* profundidade */
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 6px 18px rgba(0,0,0,0.5);
}

/* TEXTO PRINCIPAL */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 600;
}

/* LABEL */
[data-testid="stMetricLabel"] {
    color: #d1d5db !important;
}

/* CORES DE STATUS (FUNCIONAM DE VERDADE AQUI) */
.positive {
    color: #22c55e !important;
}

.negative {
    color: #ef4444 !important;
}

.neutral {
    color: #ffffff !important;
}
</style>
    """, unsafe_allow_html=True)

st.markdown("<h1>🌱 Dashboard - Soja no Paraná (2018-2024)</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #000000;'>Análise Inteligente: Clima + Produtividade + Geolocalização</h3>", unsafe_allow_html=True)

# ==========================================
# CARREGAMENTO – dataset unificado único
# ==========================================
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_parquet('./data/processed/dataset_final.parquet')
        df['cod_ibge'] = df['cod_ibge'].astype(str)

        # Renomeia colunas PAM para manter compatibilidade com o restante do app
        df = df.rename(columns={
            'Área plantada ou destinada à colheita (Hectares)':
                'Área plantada (Hectares)',
            'Área plantada ou destinada à colheita - percentual do total geral':
                'Área plantada - percentual do total geral',
            'Ano': 'ano',
            'mesorregiao': 'Mesorregião'
        })

        # Colunas derivadas
        df['Área perdida (Hectares)'] = (
            df['Área plantada (Hectares)'] - df['Área colhida (Hectares)']
        )
        df['Percentual de perda (%)'] = (
            df['Área perdida (Hectares)'] / df['Área plantada (Hectares)']
        ) * 100

        # Unidades: Mil Reais → Reais; Toneladas × 1000 → kg
        df['Quantidade produzida (Toneladas)'] = df['Quantidade produzida (Toneladas)'] * 1000
        df['Valor da produção (Mil Reais)']    = df['Valor da produção (Mil Reais)']    * 1000

        # Chave numérica para o mapa
        df['codigo_ibge'] = df['cod_ibge'].astype(str).str.zfill(7).str[:7].astype(int)

        # Garante que a coluna Mesorregião exista
        if 'Mesorregião' not in df.columns:
             st.warning("Aviso: Coluna 'mesorregiao' não encontrada no dataset. O filtro de mesorregião não terá efeito.")
             df['Mesorregião'] = "Mesorregião Desconhecida"

        return df

    except FileNotFoundError:
        st.error("⚠️ Arquivo 'dataset_final.parquet' não encontrado!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        st.stop()

df = carregar_dados()

# Identificar colunas climáticas
colunas_climaticas = [col for col in df.columns if re.match(r'.*_dec\d+_ano\d+', col)]
atributos_climaticos = list(set([col.rsplit('_dec', 1)[0] for col in colunas_climaticas]))

# ==========================================
# FUNÇÃO DE CORRELAÇÃO – com cache_key para
# invalidar o cache quando os filtros mudam
# ==========================================
@st.cache_data
def calcular_correlacoes_por_ano(_df, metrica, ano_filtro, cache_key):
    """
    Calcula correlações de Pearson entre variáveis climáticas e uma métrica de soja.

    O parâmetro cache_key é uma string derivada dos filtros ativos (municípios,
    mesorregiões, anos). Como o Streamlit NÃO hasheia argumentos com underscore (_df),
    a cache_key garante que entradas distintas sejam criadas no cache sempre que
    os filtros mudarem, evitando resultados desatualizados.
    """
    resultados = []
    for col_clima in colunas_climaticas:
        try:
            df_temp = _df[[col_clima, metrica]].dropna()
            if len(df_temp) > 5:
                corr = df_temp.corr().iloc[0, 1]
                if not np.isnan(corr):
                    atributo = col_clima.rsplit('_dec', 1)[0]
                    dec_match = re.search(r'dec(\d+)', col_clima)
                    ano_match = re.search(r'ano(\d+)', col_clima)
                    if dec_match and ano_match:
                        resultados.append({
                            'Variável Climática': atributo,
                            'Decêndio': int(dec_match.group(1)),
                            'Ano Safra': f"ano{ano_match.group(1)}",
                            'Coluna': col_clima,
                            'Correlação': corr,
                            'Correlação Abs': abs(corr)
                        })
        except:
            continue
    return pd.DataFrame(resultados)

# ==========================================
# SIDEBAR – Filtros
# ==========================================
st.sidebar.header("🔍 Filtros de Análise")

anos_disponiveis = sorted(df['ano'].unique())
anos_selecionados = st.sidebar.multiselect("Selecione os anos:", options=anos_disponiveis, default=anos_disponiveis)

# Filtro de Mesorregião
mesorregioes_disponiveis = sorted(df['Mesorregião'].dropna().unique())
todas_mesorregioes = st.sidebar.checkbox("Selecionar todas as Mesorregiões", value=True)

if todas_mesorregioes:
    mesorregioes_selecionadas = mesorregioes_disponiveis
else:
    mesorregioes_selecionadas = st.sidebar.multiselect(
        "Selecione as Mesorregiões:",
        options=mesorregioes_disponiveis,
        default=mesorregioes_disponiveis
    )

# Filtrar municípios baseado nas mesorregiões selecionadas
df_filtrado_meso = df[df['Mesorregião'].isin(mesorregioes_selecionadas)]
municipios_disponiveis_filtrados = sorted(df_filtrado_meso['Município'].unique())

visualizar_todos = st.sidebar.radio("Municípios:", options=["Todos os municípios da(s) mesorregião(ões)", "Selecionar específicos"], index=0)

if visualizar_todos == "Todos os municípios da(s) mesorregião(ões)":
    municipios_selecionados = municipios_disponiveis_filtrados
else:
    municipios_selecionados = st.sidebar.multiselect(
        "Escolha os municípios:",
        options=municipios_disponiveis_filtrados,
        default=municipios_disponiveis_filtrados[:5] if len(municipios_disponiveis_filtrados) >= 5 else municipios_disponiveis_filtrados
    )

df_filtrado = df[
    (df['ano'].isin(anos_selecionados)) &
    (df['Município'].isin(municipios_selecionados))
].copy()

st.sidebar.markdown("---")
st.sidebar.header("📊 Informações")
st.sidebar.metric("Mesorregiões", len(mesorregioes_selecionadas))
st.sidebar.metric("Municípios", len(municipios_selecionados))
st.sidebar.metric("Anos", len(anos_selecionados))
st.sidebar.metric("Registros", formatar_numero(len(df_filtrado)))
st.sidebar.metric("Variáveis Climáticas", len(colunas_climaticas))

df_agregado = df_filtrado.groupby('ano').agg({
    'Área plantada (Hectares)': 'sum',
    'Área colhida (Hectares)': 'sum',
    'Área perdida (Hectares)': 'sum',
    'Percentual de perda (%)': 'mean',
    'Quantidade produzida (Toneladas)': 'sum',
    'Valor da produção (Mil Reais)': 'sum',
    'Rendimento médio da produção (Quilogramas por Hectare)': 'mean'
}).reset_index()

# ==========================================
# CACHE KEY – derivada dos filtros ativos
# Usada em todas as chamadas de correlação
# para garantir invalidação correta do cache
# ==========================================
cache_key_filtros = (
    f"mun={','.join(sorted(municipios_selecionados))}"
    f"|meso={','.join(sorted(mesorregioes_selecionadas))}"
    f"|anos={','.join(str(a) for a in sorted(anos_selecionados))}"
)

# Correlações iniciais (spinner) – já refletem os filtros ativos
with st.spinner("🔍 Analisando correlações climáticas..."):
    df_correlacoes_inicial = calcular_correlacoes_por_ano(
        df_filtrado,
        'Rendimento médio da produção (Quilogramas por Hectare)',
        'Todos',
        cache_key_filtros
    )

# ==========================================
# MÉTRICAS PRINCIPAIS
# ==========================================

# Título e descrição do cabeçalho adaptados dinamicamente ao período selecionado
if len(anos_selecionados) == 0:
    titulo_metricas  = "📊 Indicadores Principais – Nenhum Ano Selecionado"
    descricao_metricas = "⚠️ Selecione ao menos um ano no filtro lateral para visualizar os indicadores."
elif len(anos_selecionados) == 1:
    ano_unico = anos_selecionados[0]
    titulo_metricas    = f"📊 Indicadores Principais – {ano_unico}"
    descricao_metricas = f"📋 Indicadores de produção e rendimento de soja para o ano de {ano_unico}."
else:
    ano_ini = min(anos_selecionados)
    ano_fim = max(anos_selecionados)
    # Verifica se os anos selecionados formam um intervalo contínuo
    intervalo_continuo = sorted(anos_selecionados) == list(range(ano_ini, ano_fim + 1))
    periodo_label = f"{ano_ini}–{ano_fim}" if intervalo_continuo else ", ".join(str(a) for a in sorted(anos_selecionados))
    titulo_metricas    = f"📊 Indicadores Principais – {periodo_label} (consolidado)"
    descricao_metricas = (
        f"📋 Valores consolidados para o período {periodo_label}. "
        f"O delta (▲▼) compara o **último ano ({ano_fim})** com o **penúltimo ano disponível** na seleção."
    )

st.header(titulo_metricas)
st.info(descricao_metricas)

if len(df_agregado) > 0:
    # ── Valores de referência ────────────────────────────────────────────────
    # "Atual": último ano presente em df_agregado (sempre disponível)
    # "Anterior": penúltimo ano – usado apenas para o delta; se não existir,
    #             o delta é omitido (delta_str = None → st.metric sem delta)

    ultimo_ano    = df_agregado.iloc[-1]
    tem_anterior  = len(df_agregado) > 1
    penultimo_ano = df_agregado.iloc[-2] if tem_anterior else None

    # ── Valores consolidados (soma/média de todos os anos selecionados) ──────
    soma_area_plantada  = df_agregado['Área plantada (Hectares)'].sum()
    soma_area_perdida   = df_agregado['Área perdida (Hectares)'].sum()
    soma_producao       = df_agregado['Quantidade produzida (Toneladas)'].sum()
    media_rendimento    = df_agregado['Rendimento médio da produção (Quilogramas por Hectare)'].mean()
    media_perda_pct     = df_agregado['Percentual de perda (%)'].mean()

    # ── Função utilitária: delta percentual entre dois valores ───────────────
    def delta_pct(atual, anterior):
        """Retorna variação % formatada, ou None se não houver referência."""
        if anterior is None or anterior == 0:
            return None
        v = (atual - anterior) / anterior * 100
        return formatar_numero(v, sufixo='%', decimais=2, prefixo='+ ' if v > 0 else '')

    def delta_pp(atual, anterior):
        """Retorna variação em pontos percentuais formatada, ou None."""
        if anterior is None:
            return None
        d = atual - anterior
        return formatar_numero(d, sufixo=' pp', decimais=2, prefixo='+ ' if d > 0 else '')

    # ── Escolha de valor exibido e delta conforme o modo ────────────────────
    if len(anos_selecionados) == 1:
        # Ano único: exibe o valor daquele ano; sem delta (não há comparação)
        val_area_plantada = ultimo_ano['Área plantada (Hectares)']
        val_area_perdida  = ultimo_ano['Área perdida (Hectares)']
        val_producao      = ultimo_ano['Quantidade produzida (Toneladas)']
        val_rendimento    = ultimo_ano['Rendimento médio da produção (Quilogramas por Hectare)']
        val_perda_pct     = ultimo_ano['Percentual de perda (%)']
        d_area_plantada   = None
        d_area_perdida    = None
        d_producao        = None
        d_rendimento      = None
        d_perda_pct       = None
    else:
        # Múltiplos anos: exibe consolidado; delta = último vs. penúltimo ano
        val_area_plantada = soma_area_plantada
        val_area_perdida  = soma_area_perdida
        val_producao      = soma_producao
        val_rendimento    = media_rendimento
        val_perda_pct     = media_perda_pct
        ant = penultimo_ano  # pode ser None se df_agregado tiver 1 linha (raro)
        d_area_plantada = delta_pct(
            ultimo_ano['Área plantada (Hectares)'],
            ant['Área plantada (Hectares)'] if ant is not None else None
        )
        d_area_perdida = delta_pct(
            ultimo_ano['Área perdida (Hectares)'],
            ant['Área perdida (Hectares)'] if ant is not None else None
        )
        d_producao = delta_pct(
            ultimo_ano['Quantidade produzida (Toneladas)'],
            ant['Quantidade produzida (Toneladas)'] if ant is not None else None
        )
        d_rendimento = delta_pct(
            ultimo_ano['Rendimento médio da produção (Quilogramas por Hectare)'],
            ant['Rendimento médio da produção (Quilogramas por Hectare)'] if ant is not None else None
        )
        d_perda_pct = delta_pp(
            ultimo_ano['Percentual de perda (%)'],
            ant['Percentual de perda (%)'] if ant is not None else None
        )

    # ── Renderização ─────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Área Plantada",
            formatar_numero(val_area_plantada, sufixo=' ha'),
            d_area_plantada
        )
    with col2:
        st.metric(
            "Área Perdida",
            formatar_numero(val_area_perdida, sufixo=' ha'),
            d_area_perdida,
            delta_color="inverse"
        )
    with col3:
        st.metric(
            "Produção",
            formatar_numero(val_producao, sufixo=' Kg'),
            d_producao
        )
    with col4:
        st.metric(
            "Rendimento Médio",
            formatar_numero(val_rendimento, sufixo=' kg/ha'),
            d_rendimento
        )
    with col5:
        st.metric(
            "% de Perda",
            formatar_numero(val_perda_pct, sufixo='%', decimais=2),
            d_perda_pct,
            delta_color="inverse"
        )

# ==========================================
# GRÁFICOS PRINCIPAIS
# ==========================================
st.header("📈Análise Produtiva")
st.info("📈 Avaliação temporal da evolução da área cultivada, perdas percentuais e variação da produtividade.")

col1, col2 = st.columns(2)

with col1:
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Bar(x=df_agregado['ano'], y=df_agregado['Área plantada (Hectares)'],
                          name='Área Plantada', marker_color='rgba(46, 204, 113, 0.4)',
                          marker_line_width=1, marker_line_color='#2ecc71'), secondary_y=False)
    fig1.add_trace(go.Scatter(x=df_agregado['ano'], y=df_agregado['Área colhida (Hectares)'],
                              name='Área Colhida', line=dict(color='#27ae60', width=4),
                              mode='lines+markers'), secondary_y=False)
    fig1.add_trace(go.Scatter(x=df_agregado['ano'], y=df_agregado['Área perdida (Hectares)'],
                              name='Área Perdida (Escala Direita)',
                              line=dict(color='#c0392b', width=2, dash='dot'),
                              marker=dict(size=6, color='#c0392b'),
                              fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.2)',
                              mode='lines+markers'), secondary_y=True)
    fig1.update_layout(title='<b>Evolução: Plantio (Barras) vs Perda (Vermelho)</b>',
                       hovermode='x unified', height=450, font=dict(color='black'),
                       separators=',.',
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig1.update_xaxes(type='category', tickfont=dict(color='black'), title_font=dict(color='black'), title="Ano Safra")
    fig1.update_yaxes(title_text="Área Total (ha)", tickfont=dict(color='#27ae60'),
                      title_font=dict(color='#27ae60'), secondary_y=False, showgrid=True)
    fig1.update_yaxes(title_text="Área Perdida (ha)", tickfont=dict(color='#c0392b'),
                      title_font=dict(color='#c0392b'), secondary_y=True, showgrid=False)
    st.plotly_chart(fig1, width="stretch")

with col2:
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=df_agregado['ano'], y=df_agregado['Quantidade produzida (Toneladas)'],
                          name='Produção', marker_color='#3498db'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_agregado['ano'], y=df_agregado['Percentual de perda (%)'],
                              name='% Perda', line=dict(color='#e74c3c', width=3),
                              mode='lines+markers'), secondary_y=True)
    fig2.update_layout(title='<b>Produção e Percentual de Perda</b>', hovermode='x unified',
                       height=450, font=dict(color='black'), separators=',.')
    fig2.update_xaxes(title_text="Ano", type='category', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig2.update_yaxes(title_text="Quilograma", secondary_y=False, tickfont=dict(color='black'), title_font=dict(color='black'))
    fig2.update_yaxes(title_text="% Perda", secondary_y=True, tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig2, width="stretch")

col1, col2 = st.columns(2)

with col1:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_agregado['ano'],
                              y=df_agregado['Rendimento médio da produção (Quilogramas por Hectare)'],
                              mode='lines+markers', line=dict(color='#9b59b6', width=3),
                              marker=dict(size=12)))
    fig3.update_layout(title='<b>Rendimento Médio</b>', xaxis_title='Ano', yaxis_title='kg/ha',
                       height=400, font=dict(color='black'), separators=',.')
    fig3.update_xaxes(type='category', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig3.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig3, width="stretch")

with col2:
    texto_valor = df_agregado['Valor da produção (Mil Reais)'].apply(lambda x: f"R$ {formatar_numero(x)}")
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=df_agregado['ano'], y=df_agregado['Valor da produção (Mil Reais)'],
                          marker_color='#16a085', text=texto_valor, textposition='outside'))
    fig4.update_layout(title='<b>Valor da Produção</b>', xaxis_title='Ano', yaxis_title='Reais (R$)',
                       height=400,
                       yaxis=dict(range=[0, df_agregado['Valor da produção (Mil Reais)'].max() * 1.15]),
                       font=dict(color='black'), separators=',.')
    fig4.update_xaxes(type='category', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig4.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig4, width="stretch")

# ==========================================
# MATRIZ DE CORRELAÇÃO
# ==========================================
st.header("🔗 Matriz de Correlação (Variáveis de Produção)")
st.info("📊 Correlação de Pearson entre as variáveis de área, produção, rendimento e valor.")

cols_correlacao = [
    'Área plantada (Hectares)',
    'Área colhida (Hectares)',
    'Área perdida (Hectares)',
    'Quantidade produzida (Toneladas)',
    'Rendimento médio da produção (Quilogramas por Hectare)',
    'Valor da produção (Mil Reais)',
    'Valor da produção - percentual do total geral',
    'Percentual de perda (%)',
]

cols_validas = [col for col in cols_correlacao if col in df_filtrado.columns]

if len(cols_validas) > 1:
    corr_matrix = df_filtrado[cols_validas].corr()
    text_matrix = corr_matrix.map(lambda x: f"{str(round(x, 2)).replace('.', ',')}")
    fig_corr = px.imshow(corr_matrix, text_auto=False, aspect="auto",
                         color_continuous_scale='RdYlGn', zmin=-1, zmax=1, height=600)
    fig_corr.update_traces(text=text_matrix, texttemplate="%{text}")
    fig_corr.update_layout(title='<b>Matriz de Correlação de Pearson</b>',
                           font=dict(color='black'), separators=',.')
    fig_corr.update_xaxes(tickfont=dict(color='black'))
    fig_corr.update_yaxes(tickfont=dict(color='black'))
    st.plotly_chart(fig_corr, width="stretch")
else:
    st.warning("Colunas insuficientes para gerar a matriz de correlação.")

# ==========================================
# VARIÁVEIS CLIMÁTICAS MAIS RELEVANTES
# ==========================================
st.header("🌤️ Variáveis Climáticas Mais Relevantes")
st.info("📋 **Análise automática:** Identificando as variáveis climáticas com maior correlação com rendimento, produção e perdas.")

col1, col2, col3 = st.columns(3)
with col1:
    top_n = st.slider("Número de variáveis mais relevantes:", 5, 20, 10)
with col2:
    metrica_foco = st.selectbox("Foco da análise:", [
        'Rendimento médio da produção (Quilogramas por Hectare)',
        'Quantidade produzida (Toneladas)',
        'Área perdida (Hectares)',
        'Percentual de perda (%)',
        'Valor da produção (Mil Reais)'
    ])
with col3:
    ano_clima_analise = st.selectbox(
        "Ano para análise:",
        options=["Todos os anos"] + [str(ano) for ano in sorted(anos_selecionados)],
        index=0
    )

if ano_clima_analise == "Todos os anos":
    df_para_correlacao = df_filtrado.copy()
    titulo_ano = "Todos os Anos"
else:
    df_para_correlacao = df_filtrado[df_filtrado['ano'] == int(ano_clima_analise)].copy()
    titulo_ano = ano_clima_analise

# Cache key específica para esta seção (inclui métrica e ano selecionados)
cache_key_clima = (
    f"{cache_key_filtros}"
    f"|metrica={metrica_foco}"
    f"|ano_analise={ano_clima_analise}"
)

df_corr_foco = calcular_correlacoes_por_ano(
    df_para_correlacao,
    metrica_foco,
    ano_clima_analise,
    cache_key_clima
)

if len(df_corr_foco) == 0:
    st.warning("⚠️ Não há dados suficientes para calcular correlações com os filtros selecionados.")
    st.stop()

df_corr_foco = df_corr_foco.nlargest(min(top_n, len(df_corr_foco)), 'Correlação Abs')

st.subheader(f"🔝 Top {len(df_corr_foco)} Variáveis com Maior Impacto - {titulo_ano}")

texto_corr = df_corr_foco['Correlação'].apply(lambda x: f"{x:.3f}".replace('.', ','))
fig_top = go.Figure()
fig_top.add_trace(go.Bar(
    x=df_corr_foco['Correlação'],
    y=[f"{row['Variável Climática']}_dec{row['Decêndio']}_{row['Ano Safra']}" for _, row in df_corr_foco.iterrows()],
    orientation='h',
    marker_color=df_corr_foco['Correlação'],
    marker_colorscale='RdYlGn', marker_cmin=-1, marker_cmax=1,
    text=texto_corr, textposition='outside'
))
fig_top.update_layout(
    title=f'<b>Correlação com: {metrica_foco} ({titulo_ano})</b>',
    xaxis_title='Correlação de Pearson', yaxis_title='Variável Climática',
    height=max(400, len(df_corr_foco) * 30), xaxis_range=[-1, 1],
    font=dict(color='black'), separators=',.'
)
fig_top.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
fig_top.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
fig_top.add_vline(x=0, line_dash="dash", line_color="#000000")
st.plotly_chart(fig_top, width="stretch")

st.subheader("🔍 Análise Detalhada – Top 3 Variáveis")
st.info(f"🔬 Relação entre as três variáveis climáticas de maior impacto e a produtividade - {titulo_ano}")
n_pontos = len(df_para_correlacao)
st.info(f"📊 Análise baseada em **{formatar_numero(n_pontos)} registros** ({titulo_ano})")

top3 = df_corr_foco.head(3)
for idx, row in top3.iterrows():
    corr_fmt = str(round(row['Correlação'], 4)).replace('.', ',')
    with st.expander(f"**{idx+1}. {row['Variável Climática']} - Decêndio {row['Decêndio']} ({row['Ano Safra']})** - Correlação: {corr_fmt}"):
        col1, col2 = st.columns([2, 1])
        with col1:
            df_scatter = df_para_correlacao[[row['Coluna'], metrica_foco, 'ano', 'Município',
                                             'Quantidade produzida (Toneladas)']].dropna()
            titulo_scatter = f"Dispersão ({metrica_foco.split('(')[0].strip()}) × ({row['Variável Climática']})"
            try:
                fig_scatter = px.scatter(df_scatter, x=row['Coluna'], y=metrica_foco,
                                         color='ano', size='Quantidade produzida (Toneladas)',
                                         hover_data=['Município'], trendline='ols', title=titulo_scatter)
            except:
                fig_scatter = px.scatter(df_scatter, x=row['Coluna'], y=metrica_foco,
                                         color='ano', size='Quantidade produzida (Toneladas)',
                                         hover_data=['Município'], title=titulo_scatter)
                if len(df_scatter) > 1:
                    slope, intercept, *_ = stats.linregress(df_scatter[row['Coluna']], df_scatter[metrica_foco])
                    lx = np.array([df_scatter[row['Coluna']].min(), df_scatter[row['Coluna']].max()])
                    fig_scatter.add_trace(go.Scatter(x=lx, y=slope * lx + intercept,
                                                     mode='lines', name='Tendência',
                                                     line=dict(color='red', dash='dash', width=2)))
            fig_scatter.update_layout(height=400, font=dict(color='black'), separators=',.')
            fig_scatter.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
            fig_scatter.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
            st.plotly_chart(fig_scatter, width="stretch")
        with col2:
            st.metric("Correlação", corr_fmt)
            intensidade = "🔴 Forte" if abs(row['Correlação']) > 0.7 else ("🟡 Moderada" if abs(row['Correlação']) > 0.4 else "🟢 Fraca")
            st.metric("Intensidade", intensidade)
            st.metric("Direção", "📈 Positiva" if row['Correlação'] > 0 else "📉 Negativa")
            st.markdown("**Interpretação:**")
            if row['Correlação'] > 0:
                st.success(f"Aumento de {row['Variável Climática']} associado ao aumento de {metrica_foco.split('(')[0].strip()}")
            else:
                st.warning(f"Aumento de {row['Variável Climática']} associado à redução de {metrica_foco.split('(')[0].strip()}")

# ==========================================
# MAPA DE CALOR
# ==========================================
st.header(f"🗺️ Mapa de Calor: Ciclo Completo da Safra - {titulo_ano}")
st.info("📅 **Ciclo da Soja:** Ano 1 (Dec 26-36: Set-Dez) → Ano 2 (Dec 1-15: Jan-Mai).")

variaveis_disponiveis = sorted(df_corr_foco['Variável Climática'].unique())
vars_heatmap = st.multiselect("Selecione variáveis climáticas para o mapa de calor:",
                               options=variaveis_disponiveis,
                               default=variaveis_disponiveis[:min(5, len(variaveis_disponiveis))])

if vars_heatmap:
    decendios_ano1 = list(range(26, 37))
    decendios_ano2 = list(range(1, 16))
    heatmap_data = []
    for var_clima in vars_heatmap:
        for dec_num in decendios_ano1:
            col_name = f"{var_clima}_dec{dec_num}_ano1"
            if col_name in df_para_correlacao.columns:
                try:
                    df_temp = df_para_correlacao[[col_name, metrica_foco]].dropna()
                    if len(df_temp) > 5:
                        corr = df_temp.corr().iloc[0, 1]
                        if not np.isnan(corr):
                            heatmap_data.append({'Variável': var_clima, 'Período': f"Ano1_Dec{dec_num}",
                                                 'Decêndio_Order': dec_num - 26, 'Correlação': corr})
                except: pass
        for dec_num in decendios_ano2:
            col_name = f"{var_clima}_dec{dec_num}_ano2"
            if col_name in df_para_correlacao.columns:
                try:
                    df_temp = df_para_correlacao[[col_name, metrica_foco]].dropna()
                    if len(df_temp) > 5:
                        corr = df_temp.corr().iloc[0, 1]
                        if not np.isnan(corr):
                            heatmap_data.append({'Variável': var_clima, 'Período': f"Ano2_Dec{dec_num}",
                                                 'Decêndio_Order': 11 + (dec_num - 1), 'Correlação': corr})
                except: pass

    df_heatmap = pd.DataFrame(heatmap_data)
    if len(df_heatmap) > 0:
        pivot_heatmap = df_heatmap.pivot_table(values='Correlação', index='Variável',
                                               columns='Período', aggfunc='first')
        colunas_ordenadas = [f"Ano1_Dec{i}" for i in decendios_ano1] + [f"Ano2_Dec{i}" for i in decendios_ano2]
        pivot_heatmap = pivot_heatmap[[c for c in colunas_ordenadas if c in pivot_heatmap.columns]]
        text_heatmap = pivot_heatmap.map(lambda x: f"{x:.2f}".replace('.', ','))
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_heatmap.values, x=pivot_heatmap.columns, y=pivot_heatmap.index,
            colorscale='RdYlGn', zmid=0, text=text_heatmap.values,
            texttemplate='%{text}', textfont={"size": 8},
            colorbar=dict(title="Correlação"), zmin=-1, zmax=1
        ))
        fig_heatmap.update_layout(
            title=f'<b>Correlação ao longo do Ciclo da Safra: {metrica_foco.split("(")[0].strip()} ({titulo_ano})</b>',
            xaxis_title='Período (Ano1: Set-Dez | Ano2: Jan-Mai)', yaxis_title='Variável Climática',
            height=max(500, len(pivot_heatmap) * 70),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9, color='black'), title_font=dict(color='black')),
            yaxis=dict(tickfont=dict(color='black'), title_font=dict(color='black')),
            font=dict(color='black'), separators=',.'
        )
        fig_heatmap.add_vline(x=10.5, line_dash="dash", line_color="white", line_width=2)
        st.plotly_chart(fig_heatmap, width="stretch")

        st.subheader("📊 Correlação Média por Fase da Safra")
        st.info("📋 Correlação média consolidada em cada fase fenológica.")
        col1, col2 = st.columns(2)
        with col1:
            ano1_cols = [c for c in pivot_heatmap.columns if c.startswith("Ano1")]
            if ano1_cols:
                st.metric("Fase 1: Semeadura/Desenvolvimento",
                          formatar_numero(pivot_heatmap[ano1_cols].mean().mean(), decimais=4),
                          help="Ano1 Dec26-36: Set-Dez")
        with col2:
            ano2_cols = [c for c in pivot_heatmap.columns if c.startswith("Ano2")]
            if ano2_cols:
                st.metric("Fase 2: Maturação/Colheita",
                          formatar_numero(pivot_heatmap[ano2_cols].mean().mean(), decimais=4),
                          help="Ano2 Dec1-15: Jan-Mai")

# ==========================================
# RANKING DE MUNICÍPIOS
# ==========================================
st.header("🏘️ Evolução dos Top Municípios")
st.info("📋 Acompanhamento da evolução anual dos principais municípios produtores de soja no Paraná.")

num_municipios = st.slider("Número de municípios no ranking:", 3, 15, 5)

top_prod_municipios  = df_filtrado.groupby('Município')['Quantidade produzida (Toneladas)'].mean().nlargest(num_municipios).index
top_rend_municipios  = df_filtrado.groupby('Município')['Rendimento médio da produção (Quilogramas por Hectare)'].mean().nlargest(num_municipios).index
top_area_municipios  = df_filtrado.groupby('Município')['Área plantada (Hectares)'].mean().nlargest(num_municipios).index
top_valor_municipios = df_filtrado.groupby('Município')['Valor da produção (Mil Reais)'].mean().nlargest(num_municipios).index

col1, col2 = st.columns(2)

with col1:
    fig_p = go.Figure()
    for municipio in top_prod_municipios:
        dm = df_filtrado[df_filtrado['Município'] == municipio].sort_values('ano')
        fig_p.add_trace(go.Scatter(x=dm['ano'], y=dm['Quantidade produzida (Toneladas)'],
                                   mode='lines+markers', name=municipio,
                                   line=dict(width=2), marker=dict(size=8)))
    fig_p.update_layout(title=f'<b>Top {num_municipios} – Evolução da Produção Total</b>',
                        xaxis_title='Ano', yaxis_title='Produção (Kg)', height=500,
                        hovermode='x unified',
                        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                        font=dict(color='black'), separators=',.')
    fig_p.update_xaxes(type='linear', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig_p.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig_p, width="stretch")

with col2:
    fig_r = go.Figure()
    for municipio in top_rend_municipios:
        dm = df_filtrado[df_filtrado['Município'] == municipio].sort_values('ano')
        fig_r.add_trace(go.Scatter(x=dm['ano'], y=dm['Rendimento médio da produção (Quilogramas por Hectare)'],
                                   mode='lines+markers', name=municipio,
                                   line=dict(width=2), marker=dict(size=8)))
    fig_r.update_layout(title=f'<b>Top {num_municipios} – Evolução da Produtividade</b>',
                        xaxis_title='Ano', yaxis_title='Rendimento (kg/ha)', height=500,
                        hovermode='x unified',
                        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                        font=dict(color='black'), separators=',.')
    fig_r.update_xaxes(type='linear', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig_r.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig_r, width="stretch")

col3, col4 = st.columns(2)

with col3:
    fig_a = go.Figure()
    for municipio in top_area_municipios:
        dm = df_filtrado[df_filtrado['Município'] == municipio].sort_values('ano')
        fig_a.add_trace(go.Scatter(x=dm['ano'], y=dm['Área plantada (Hectares)'],
                                   mode='lines+markers', name=municipio,
                                   line=dict(width=2), marker=dict(size=8)))
    fig_a.update_layout(title=f'<b>Top {num_municipios} – Evolução da Área Plantada</b>',
                        xaxis_title='Ano', yaxis_title='Área Plantada (ha)', height=500,
                        hovermode='x unified',
                        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                        font=dict(color='black'), separators=',.')
    fig_a.update_xaxes(type='linear', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig_a.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig_a, width="stretch")

with col4:
    fig_v = go.Figure()
    for municipio in top_valor_municipios:
        dm = df_filtrado[df_filtrado['Município'] == municipio].sort_values('ano')
        fig_v.add_trace(go.Scatter(x=dm['ano'], y=dm['Valor da produção (Mil Reais)'],
                                   mode='lines+markers', name=municipio,
                                   line=dict(width=2), marker=dict(size=8)))
    fig_v.update_layout(title=f'<b>Top {num_municipios} – Valor da produção</b>',
                        xaxis_title='Ano', yaxis_title='Valor da produção (R$)', height=500,
                        hovermode='x unified',
                        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                        font=dict(color='black'), separators=',.')
    fig_v.update_xaxes(type='linear', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig_v.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig_v, width="stretch")

# ==========================================
# RODAPÉ
# ==========================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #683;'>
        🌱 <b>Dashboard Inteligente - Soja Paraná</b> | 
        Fonte: PAM/SIDRA + NASA POWER | Desenvolvido por: Bruno Proença de Souza
    </div>
""", unsafe_allow_html=True)