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
from pathlib import Path

try:
    from .stats_utils import calcular_pearson, formatar_correlacao_pvalor
except ImportError:
    from stats_utils import calcular_pearson, formatar_correlacao_pvalor

# ==========================================
# NOMES CANÔNICOS DAS COLUNAS
# Definidos uma única vez após conversão de unidades.
# Qualquer referência futura usa estas constantes.
# ==========================================
COL_PRODUCAO  = 'producao_kg'                       # convertido de Toneladas × 1000
COL_VALOR     = 'valor_producao_rs'                 # convertido de Mil Reais × 1000
COL_VALOR_CORRIGIDO = 'valor_producao_ipca_rs'      # Valor em R$ 2024 (base IPCA)
COL_VALOR_POR_HA = 'valor_producao_ipca_mil_reais_ha'
COL_FATOR_CORRECAO = 'fator_correcao_ipca'          # Fator de deflação IPCA
COL_AREA_P    = 'area_plantada_ha'
COL_AREA_C    = 'area_colhida_ha'
COL_AREA_PERD        = 'area_perdida_ha'
COL_REND      = 'produtividade_kg_ha'
COL_VALOR_PCT = 'participacao_valor_pct'
COL_PERDA_PCT = 'taxa_perda_pct'

LABEL_COLUNAS = {
    COL_AREA_P: 'Área Plantada (ha)',
    COL_AREA_C: 'Área Colhida (ha)',
    COL_AREA_PERD: 'Área Perdida (ha)',
    COL_PRODUCAO: 'Produção (kg)',
    COL_REND: 'Produtividade (kg/ha)',
    COL_VALOR: 'Valor da Produção (R$)',
    COL_VALOR_CORRIGIDO: 'Valor Corrigido (R$)',
    COL_VALOR_POR_HA: 'Valor Corrigido IPCA (mil R$/ha)',
    COL_VALOR_PCT: 'Participação no Valor (%)',
    COL_PERDA_PCT: 'Taxa de Perda (%)',
}


def label_coluna(coluna):
    return LABEL_COLUNAS.get(coluna, coluna)

# ==========================================
# FORMATAÇÃO PT-BR
# ==========================================
def formatar_numero(valor, prefixo='', sufixo='', decimais=0):
    """Formata números para o padrão brasileiro (1.000,00)."""
    if pd.isna(valor):
        return "-"
    s = f"{valor:,.{decimais}f}" if decimais > 0 else f"{valor:,.0f}"
    s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"{prefixo}{s}{sufixo}".strip()


def formatar_inteligente(valor, prefixo='', sufixo=''):
    """
    Formata números grandes de forma legível (pt-BR), evitando
    sequências de dígitos que tornam métricas ilegíveis.

      ≥ 1 000 000 000  →  "X,X bi"
      ≥     1 000 000  →  "X,X mi"
      ≥        10 000  →  "X,X mil"
      < 10 000         →  formatação padrão sem sufixo de escala
    """
    if pd.isna(valor):
        return "-"

    abs_val = abs(valor)
    sinal   = '- ' if valor < 0 else ''

    if abs_val >= 1_000_000_000:
        s = f"{abs_val / 1_000_000_000:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        return f"{sinal}{prefixo}{s} bi{sufixo}".strip()
    if abs_val >= 1_000_000:
        s = f"{abs_val / 1_000_000:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        return f"{sinal}{prefixo}{s} mi{sufixo}".strip()
    if abs_val >= 10_000:
        s = f"{abs_val / 1_000:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        return f"{sinal}{prefixo}{s} mil{sufixo}".strip()
    return formatar_numero(valor, prefixo=f"{sinal}{prefixo}", sufixo=sufixo)


# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Dashboard Soja Paraná",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
.main { padding: 0rem 1rem; }

h1 {
    color: #2c5f2d;
    text-align: center;
    padding: 20px;
    margin-bottom: 10px;
}

.stMetric {
    background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #0f172a 100%);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 6px 18px rgba(0,0,0,0.5);
}

[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 600; }
[data-testid="stMetricLabel"] { color: #d1d5db !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌱 Dashboard - Soja no Paraná (2018-2024)</h1>", unsafe_allow_html=True)

# ==========================================
# CARREGAMENTO DE DADOS
# ==========================================
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_parquet('./data/processed/dataset_final.parquet')
        df['cod_ibge'] = df['cod_ibge'].astype(str)

        # Renomeia colunas PAM para manter compatibilidade
        df = df.rename(columns={
            'area_plantada_ha':
                COL_AREA_P,
            'area_plantada_pct':
                'area_plantada_pct',
            'rendimento_kg_ha':
                COL_REND,
            'valor_producao_pct':
                COL_VALOR_PCT,
            'valor_producao_ipca_mil_reais':
                COL_VALOR_CORRIGIDO,
            'valor_producao_ipca_mil_reais_ha':
                COL_VALOR_POR_HA,
            'valor_corrigido':
                COL_VALOR_CORRIGIDO,
            'valor_por_ha':
                COL_VALOR_POR_HA,
            'fator_correcao':
                COL_FATOR_CORRECAO,
            'ano':         'ano',
            'mesorregiao': 'Mesorregião',
        })

        # ── Colunas derivadas ──────────────────────────────────────────────
        df[COL_AREA_PERD] = df[COL_AREA_P] - df[COL_AREA_C]
        df[COL_PERDA_PCT] = np.where(df[COL_AREA_P] > 0,(df[COL_AREA_PERD] / df[COL_AREA_P]) * 100, 0.0)

        # ── Conversão de unidades + renomeação IMEDIATA ────────────────────
        # Toneladas → kg  (× 1 000)
        df[COL_PRODUCAO] = df['quantidade_produzida_ton'] * 1_000
        df.drop(columns=['quantidade_produzida_ton'], inplace=True)

        # ── Prioridade de Valor: correção IPCA 2024 > Valor nominal
        # Se valor_producao_ipca_mil_reais existe, converte para R$ no dashboard.
        if COL_VALOR_CORRIGIDO in df.columns:
            # Valor corrigido pelo IPCA vem em mil reais no parquet; converter para R$.
            df[COL_VALOR] = df[COL_VALOR_CORRIGIDO] * 1_000
            df[COL_VALOR_CORRIGIDO] = df[COL_VALOR]
            if COL_VALOR_POR_HA in df.columns:
                df[COL_VALOR_POR_HA] = df[COL_VALOR_POR_HA]  # já existe, apenas renomeia
            if COL_FATOR_CORRECAO in df.columns:
                df[COL_FATOR_CORRECAO] = df[COL_FATOR_CORRECAO]  # já existe
        else:
            # Fallback: usa valor nominal original (compatibilidade com dados antigos)
            # Mil Reais → R$  (× 1 000)
            df[COL_VALOR] = df['valor_producao_mil_reais'] * 1_000
            st.info("ℹ️ Dados em valores nominais (sem correção IPCA)")

        df.drop(columns=['valor_producao_mil_reais'], inplace=True)

        # ── Chave numérica para o mapa ─────────────────────────────────────
        df['codigo_ibge'] = (
            df['cod_ibge'].astype(str).str.zfill(7).str[:7].astype(int)
        )

        if 'Mesorregião' not in df.columns:
            st.warning(
                "Aviso: Coluna 'mesorregiao' não encontrada. "
                "O filtro de mesorregião não terá efeito."
            )
            df['Mesorregião'] = "Mesorregião Desconhecida"

        return df

    except FileNotFoundError:
        st.error("⚠️ Arquivo 'dataset_final.parquet' não encontrado!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        st.stop()


@st.cache_data
def carregar_descricoes_atributos():
    """
    Carrega o CSV com as descrições dos atributos climáticos.
    Retorna dict {atributo: descrição} ou vazio se não encontrado.
    """

    try:
        caminho_atributos = Path(__file__).parent / "Atributos.csv"
        df_desc = pd.read_csv(caminho_atributos)
        if 'Atributo' in df_desc.columns and 'Descrição' in df_desc.columns:
            return dict(zip(df_desc['Atributo'].str.strip(),
                            df_desc['Descrição'].str.strip()))
    except FileNotFoundError:
        st.error("⚠️ Arquivo 'Atributos.csv' não encontrado!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar descrições: {e}")
        st.stop()
    return {}


df             = carregar_dados()
desc_atributos = carregar_descricoes_atributos()

# ── Verificar e Exibir Status de Correção IPCA ────────────────────────────────
if COL_VALOR_CORRIGIDO in df.columns:
    pass
elif COL_FATOR_CORRECAO in df.columns:
    st.info(
        "ℹ️ **Dados Parcialmente Corrigidos**\n\n"
        "Fator de correção disponível, mas colunas de valor corrigido podem estar em processamento."
    )
else:
    st.info(
        "ℹ️ **Dados em Valores Nominais**\n\n"
        "Execute o pipeline (merge.py) para aplicar correção IPCA aos valores de produção."
    )

def _base_atributo(atributo: str) -> str:
    """Remove o sufixo _dec{N}_ano{N} e retorna o código base."""
    return re.sub(r'_dec.*$', '', atributo.strip())


def desc_clima(atributo: str) -> str:
    """
    Descrição legível de um atributo climático.
    Aceita tanto 'AIRMASS' quanto 'AIRMASS_dec1_ano1'.
    Retorna a descrição do CSV, ou apenas o código base se não encontrado.
    """
    base = _base_atributo(atributo)
    return desc_atributos.get(base, base)


def label_atributo(atributo: str) -> str:
    """
    Formato canônico para exibição ao usuário: 'Descrição (CÓDIGO)'.
    Se o atributo não estiver no CSV, exibe apenas o código, sem repetição.

    Exemplos:
      AIRMASS  →  'Massa de ar (AIRMASS)'
      RHOA     →  'RHOA'            ← sem descrição no CSV; não repete
    """
    base = _base_atributo(atributo)
    desc = desc_atributos.get(base)          # None se não encontrado
    return f"{base} = {desc}" if desc else base


def label_clima_grafico(atributo: str, decendio: int, ano_safra: str) -> str:
    """
    Label para eixos Plotly: 'Descrição (CÓDIGO)  ·  dec{N} {ano_safra}'.
    """
    return f"{label_atributo(atributo)}  ·  dec{decendio} {ano_safra}"


# Identifica colunas climáticas
colunas_climaticas  = [c for c in df.columns if re.match(r'.*_dec\d+_ano\d+', c)]
atributos_climaticos = sorted(set(c.rsplit('_dec', 1)[0] for c in colunas_climaticas))


# ==========================================
# CORRELAÇÃO – com cache_key explícita
# ==========================================
@st.cache_data
def calcular_correlacoes_por_ano(_df, metrica, cache_key):
    """
    Calcula correlações de Pearson entre variáveis climáticas e uma métrica.
    cache_key garante invalidação quando os filtros mudam.
    """
    resultados = []
    for col_clima in colunas_climaticas:
        try:
            df_temp = _df[[col_clima, metrica]].dropna()
            if len(df_temp) > 5:
                corr, p_valor = calcular_pearson(df_temp, col_clima, metrica)
                if not np.isnan(corr):
                    atributo   = col_clima.rsplit('_dec', 1)[0]
                    dec_match  = re.search(r'dec(\d+)', col_clima)
                    ano_match  = re.search(r'ano(\d+)', col_clima)
                    if dec_match and ano_match:
                        resultados.append({
                            'Variável Climática': atributo,
                            'Decêndio':   int(dec_match.group(1)),
                            'Ano Safra':  f"ano{ano_match.group(1)}",
                            'Coluna':     col_clima,
                            'P-valor':    p_valor,
                            'Correlação': corr,
                            'Correlação Abs': abs(corr)
                        })
        except Exception:
            continue
    return pd.DataFrame(resultados)


# ==========================================
# SIDEBAR – Filtros
# ==========================================
st.sidebar.header("🔍 Filtros de Análise")

anos_disponiveis  = sorted(df['ano'].unique())
anos_selecionados = st.sidebar.multiselect(
    "Selecione os anos:", options=anos_disponiveis, default=anos_disponiveis
)

mesorregioes_disponiveis = sorted(df['Mesorregião'].dropna().unique())
todas_mesorregioes = st.sidebar.checkbox("Selecionar todas as Mesorregiões", value=True)

mesorregioes_selecionadas = (
    mesorregioes_disponiveis if todas_mesorregioes
    else st.sidebar.multiselect(
        "Selecione as Mesorregiões:",
        options=mesorregioes_disponiveis,
        default=mesorregioes_disponiveis,
    )
)

df_filtrado_meso = df[df['Mesorregião'].isin(mesorregioes_selecionadas)]
municipios_disponiveis_filtrados = sorted(df_filtrado_meso['municipio'].unique())

visualizar_todos = st.sidebar.radio(
    "municipios:",
    options=["Todos os municipios da(s) mesorregião(ões)", "Selecionar específicos"],
    index=0,
)

if visualizar_todos == "Todos os municipios da(s) mesorregião(ões)":
    municipios_selecionados = municipios_disponiveis_filtrados
else:
    municipios_selecionados = st.sidebar.multiselect(
        "Escolha os municipios:",
        options=municipios_disponiveis_filtrados,
        default=municipios_disponiveis_filtrados[:5]
            if len(municipios_disponiveis_filtrados) >= 5
            else municipios_disponiveis_filtrados,
    )

# ── Filtro principal ──────────────────────────────────────────────────────────
df_filtrado = df[
    (df['ano'].isin(anos_selecionados)) &
    (df['municipio'].isin(municipios_selecionados))
].copy()

st.sidebar.markdown("---")
st.sidebar.header("📊 Informações")
st.sidebar.metric("Mesorregiões",        len(mesorregioes_selecionadas))
st.sidebar.metric("municipios",          len(municipios_selecionados))
st.sidebar.metric("Anos",               len(anos_selecionados))
st.sidebar.metric("Registros",          formatar_numero(len(df_filtrado)))
st.sidebar.metric("Variáveis Climáticas", len(colunas_climaticas))

# ==========================================
# AGREGAÇÃO ANUAL
# FIX: produtividade recalculada como média ponderada (produção / área colhida)
# FIX: percentual de perda recalculado sobre os totais anuais
# FIX: sort_values garante ordem cronológica antes de iloc
# ==========================================
df_agregado = (
    df_filtrado
    .groupby('ano')
    .agg(
        **{
            COL_AREA_P:    (COL_AREA_P,    'sum'),
            COL_AREA_C:    (COL_AREA_C,    'sum'),
            COL_AREA_PERD: (COL_AREA_PERD, 'sum'),
            COL_PRODUCAO:  (COL_PRODUCAO,  'sum'),
            COL_VALOR:     (COL_VALOR,     'sum'),
        }
    )
    .reset_index()
    .sort_values('ano')
    .reset_index(drop=True)
)

# Recalcula métricas derivadas sobre os totais anuais
df_agregado[COL_REND] = (
    df_agregado[COL_PRODUCAO] / df_agregado[COL_AREA_C]
)
df_agregado[COL_PERDA_PCT] = np.where(df_agregado[COL_AREA_P] > 0, df_agregado[COL_AREA_PERD] / df_agregado[COL_AREA_P] * 100, 0.0)


# ── Cache key derivada dos filtros ativos ─────────────────────────────────────
cache_key_filtros = (
    f"mun={','.join(sorted(municipios_selecionados))}"
    f"|meso={','.join(sorted(mesorregioes_selecionadas))}"
    f"|anos={','.join(str(a) for a in sorted(anos_selecionados))}"
)

# ── Correlações iniciais ──────────────────────────────────────────────────────
with st.spinner("🔍 Analisando correlações climáticas..."):
    df_correlacoes_inicial = calcular_correlacoes_por_ano(
        df_filtrado,
        COL_REND,
        cache_key_filtros,
    )


# ==========================================
# MÉTRICAS PRINCIPAIS
# ==========================================
# Lógica de título / descrição
if len(anos_selecionados) == 0:
    titulo_metricas    = "📊 Indicadores Principais – Nenhum Ano Selecionado"
    descricao_metricas = "⚠️ Selecione ao menos um ano no filtro lateral para visualizar os indicadores."
elif len(anos_selecionados) == 1:
    ano_unico          = anos_selecionados[0]
    titulo_metricas    = f"📊 Indicadores Principais – {ano_unico}"
    descricao_metricas = f"📋 Indicadores de produção e produtividade de soja para o ano de {ano_unico}."
else:
    ano_ini = min(anos_selecionados)
    ano_fim = max(anos_selecionados)
    intervalo_continuo = sorted(anos_selecionados) == list(range(ano_ini, ano_fim + 1))
    periodo_label = (
        f"{ano_ini}–{ano_fim}" if intervalo_continuo
        else ", ".join(str(a) for a in sorted(anos_selecionados))
    )
    titulo_metricas    = f"📊 Indicadores Principais – {periodo_label} (consolidado)"
    descricao_metricas = (
        f"📋 **Área plantada, área perdida e produção** exibem a **soma total** do período {periodo_label}. "
        f"**Produtividade** e **taxa de perda** são calculadas sobre os **totais agregados** (média ponderada pela área). "
        f"O delta (▲▼) compara **{ano_fim} versus {ano_ini}**."
    )

st.header(titulo_metricas)
st.info(descricao_metricas)

if len(df_agregado) > 0:
    # ── Valores consolidados do período ──────────────────────────────────────
    soma_area_plantada = df_agregado[COL_AREA_P].sum()
    soma_area_colhida  = df_agregado[COL_AREA_C].sum()
    soma_area_perdida  = df_agregado[COL_AREA_PERD].sum()
    soma_producao      = df_agregado[COL_PRODUCAO].sum()
    soma_valor         = df_agregado[COL_VALOR].sum()

    # Produtividade consolidada: média ponderada = Σprodução / Σárea_colhida
    produtividade_consolidada = soma_producao / soma_area_colhida if soma_area_colhida > 0 else np.nan

    # % de perda consolidada: calculada sobre os totais, não média de médias
    perda_pct_consolidada  = (soma_area_perdida / soma_area_plantada * 100) if soma_area_plantada > 0 else np.nan

    # ── Referências para o delta ──────────────────────────────────────────────
    # Delta: último ano selecionado vs. primeiro ano selecionado (variação total)
    # Para ano único: sem delta (sem comparação possível)
    primeiro_ano = df_agregado.iloc[0]
    ultimo_ano   = df_agregado.iloc[-1]
    tem_dois     = len(df_agregado) > 1

    def delta_pct_fmt(val_atual, val_ref, sufixo_abs=' ha', sempre_pct=False):
        """
        Variação percentual formatada (pt-BR), ou None quando não há referência.

        Casos especiais de base zero:
        - ref == 0 e atual == 0  → '= 0' (sem mudança)
        - ref == 0 e atual != 0  → variação absoluta (% indefinida) ou '∞ %' se sempre_pct=True
        """
        if val_ref is None or not tem_dois or pd.isna(val_ref) or pd.isna(val_atual):
            return None
        if val_ref == 0:
            if val_atual == 0:
                return "= 0"
            if sempre_pct:
                return "∞ %"
            # Base zero: % indefinida → mostra variação absoluta com unidade
            diff  = val_atual - val_ref
            sinal = '+ ' if diff > 0 else ''
            return formatar_inteligente(diff, prefixo=sinal, sufixo=sufixo_abs)
        v = (val_atual - val_ref) / abs(val_ref) * 100
        sinal = '+ ' if v > 0 else ''
        return formatar_numero(v, prefixo=sinal, sufixo=' %', decimais=2)

    def delta_pp_fmt(val_atual, val_ref):
        """Variação em pontos percentuais formatada, ou None."""
        if val_ref is None or not tem_dois or pd.isna(val_ref):
            return None
        d = val_atual - val_ref
        sinal = '+ ' if d > 0 else ''
        return formatar_numero(d, prefixo=sinal, sufixo=' pp', decimais=2)

    # ── Seleção de valor exibido e delta ─────────────────────────────────────
    if len(anos_selecionados) == 1:
        # Ano único: valor pontual; sem delta
        val_area_p   = ultimo_ano[COL_AREA_P]
        val_area_perd = ultimo_ano[COL_AREA_PERD]
        val_prod     = ultimo_ano[COL_PRODUCAO]
        val_produtividade = ultimo_ano[COL_REND]
        val_perda    = ultimo_ano[COL_PERDA_PCT]
        d_area_p = d_area_perd = d_prod = d_produtividade = d_perda = None
    else:
        # Período: consolidado; delta = último ano vs. primeiro ano da seleção
        val_area_p    = soma_area_plantada
        val_area_perd = soma_area_perdida
        val_prod      = soma_producao
        val_produtividade = produtividade_consolidada
        val_perda     = perda_pct_consolidada

        d_area_p    = delta_pct_fmt(ultimo_ano[COL_AREA_P],    primeiro_ano[COL_AREA_P],    sufixo_abs=' ha')
        d_area_perd = delta_pct_fmt(ultimo_ano[COL_AREA_PERD], primeiro_ano[COL_AREA_PERD], sufixo_abs=' ha', sempre_pct=True)
        d_prod      = delta_pct_fmt(ultimo_ano[COL_PRODUCAO],  primeiro_ano[COL_PRODUCAO],  sufixo_abs=' kg')
        d_produtividade = delta_pct_fmt(ultimo_ano[COL_REND],      primeiro_ano[COL_REND],      sufixo_abs=' kg/ha')
        d_perda     = delta_pp_fmt( ultimo_ano[COL_PERDA_PCT],  primeiro_ano[COL_PERDA_PCT])

    ano_ref_label = (
        f"Δ {int(ultimo_ano['ano'])} vs {int(primeiro_ano['ano'])}"
        if tem_dois else "Ano único"
    )

    # ── Renderização ──────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Área Plantada (ha)",
            formatar_inteligente(val_area_p, sufixo=' ha'),
            d_area_p,
            help=f"Soma total de hectares plantados no período. {ano_ref_label}.",
        )
    with col2:
        st.metric(
            "Área Perdida (ha)",
            formatar_inteligente(val_area_perd, sufixo=' ha'),
            d_area_perd,
            delta_color="inverse",
            help=f"Soma de área plantada menos área colhida. {ano_ref_label}.",
        )
    with col3:
        st.metric(
            "Produção (kg)",
            formatar_inteligente(val_prod, sufixo=' kg'),
            d_prod,
            help=f"Soma total de quilogramas produzidos no período. {ano_ref_label}.",
        )
    with col4:
        st.metric(
            "Produtividade (kg/ha)",
            formatar_inteligente(val_produtividade, sufixo=' kg/ha'),
            d_produtividade,
            help=(
                "Média ponderada pela área colhida: Σ(produção) ÷ Σ(área colhida). "
                f"Evita distorção por anos com áreas menores. {ano_ref_label}."
            ),
        )
    with col5:
            st.metric(
                "Taxa de Perda (%)",
                formatar_numero(val_perda, sufixo=' %', decimais=2),
                d_perda,
                delta_color="inverse",
                help=(
                    "Calculado sobre os totais: Σ(área perdida) ÷ Σ(área plantada) × 100. "
                    f"Não é média de percentuais anuais. {ano_ref_label}."
                ),
            )
    # ── Métricas Econômicas Corrigidas (IPCA 2024) ──────────────────────────
    if COL_VALOR_CORRIGIDO in df_agregado.columns:
        st.divider()
        st.markdown("### 💰 Métricas Econômicas (Base 2024 - IPCA Corrigido)")
        
        col_ec1, col_ec2, col_ec3 = st.columns(3)
        
        soma_valor_ipca_rs = df_agregado[COL_VALOR_CORRIGIDO].sum()
        d_valor_ec = delta_pct_fmt(
            ultimo_ano[COL_VALOR_CORRIGIDO], 
            primeiro_ano[COL_VALOR_CORRIGIDO], 
            sufixo_abs=' reais'
        ) if COL_VALOR_CORRIGIDO in df_agregado.columns else None
        
        with col_ec1:
            st.metric(
                "Valor Corrigido (R$)",
                formatar_inteligente(soma_valor_ipca_rs, prefixo='R$ '),
                d_valor_ec,
                help=f"Soma do valor de produção corrigido pela inflação IPCA (base 2024). {ano_ref_label}.",
            )
        
        if COL_VALOR_POR_HA in df_agregado.columns:
            media_valor_ipca_mil_reais_ha = df_agregado[COL_VALOR_POR_HA].mean()
            d_valor_ha = delta_pct_fmt(
                ultimo_ano[COL_VALOR_POR_HA], 
                primeiro_ano[COL_VALOR_POR_HA],
                sufixo_abs=' mil reais/ha'
            ) if len(df_agregado) > 1 else None
            
            with col_ec2:
                st.metric(
                    "Valor Corrigido por ha",
                    formatar_inteligente(media_valor_ipca_mil_reais_ha, prefixo='R$ ', sufixo=' mil/ha'),
                    d_valor_ha,
                    help=f"Valor corrigido dividido pela área plantada, indicador de rentabilidade. {ano_ref_label}.",
                )
        
        # Mostra fatores de correção se disponível
        if COL_FATOR_CORRECAO in df_agregado.columns:
            with col_ec3:
                fatores = df_agregado[COL_FATOR_CORRECAO].unique()
                if len(fatores) == 1:
                    st.metric(
                        "Fator IPCA",
                        f"{fatores[0]:.4f}",
                        help=f"Fator de deflação aplicado para trazer valores a base 2024.",
                    )
                else:
                    st.info(
                        f"**Fatores IPCA variáveis** (diferentes anos selecionados)\n\n"
                        f"Min: {fatores.min():.4f} | Max: {fatores.max():.4f}"
                    )

# ==========================================
# GRÁFICOS PRINCIPAIS
# ==========================================
st.header("📈 Análise Produtiva")
st.info("📈 Avaliação temporal da evolução da área cultivada, perdas percentuais e variação da produtividade.")

col1, col2 = st.columns(2)

with col1:
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Bar(
        x=df_agregado['ano'], y=df_agregado[COL_AREA_P],
        name='Área Plantada',
        marker_color='rgba(46, 204, 113, 0.4)',
        marker_line_width=1, marker_line_color='#2ecc71',
    ), secondary_y=False)
    fig1.add_trace(go.Scatter(
        x=df_agregado['ano'], y=df_agregado[COL_AREA_C],
        name='Área Colhida',
        line=dict(color='#27ae60', width=4), mode='lines+markers',
    ), secondary_y=False)
    fig1.add_trace(go.Scatter(
        x=df_agregado['ano'], y=df_agregado[COL_AREA_PERD],
        name='Área Perdida (eixo direito)',
        line=dict(color='#c0392b', width=2, dash='dot'),
        marker=dict(size=6, color='#c0392b'),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.2)',
        mode='lines+markers',
    ), secondary_y=True)
    fig1.update_layout(
        title='<b>Evolução: Plantio (Barras) vs Perda (Vermelho)</b>',
        hovermode='x unified', height=450, font=dict(color='black'), separators=',.',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig1.update_xaxes(type='category', tickfont=dict(color='black'),
                      title_font=dict(color='black'), title="Ano Safra")
    fig1.update_yaxes(title_text="Área Total (ha)",
                      tickfont=dict(color='#27ae60'), title_font=dict(color='#27ae60'),
                      secondary_y=False, showgrid=True)
    fig1.update_yaxes(title_text="Área Perdida (ha)",
                      tickfont=dict(color='#c0392b'), title_font=dict(color='#c0392b'),
                      secondary_y=True, showgrid=False)
    st.plotly_chart(fig1, width='stretch')

with col2:
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(
        x=df_agregado['ano'], y=df_agregado[COL_PRODUCAO],
        name='Produção (kg)', marker_color='#3498db',
    ), secondary_y=False)
    fig2.add_trace(go.Scatter(
        x=df_agregado['ano'], y=df_agregado[COL_PERDA_PCT],
        name='Taxa de Perda (%)', line=dict(color='#e74c3c', width=3), mode='lines+markers',
    ), secondary_y=True)
    fig2.update_layout(
        title='<b>Produção (kg) e Taxa de Perda (%)</b>',
        hovermode='x unified', height=450, font=dict(color='black'), separators=',.',
    )
    fig2.update_xaxes(title_text="Ano", type='category',
                      tickfont=dict(color='black'), title_font=dict(color='black'))
    fig2.update_yaxes(title_text="Quilogramas (kg)", secondary_y=False,
                      tickfont=dict(color='black'), title_font=dict(color='black'))
    fig2.update_yaxes(title_text="Taxa de Perda (%)", secondary_y=True,
                      tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig2, width='stretch')

col1, col2 = st.columns(2)

with col1:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_agregado['ano'], y=df_agregado[COL_REND],
        mode='lines+markers', line=dict(color='#9b59b6', width=3),
        marker=dict(size=12),
    ))
    fig3.update_layout(
        title='<b>Produtividade Média Ponderada (Σkg ÷ Σha colhido)</b>',
        xaxis_title='ano', yaxis_title='Produtividade (kg/ha)',
        height=400, font=dict(color='black'), separators=',.',
    )
    fig3.update_xaxes(type='category', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig3.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig3, width='stretch')

with col2:
    # ── Prioriza valor corrigido (IPCA 2024) se disponível ────────────────
    col_valor_exibir = COL_VALOR_CORRIGIDO if (COL_VALOR_CORRIGIDO in df_agregado.columns) else COL_VALOR
    titulo_valor = 'Valor Corrigido (R$)' if col_valor_exibir == COL_VALOR_CORRIGIDO else 'Valor da Produção (R$)'
    
    texto_valor = df_agregado[col_valor_exibir].apply(
        lambda x: f"R$ {formatar_inteligente(x)}"
    )
    fig4 = go.Figure()

    y_valores = df_agregado[col_valor_exibir] if col_valor_exibir == COL_VALOR_CORRIGIDO else df_agregado[col_valor_exibir]

    fig4.add_trace(go.Bar(
        x=df_agregado['ano'], y=y_valores,
        marker_color='#16a085', text=texto_valor, textposition='outside',
    ))
    fig4.update_layout(
        title=f'<b>{titulo_valor}</b>',
        xaxis_title='ano', yaxis_title='Valor Corrigido (R$)' if col_valor_exibir == COL_VALOR_CORRIGIDO else 'Valor da Produção (R$)',
        height=400,
        yaxis=dict(range=[0, y_valores.max() * 1.15]),
        font=dict(color='black'), separators=',.',
    )
    fig4.update_xaxes(type='category', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig4.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig4, width='stretch')


# ==========================================
# MATRIZ DE CORRELAÇÃO
# ==========================================
st.header("🔗 Matriz de Correlação (Variáveis de Produção)")
st.info("📊 Correlação de Pearson entre as variáveis de área, produção, produtividade e valor.")

cols_correlacao = [
    COL_AREA_P,
    COL_AREA_C,
    COL_AREA_PERD,
    COL_PRODUCAO,
    COL_REND,
    COL_VALOR,
    COL_VALOR_PCT,
    COL_PERDA_PCT,
]

# Adiciona colunas corrigidas se disponíveis
if COL_VALOR_CORRIGIDO in df_filtrado.columns and COL_VALOR_CORRIGIDO not in cols_correlacao:
    cols_correlacao.insert(6, COL_VALOR_CORRIGIDO)  # Após COL_VALOR
if COL_VALOR_POR_HA in df_filtrado.columns and COL_VALOR_POR_HA not in cols_correlacao:
    cols_correlacao.append(COL_VALOR_POR_HA)  # Ao final
cols_validas = [c for c in cols_correlacao if c in df_filtrado.columns]

if len(cols_validas) > 1:
    corr_matrix  = df_filtrado[cols_validas].corr()
    corr_matrix = corr_matrix.rename(index=LABEL_COLUNAS, columns=LABEL_COLUNAS)
    text_matrix  = corr_matrix.map(lambda x: f"{str(round(x, 2)).replace('.', ',')}")
    fig_corr = px.imshow(
        corr_matrix, text_auto=False, aspect="auto",
        color_continuous_scale='RdYlGn', zmin=-1, zmax=1, height=600,
    )
    fig_corr.update_traces(text=text_matrix, texttemplate="%{text}")
    fig_corr.update_layout(
        title='<b>Matriz de Correlação de Pearson</b>',
        font=dict(color='black'), separators=',.',
    )
    fig_corr.update_xaxes(tickfont=dict(color='black'))
    fig_corr.update_yaxes(tickfont=dict(color='black'))
    st.plotly_chart(fig_corr, width='stretch')
else:
    st.warning("Colunas insuficientes para gerar a matriz de correlação.")


# ==========================================
# VARIÁVEIS CLIMÁTICAS MAIS RELEVANTES
# ==========================================
st.header("🌤️ Variáveis Climáticas Mais Relevantes")
st.info(
    "📋 **Análise automática:** Identificando as variáveis climáticas com maior "
    "correlação com produtividade, produção e perdas."
)

col1, col2, col3 = st.columns(3)
with col1:
    top_n = st.slider("Número de variáveis mais relevantes:", 5, 20, 10)
with col2:
    # Cria lista de métricas, incluindo as novas colunas corrigidas se disponíveis
    metricas_disponiveis = [
        COL_REND,
        COL_PRODUCAO,
        COL_AREA_PERD,
        COL_PERDA_PCT,
        COL_VALOR,
    ]
    # Adiciona colunas econômicas corrigidas se existirem
    if COL_VALOR_CORRIGIDO in df_filtrado.columns and COL_VALOR_CORRIGIDO != COL_VALOR:
        metricas_disponiveis.insert(5, COL_VALOR_CORRIGIDO)  # Insere após COL_VALOR
    if COL_VALOR_POR_HA in df_filtrado.columns:
        metricas_disponiveis.append(COL_VALOR_POR_HA)        # Adiciona ao final
    
    metrica_foco = st.selectbox(
        "Foco da análise:",
        metricas_disponiveis,
        format_func=label_coluna,
    )
with col3:
    ano_clima_analise = st.selectbox(
        "Ano para análise:",
        options=["Todos os anos"] + [str(a) for a in sorted(anos_selecionados)],
        index=0,
    )

df_para_correlacao = (
    df_filtrado.copy() if ano_clima_analise == "Todos os anos"
    else df_filtrado[df_filtrado['ano'] == int(ano_clima_analise)].copy()
)
titulo_ano = "Todos os Anos" if ano_clima_analise == "Todos os anos" else ano_clima_analise

cache_key_clima = (
    f"{cache_key_filtros}"
    f"|metrica={metrica_foco}"
    f"|ano_analise={ano_clima_analise}"
)

df_corr_foco = calcular_correlacoes_por_ano(
    df_para_correlacao, metrica_foco, cache_key_clima
)
metrica_foco_label = label_coluna(metrica_foco)

if len(df_corr_foco) == 0:
    st.warning("⚠️ Não há dados suficientes para calcular correlações com os filtros selecionados.")
    st.stop()

df_corr_foco = (
    df_corr_foco
    .nlargest(top_n, 'Correlação Abs')
)



st.subheader(f"🔝 Top {len(df_corr_foco)} Variáveis com Maior Impacto – {titulo_ano}")

texto_corr = df_corr_foco.apply(
    lambda row: formatar_correlacao_pvalor(
        row['Correlação'],
        row['P-valor'],
        decimais_corr=3,
        decimais_pvalor=2,
        usar_limite_pvalor=False,
        notacao_cientifica_pvalor=True,
    ),
    axis=1
)
fig_top = go.Figure()
fig_top.add_trace(go.Bar(
    x=df_corr_foco['Correlação'],
    y=[f"{row['Variável Climática']}_dec{row['Decêndio']}_{row['Ano Safra']}"
       for _, row in df_corr_foco.iterrows()],
    orientation='h',
    marker_color=df_corr_foco['Correlação'],
    marker_colorscale='RdYlGn', marker_cmin=-1, marker_cmax=1,
    text=texto_corr, textposition='outside',
))
fig_top.update_layout(
    title=f'<b>Correlação com: {metrica_foco_label} ({titulo_ano})</b>',
    xaxis_title='Correlação de Pearson', yaxis_title='Variável Climática',
    height=max(400, len(df_corr_foco) * 30), xaxis_range=[-1, 1],
    font=dict(color='black'), separators=',.',
)
fig_top.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
fig_top.update_yaxes(
    tickfont=dict(color='black'),
    title_font=dict(color='black'),
    autorange='reversed',
)
fig_top.add_vline(x=0, line_dash="dash", line_color="#000000")
st.plotly_chart(fig_top, width='stretch')

# ── legenda ─────────────────────────────────────────
if desc_atributos:

    atribs_no_top = sorted(
        set(df_corr_foco['Variável Climática'].unique())
    )

    legenda_atribs = "; ".join([
        f"{label_atributo(a)}"
        for a in atribs_no_top
    ])

    st.markdown(
        f"""
        <div style="
            font-size: 0.92rem;
            line-height: 1.55;
            color: #000000;
            text-align: justify;
            font-family: 'Times New Roman', serif;
            padding-top: 0.20rem;
            padding-bottom: 0.15rem;
        ">
            <b>Em que:</b> {legenda_atribs}
        </div>
        """,
        unsafe_allow_html=True
    )

# ── Análise detalhada – Top 3 ─────────────────────────────────────────────────
st.subheader("🔍 Análise Detalhada – Top 3 Variáveis")
st.info(f"🔬 Relação entre as três variáveis climáticas de maior impacto e a produtividade – {titulo_ano}")
n_pontos = len(df_para_correlacao)
st.info(f"📊 Análise baseada em **{formatar_numero(n_pontos)} registros** ({titulo_ano})")

top3 = df_corr_foco.head(3)
for idx, row in top3.iterrows():
    corr_fmt = formatar_correlacao_pvalor(
        row['Correlação'],
        row['P-valor'],
        decimais_corr=4,
        decimais_pvalor=2,
        usar_limite_pvalor=False,
        notacao_cientifica_pvalor=True,
    )

    with st.expander(
        f"**{idx + 1}. {label_atributo(row['Variável Climática'])}"
        f"  ·  dec{row['Decêndio']} {row['Ano Safra']}  | {corr_fmt}**"
    ):
        col1, col2 = st.columns([2, 1])
        with col1:
            cols_scatter = list(dict.fromkeys([
                row['Coluna'], metrica_foco, 'ano', 'municipio', COL_PRODUCAO
            ]))
            df_scatter = df_para_correlacao[cols_scatter].dropna()
            titulo_scatter = (
                f"Dispersão ({metrica_foco_label.split('(')[0].strip()}) "
                f"× {label_atributo(row['Variável Climática'])}"
            )
            try:
                fig_scatter = px.scatter(
                    df_scatter, x=row['Coluna'], y=metrica_foco,
                    color='ano', size=COL_PRODUCAO,
                    hover_data=['municipio'], trendline='ols', title=titulo_scatter,
                )
            except Exception:
                fig_scatter = px.scatter(
                    df_scatter, x=row['Coluna'], y=metrica_foco,
                    color='ano', size=COL_PRODUCAO,
                    hover_data=['municipio'], title=titulo_scatter,
                )
                if len(df_scatter) > 1:
                    slope, intercept, *_ = stats.linregress(
                        df_scatter[row['Coluna']], df_scatter[metrica_foco]
                    )
                    lx = np.array([df_scatter[row['Coluna']].min(),
                                   df_scatter[row['Coluna']].max()])
                    fig_scatter.add_trace(go.Scatter(
                        x=lx, y=slope * lx + intercept,
                        mode='lines', name='Tendência',
                        line=dict(color='red', dash='dash', width=2),
                    ))
            fig_scatter.update_layout(
    title=dict(
        text=titulo_scatter,
        y=0.98
    ),
    margin=dict(l=80, t=100)
)
            fig_scatter.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
            fig_scatter.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
            st.plotly_chart(fig_scatter, width='stretch')
        with col2:
            st.info(f"📖 {label_atributo(row['Variável Climática'])}")
            st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 1.75rem !important;
            }
            </style>
            """, unsafe_allow_html=True)

            with st.container():
                st.markdown("""
                <style>
                div[data-testid="stVerticalBlock"] div[data-testid="stMetricValue"] {
                    font-size: 1.75rem !important;
                }
                </style>
                """, unsafe_allow_html=True)
                st.metric("Correlação", corr_fmt)
            intensidade = (
                "🔴 Forte"    if abs(row['Correlação']) > 0.7 else
                "🟡 Moderada" if abs(row['Correlação']) > 0.4 else
                "🟢 Fraca"
            )
            st.metric("Intensidade", intensidade)
            st.metric("Direção", "📈 Positiva" if row['Correlação'] > 0 else "📉 Negativa")
            st.markdown("**Interpretação:**")
            metrica_curta = metrica_foco_label.split('(')[0].strip()
            atrib_legivel = label_atributo(row['Variável Climática'])
            if row['Correlação'] > 0:
                st.success(
                    f"Aumento de **{atrib_legivel}** "
                    f"associado ao **aumento** de {metrica_curta}."
                )
            else:
                st.warning(
                    f"Aumento de **{atrib_legivel}** "
                    f"associado à **redução** de {metrica_curta}."
                )


# ==========================================
# MAPA DE CALOR – Ciclo Completo da Safra
# ==========================================
st.header(f"🗺️ Mapa de Calor: Ciclo Completo da Safra – {titulo_ano}")
st.info("📅 **Ciclo da Soja:** Ano 1 (Dec 26–36: Set–Dez) → Ano 2 (Dec 1–15: Jan–Mai).")

variaveis_disponiveis = sorted(df_corr_foco['Variável Climática'].unique())
vars_heatmap = st.multiselect(
    "Selecione variáveis climáticas para o mapa de calor:",
    options=variaveis_disponiveis,
    default=variaveis_disponiveis[:min(5, len(variaveis_disponiveis))],
    format_func=desc_clima,   # exibe descrição legível; valor interno continua sendo o código
)

if vars_heatmap:
    decendios_ano1 = list(range(26, 37))
    decendios_ano2 = list(range(1, 16))
    heatmap_data   = []

    for var_clima in vars_heatmap:
        for dec_num in decendios_ano1:
            col_name = f"{var_clima}_dec{dec_num}_ano1"
            if col_name in df_para_correlacao.columns:
                try:
                    df_temp = df_para_correlacao[[col_name, metrica_foco]].dropna()
                    if len(df_temp) > 5:
                        corr, p_valor = calcular_pearson(df_temp, col_name, metrica_foco)
                        if not np.isnan(corr):
                            heatmap_data.append({
                                'Variável':        var_clima,
                                'Período':         f"Ano1_Dec{dec_num}",
                                'Decêndio_Order':  dec_num - 26,
                                'Correlação':      corr,
                                'P-valor':         p_valor,
                            })
                except Exception:
                    pass
        for dec_num in decendios_ano2:
            col_name = f"{var_clima}_dec{dec_num}_ano2"
            if col_name in df_para_correlacao.columns:
                try:
                    df_temp = df_para_correlacao[[col_name, metrica_foco]].dropna()
                    if len(df_temp) > 5:
                        corr, p_valor = calcular_pearson(df_temp, col_name, metrica_foco)
                        if not np.isnan(corr):
                            heatmap_data.append({
                                'Variável':        var_clima,
                                'Período':         f"Ano2_Dec{dec_num}",
                                'Decêndio_Order':  11 + (dec_num - 1),
                                'Correlação':      corr,
                                'P-valor':         p_valor,
                            })
                except Exception:
                    pass

    df_heatmap = pd.DataFrame(heatmap_data)
    if len(df_heatmap) > 0:
        pivot_heatmap = df_heatmap.pivot_table(
            values='Correlação', index='Variável', columns='Período', aggfunc='first'
        )
        colunas_ordenadas = (
            [f"Ano1_Dec{i}" for i in decendios_ano1] +
            [f"Ano2_Dec{i}" for i in decendios_ano2]
        )
        pivot_heatmap = pivot_heatmap[[c for c in colunas_ordenadas if c in pivot_heatmap.columns]]
        # Substitui o índice (código) pela descrição legível do CSV
        pivot_heatmap.index = [
    _base_atributo(v) for v in pivot_heatmap.index
]
        text_heatmap  = pivot_heatmap.map(lambda x: f"{x:.2f}".replace('.', ','))

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_heatmap.values, x=pivot_heatmap.columns, y=pivot_heatmap.index,
            colorscale='RdYlGn', zmid=0,
            text=text_heatmap.values, texttemplate='%{text}',
            textfont={"size": 8},
            colorbar=dict(title="Correlação"), zmin=-1, zmax=1,
        ))
        fig_heatmap.update_layout(
            title=(
                f'<b>Correlação ao longo do Ciclo da Safra: '
                f'{metrica_foco_label.split("(")[0].strip()} ({titulo_ano})</b>'
            ),
            xaxis_title='Período (Ano1: Set–Dez | Ano2: Jan–Mai)',
            yaxis_title='Variável Climática',
            height=max(500, len(pivot_heatmap) * 70),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9, color='black'),
                       title_font=dict(color='black')),
            yaxis=dict(tickfont=dict(color='black'), title_font=dict(color='black')),
            font=dict(color='black'), separators=',.',
        )
        fig_heatmap.add_vline(x=10.5, line_dash="dash", line_color="white", line_width=2)
        st.plotly_chart(fig_heatmap, width='stretch')

        # ── Legenda científica compacta ─────────────────────────────
        variaveis_legenda = sorted(set(
            _base_atributo(v)
            for v in vars_heatmap
        ))

        legenda_txt = "; ".join([
            f"{v} = {desc_clima(v)}"
            for v in variaveis_legenda
        ])

        st.markdown(
            f"""
            <div style="
                font-size: 0.92rem;
                line-height: 1.55;
                color: #000000;
                text-align: justify;
                font-family: 'Times New Roman', serif;
                padding-top: 0.35rem;
                padding-bottom: 0.15rem;
            ">
                <b>Em que: </b> {legenda_txt}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("📊 Correlação Média por Fase da Safra")
        st.info("📋 Correlação média consolidada em cada fase fenológica (calculada sobre os valores do mapa).")
        col1, col2 = st.columns(2)
        with col1:
            ano1_cols = [c for c in pivot_heatmap.columns if c.startswith("Ano1")]
            if ano1_cols:
                st.metric(
                    "Fase 1: Semeadura / Desenvolvimento",
                    formatar_numero(pivot_heatmap[ano1_cols].mean().mean(), decimais=4),
                    help="Decêndios 26–36 do Ano 1 (setembro a dezembro).",
                )
        with col2:
            ano2_cols = [c for c in pivot_heatmap.columns if c.startswith("Ano2")]
            if ano2_cols:
                st.metric(
                    "Fase 2: Maturação / Colheita",
                    formatar_numero(pivot_heatmap[ano2_cols].mean().mean(), decimais=4),
                    help="Decêndios 1–15 do Ano 2 (janeiro a maio).",
                )


# ==========================================
# RANKING DE municipioS
# ==========================================
st.header("🏘️ Evolução dos Top municipios")
st.info("📋 Acompanhamento da evolução anual dos principais municipios produtores de soja no Paraná.")

num_municipios = st.slider("Número de municipios no ranking:", 3, 15, 5)

top_prod_municipios  = df_filtrado.groupby('municipio')[COL_PRODUCAO].mean().nlargest(num_municipios).index
top_produtividade_municipios = df_filtrado.groupby('municipio')[COL_REND].mean().nlargest(num_municipios).index
top_area_municipios  = df_filtrado.groupby('municipio')[COL_AREA_P].mean().nlargest(num_municipios).index
top_valor_municipios = df_filtrado.groupby('municipio')[COL_VALOR].mean().nlargest(num_municipios).index

col1, col2 = st.columns(2)

with col1:
    fig_p = go.Figure()
    for municipio in top_prod_municipios:
        dm = df_filtrado[df_filtrado['municipio'] == municipio].sort_values('ano')
        fig_p.add_trace(go.Scatter(
            x=dm['ano'], y=dm[COL_PRODUCAO],
            mode='lines+markers', name=municipio,
            line=dict(width=2), marker=dict(size=8),
        ))
    fig_p.update_layout(
        title=f'<b>Top {num_municipios} – Evolução da Produção (kg)</b>',
        xaxis_title='ano', yaxis_title='Produção (kg)', height=500,
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        font=dict(color='black'), separators=',.',
    )
    fig_p.update_xaxes(type='linear', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig_p.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig_p, width='stretch')

with col2:
    fig_r = go.Figure()
    for municipio in top_produtividade_municipios:
        dm = df_filtrado[df_filtrado['municipio'] == municipio].sort_values('ano')
        fig_r.add_trace(go.Scatter(
            x=dm['ano'], y=dm[COL_REND],
            mode='lines+markers', name=municipio,
            line=dict(width=2), marker=dict(size=8),
        ))
    fig_r.update_layout(
        title=f'<b>Top {num_municipios} – Evolução da Produtividade (kg/ha)</b>',
        xaxis_title='ano', yaxis_title='Produtividade (kg/ha)', height=500,
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        font=dict(color='black'), separators=',.',
    )
    fig_r.update_xaxes(type='linear', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig_r.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig_r, width='stretch')

col3, col4 = st.columns(2)

with col3:
    fig_a = go.Figure()
    for municipio in top_area_municipios:
        dm = df_filtrado[df_filtrado['municipio'] == municipio].sort_values('ano')
        fig_a.add_trace(go.Scatter(
            x=dm['ano'], y=dm[COL_AREA_P],
            mode='lines+markers', name=municipio,
            line=dict(width=2), marker=dict(size=8),
        ))
    fig_a.update_layout(
        title=f'<b>Top {num_municipios} – Evolução da Área Plantada</b>',
        xaxis_title='ano', yaxis_title='Área Plantada (ha)', height=500,
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        font=dict(color='black'), separators=',.',
    )
    fig_a.update_xaxes(type='linear', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig_a.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig_a, width='stretch')

with col4:
    # ── Prioriza valor corrigido (IPCA 2024) se disponível ────────────────
    col_valor_top = COL_VALOR_CORRIGIDO if (COL_VALOR_CORRIGIDO in df_filtrado.columns) else COL_VALOR
    titulo_valor_top = f'<b>Top {num_municipios} – Valor Corrigido (R$)</b>' if col_valor_top == COL_VALOR_CORRIGIDO else f'<b>Top {num_municipios} – Valor da Produção (R$)</b>'
    
    fig_v = go.Figure()
    for municipio in top_valor_municipios:
        dm = df_filtrado[df_filtrado['municipio'] == municipio].sort_values('ano')
        fig_v.add_trace(go.Scatter(
            x=dm['ano'], y=dm[col_valor_top],
            mode='lines+markers', name=municipio,
            line=dict(width=2), marker=dict(size=8),
        ))
    fig_v.update_layout(
        title=titulo_valor_top,
        xaxis_title='ano', yaxis_title='Valor Corrigido (R$)' if col_valor_top == COL_VALOR_CORRIGIDO else 'Valor da Produção (R$)', height=500,
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        font=dict(color='black'), separators=',.',
    )
    fig_v.update_xaxes(type='linear', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig_v.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    st.plotly_chart(fig_v, width='stretch')


# ==========================================
# RODAPÉ
# ==========================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #683;'>
        🌱 <b>Dashboard Inteligente – Soja Paraná</b> |
        Fonte: PAM/SIDRA + NASA POWER | Desenvolvido por: Bruno Proença de Souza
    </div>
""", unsafe_allow_html=True)
