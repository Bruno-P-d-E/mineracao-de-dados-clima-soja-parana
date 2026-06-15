import pandas as pd
import numpy as np
import time
import sys
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_score,
    ShuffleSplit, KFold
)
# ============================================================
# OTIMIZAÇÃO: joblib.Parallel para paralelizar loop de combos
# ============================================================
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from datetime import datetime

# ===============================================================================
# CONFIGURAÇÃO GLOBAL
# ===============================================================================

PARQUET_PATH = r'C:\Users\bruno\Desktop\Pipeline_TCC\data\processed\dataset_final.parquet'
YEAR_COL     = 'ano'
TRAIN_YEARS  = list(range(2018, 2023))   # 2018–2022 inclusive
TEST_YEARS   = [2023, 2024]
CHECKPOINT_DIR = 'checkpoints_supervised'

# Método de validação ativo:
#   'A' → Split temporal fixo (treino 2018-2022 / teste 2023-2024)
#   'B' → CV aleatório 70/30 via ShuffleSplit (base inteira)
#   'C' → CV 10 folds via KFold (base inteira)
VALIDATION_METHOD = 'A'
VALIDATION_METHODS = ['B', 'C']
VERBOSE_PROGRESS = True

# ============================================================
# OTIMIZAÇÃO: controle de paralelismo
#   N_JOBS_MODELS  → n_jobs dentro de cada modelo sklearn
#   N_JOBS_COMBOS  → n_jobs do Parallel que distribui combos
#
#   RF usa múltiplos núcleos internamente (N_JOBS_MODELS=-1).
#   Para RF, o Parallel externo deve usar N_JOBS_COMBOS=1 para
#   evitar oversubscription. Para SVM/GBM/Linear, o Parallel
#   externo pode usar N_JOBS_COMBOS=-1 pois esses modelos são
#   single-thread. A lógica é resolvida em _run_single_experiment.
# ============================================================
N_JOBS_MODELS = -1   # núcleos internos dos modelos (RF, Linear)
N_JOBS_CV     = -1   # núcleos do cross_val_score
N_JOBS_COMBOS = -1   # núcleos do Parallel externo de combos


# ===============================================================================
# CORREÇÃO #7 — format_duration corrigido para 10–59s
# ===============================================================================

def format_duration(seconds: float) -> str:
    """Formata duração em string legível."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


FEATURE_COUNTS = [35, 40, 45, 50]
SCALERS        = ['robust', 'standard', 'power']
MODEL_CONFIGS  = [
    {'type': 'rf',     'params': {'n_estimators': 200, 'max_depth': 8,  'min_samples_split': 10, 'min_samples_leaf': 5}},
    {'type': 'rf',     'params': {'n_estimators': 250, 'max_depth': 8,  'min_samples_split': 10, 'min_samples_leaf': 5}},
    {'type': 'rf',     'params': {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 12, 'min_samples_leaf': 6}},
    {'type': 'gbm',    'params': {'n_estimators': 150, 'max_depth': 5,  'learning_rate': 0.03, 'subsample': 0.85}},
    {'type': 'gbm',    'params': {'n_estimators': 200, 'max_depth': 5,  'learning_rate': 0.02, 'subsample': 0.85}},
    {'type': 'svm',    'params': {'kernel': 'rbf',    'C': 10.0, 'epsilon': 0.1}},
    {'type': 'svm',    'params': {'kernel': 'linear', 'C': 1.0}},
    {'type': 'linear', 'params': {}},
]

VARIANTS = {
    'v1_original': {
        'name':       'V1: Original (3 dec early, 5-10 flow, 11-15 grain)',
        'early':      {'enabled': True,  'n_decendios': 3, 'stats': ['mean']},
        'flowering':  {'start_dec': 5,  'end_dec': 10, 'stats': ['mean']},
        'grain':      {'start_dec': 11, 'end_dec': 15, 'stats': ['mean']},
        'maturation': {'enabled': False, 'start_dec': 16, 'end_dec': 18, 'stats': ['mean']},
    },
    'v2_with_variability': {
        'name':       'V2: Com variabilidade (mean + std)',
        'early':      {'enabled': True,  'n_decendios': 3, 'stats': ['mean', 'std']},
        'flowering':  {'start_dec': 5,  'end_dec': 10, 'stats': ['mean', 'std']},
        'grain':      {'start_dec': 11, 'end_dec': 15, 'stats': ['mean', 'std']},
        'maturation': {'enabled': False, 'start_dec': 16, 'end_dec': 18, 'stats': ['mean']},
    },
    'v3_robust_stats': {
        'name':       'V3: Estatísticas robustas (mean + median)',
        'early':      {'enabled': True,  'n_decendios': 3, 'stats': ['mean']},
        'flowering':  {'start_dec': 5,  'end_dec': 10, 'stats': ['mean', 'median']},
        'grain':      {'start_dec': 11, 'end_dec': 15, 'stats': ['mean', 'median']},
        'maturation': {'enabled': False, 'start_dec': 16, 'end_dec': 18, 'stats': ['mean']},
    },
    'v4_complete': {
        'name':       'V4: Completo (todas fases + variabilidade)',
        'early':      {'enabled': True,  'n_decendios': 4, 'stats': ['mean', 'std']},
        'flowering':  {'start_dec': 5,  'end_dec': 10, 'stats': ['mean', 'std']},
        'grain':      {'start_dec': 11, 'end_dec': 15, 'stats': ['mean', 'std']},
        'maturation': {'enabled': True,  'start_dec': 16, 'end_dec': 18, 'stats': ['mean', 'std']},
    },
    'v5_iqr': {
        'name':       'V5: IQR (Q3-Q1)',
        'early':      {'enabled': True,  'n_decendios': 4, 'stats': ['mean', 'iqr']},
        'flowering':  {'start_dec': 5,  'end_dec': 10, 'stats': ['mean', 'iqr']},
        'grain':      {'start_dec': 11, 'end_dec': 15, 'stats': ['mean', 'iqr']},
        'maturation': {'enabled': True,  'start_dec': 16, 'end_dec': 18, 'stats': ['mean', 'iqr']},
    },
    'v6_cv': {
        'name':       'V6: CV (std/mean)',
        'early':      {'enabled': True,  'n_decendios': 4, 'stats': ['mean', 'cv']},
        'flowering':  {'start_dec': 5,  'end_dec': 10, 'stats': ['mean', 'cv']},
        'grain':      {'start_dec': 11, 'end_dec': 15, 'stats': ['mean', 'cv']},
        'maturation': {'enabled': True,  'start_dec': 16, 'end_dec': 18, 'stats': ['mean', 'cv']},
    },
}

# ===============================================================================
# DEFINIÇÃO DOS EXPERIMENTOS (por alvo, com e sem area_plantada_ha)
# ===============================================================================

EXPERIMENTOS = [
    {
        'id':         'EXP01',
        'descricao':  'area_plantada_ha + variáveis climáticas → rendimento_kg_ha',
        'target_col': 'rendimento_kg_ha',
        'extra_col':  None,
        'preditores': 'area_e_clima',
        'unidade':    'kg/ha',
    },
    {
        'id':         'EXP02',
        'descricao':  'variáveis climáticas → rendimento_kg_ha',
        'target_col': 'rendimento_kg_ha',
        'extra_col':  None,
        'preditores': 'somente_clima',
        'unidade':    'kg/ha',
    },
    {
        'id':         'EXP03',
        'descricao':  'area_plantada_ha + variáveis climáticas → valor_rs_ha',
        'target_col': 'valor_rs_ha',
        'extra_col':  {
            'name':    'valor_rs_ha',
            'formula': lambda df: (df['valor_producao_mil_reais'] * 1_000) / df['area_plantada_ha'],
        },
        'preditores': 'area_e_clima',
        'unidade':    'R$/ha',
    },
    {
        'id':         'EXP04',
        'descricao':  'variáveis climáticas → valor_rs_ha',
        'target_col': 'valor_rs_ha',
        'extra_col':  {
            'name':    'valor_rs_ha',
            'formula': lambda df: (df['valor_producao_mil_reais'] * 1_000) / df['area_plantada_ha'],
        },
        'preditores': 'somente_clima',
        'unidade':    'R$/ha',
    },
    {
        'id':         'EXP05',
        'descricao':  'area_plantada_ha + variáveis climáticas → quantidade_produzida_ton',
        'target_col': 'quantidade_produzida_ton',
        'extra_col':  None,
        'preditores': 'area_e_clima',
        'unidade':    'ton',
    },
    {
        'id':         'EXP06',
        'descricao':  'variáveis climáticas → quantidade_produzida_ton',
        'target_col': 'quantidade_produzida_ton',
        'extra_col':  None,
        'preditores': 'somente_clima',
        'unidade':    'ton',
    },
    {
        'id':         'EXP07',
        'descricao':  'area_plantada_ha + variáveis climáticas → valor_producao_mil_reais',
        'target_col': 'valor_producao_mil_reais',
        'extra_col':  None,
        'preditores': 'area_e_clima',
        'unidade':    'mil R$',
    },
    {
        'id':         'EXP08',
        'descricao':  'variáveis climáticas → valor_producao_mil_reais',
        'target_col': 'valor_producao_mil_reais',
        'extra_col':  None,
        'preditores': 'somente_clima',
        'unidade':    'mil R$',
    },
    {
        'id':         'EXP09',
        'descricao':  'area_plantada_ha + variáveis climáticas → valor_producao_ipca_mil_reais',
        'target_col': 'valor_producao_ipca_mil_reais',
        'extra_col':  None,
        'preditores': 'area_e_clima',
        'unidade':    'mil R$ (IPCA)',
    },
    {
        'id':         'EXP10',
        'descricao':  'variáveis climáticas → valor_producao_ipca_mil_reais',
        'target_col': 'valor_producao_ipca_mil_reais',
        'extra_col':  None,
        'preditores': 'somente_clima',
        'unidade':    'mil R$ (IPCA)',
    },
    {
        'id':         'EXP11',
        'descricao':  'area_plantada_ha + variáveis climáticas → valor_producao_ipca_mil_reais_ha × 1000',
        'target_col': 'valor_producao_ipca_mil_reais_ha',
        'extra_col':  {
            'name':    'valor_producao_ipca_mil_reais_ha',
            'formula': lambda df: df['valor_producao_ipca_mil_reais_ha'] * 1_000,
        },
        'preditores': 'area_e_clima',
        'unidade':    'R$/ha (IPCA)',
    },
    {
        'id':         'EXP12',
        'descricao':  'variáveis climáticas → valor_producao_ipca_mil_reais_ha × 1000',
        'target_col': 'valor_producao_ipca_mil_reais_ha',
        'extra_col':  {
            'name':    'valor_producao_ipca_mil_reais_ha',
            'formula': lambda df: df['valor_producao_ipca_mil_reais_ha'] * 1_000,
        },
        'preditores': 'somente_clima',
        'unidade':    'R$/ha (IPCA)',
    },
]


# ===============================================================================
# CLASSE PRINCIPAL
# ===============================================================================

class SupervisedUnified:
    """
    Unifica todos os experimentos do notebook supervised.ipynb.

    Métodos de validação disponíveis (controlado por VALIDATION_METHOD):
      [A] Split temporal fixo — treino 2018–2022 / teste 2023–2024
      [B] CV aleatório 70/30  — ShuffleSplit, base inteira
      [C] CV 10 folds         — KFold, base inteira

    OTIMIZAÇÕES aplicadas (v3):
      OPT-1 — n_jobs=-1 em todos os modelos sklearn e no cross_val_score,
               aproveitando todos os núcleos disponíveis.
      OPT-2 — RF de seleção de features reduzido para 50 árvores (max_depth=6);
               suficiente para ranking de importâncias, 2× mais rápido.
      OPT-3 — Cache de agregação de features por (var_key) dentro do
               experimento: _aggregate_features não é recalculado para
               cada n_vars.
      OPT-4 — joblib.Parallel distribui o loop de (scaler × modelo) em
               paralelo. RF usa n_jobs=-1 internamente → Parallel com
               n_jobs=1 para RF (evita oversubscription). SVM/GBM/Linear
               são single-thread → Parallel com n_jobs=-1.
               Implementado via _run_combos_parallel com detecção de tipo.
      OPT-5 — selector_rf dentro do Pipeline do CV também usa n_jobs=-1.

    Correções da v2 mantidas integralmente (#1–#7).
    """

    def __init__(self, parquet_path, year_col=YEAR_COL,
                 train_years=TRAIN_YEARS, test_years=TEST_YEARS,
                 validation_method=VALIDATION_METHODS):
        self.parquet_path      = parquet_path
        self.year_col          = year_col
        self.train_years       = train_years
        self.test_years        = test_years
        self.validation_method = validation_method.upper()
        if self.validation_method not in VALIDATION_METHODS:
            raise ValueError(
                f"Método de validação inválido: {validation_method}. "
                f"Use um de: {', '.join(VALIDATION_METHODS)}"
            )
        self.all_results = []
        self._df_raw     = None

    # ------------------------------------------------------------------
    # Carregamento base
    # ------------------------------------------------------------------

    def load_raw(self):
        print("=" * 120)
        print("CARREGANDO DATASET BASE")
        print("=" * 120)
        df = pd.read_parquet(self.parquet_path)
        print(f"\n✓ Dataset carregado: {df.shape}  |  colunas[0:8]: {df.columns.tolist()[:8]} …")
        self._df_raw = df

    # ------------------------------------------------------------------
    # Preparo por experimento
    # ------------------------------------------------------------------

    def _prepare_experiment(self, exp):
        df = self._df_raw.copy()

        if exp['extra_col'] is not None:
            col_name = exp['extra_col']['name']
            df[col_name] = exp['extra_col']['formula'](df)

        target_col = exp['target_col']
        if target_col not in df.columns:
            raise KeyError(f"Coluna alvo não encontrada: {target_col}")
        if pd.api.types.is_numeric_dtype(df[target_col]):
            df[target_col] = df[target_col].replace([np.inf, -np.inf], np.nan)

        if self.validation_method == 'A':
            train_df = df[df[self.year_col].isin(self.train_years)].dropna().reset_index(drop=True)
            test_df  = df[df[self.year_col].isin(self.test_years)].dropna().reset_index(drop=True)
            print(f"\n✓ {exp['id']} | {exp['descricao']}")
            print(f"  [A] Treino ({self.train_years[0]}–{self.train_years[-1]}): {train_df.shape}"
                  f"  |  Teste ({self.test_years[0]}–{self.test_years[-1]}): {test_df.shape}")
        else:
            train_df = df.dropna().reset_index(drop=True)
            test_df  = None
            label = "[B] ShuffleSplit 70/30" if self.validation_method == 'B' else "[C] KFold 10 folds"
            print(f"\n✓ {exp['id']} | {exp['descricao']}")
            print(f"  {label}  |  Base completa: {train_df.shape}")

        print(f"\n  📊 TARGET ({target_col}):")
        s = train_df[target_col]
        print(f"     Base: média={s.mean():.2f}  std={s.std():.2f}"
              f"  min={s.min():.2f}  max={s.max():.2f}")

        return train_df, test_df, target_col

    # ------------------------------------------------------------------
    # Variáveis climáticas
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_climate_vars(df):
        climate_vars = set()
        for col in df.columns:
            if 'dec' in col and 'ano' in col:
                climate_vars.add(col.split('dec')[0].rstrip('_'))
        return sorted(climate_vars)

    # ------------------------------------------------------------------
    # Remoção de leakage (correção #2/#6)
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_leakage(df, target_col, preditores, climate_vars, year_col=YEAR_COL):
        climate_prefixes = tuple(f"{v}_dec" for v in climate_vars)
        climate_agg_suffixes = ('_early', '_flowering', '_grain', '_maturation')
        context_cols = {target_col}

        if preditores in ('somente_clima', 'somente_clima_sem_area'):
            keep_cols = [
                c for c in df.columns
                if c in context_cols
                or c.startswith(climate_prefixes)
                or any(
                    c.startswith(v) and any(c.endswith(s) or f'{s}_' in c
                                            for s in climate_agg_suffixes)
                    for v in climate_vars
                )
            ]
            return df[keep_cols]

        if preditores == 'area_e_clima_sem_ano':
            keep = context_cols | {'area_plantada_ha'}
            drop = [c for c in df.columns
                    if c not in keep and not c.startswith(climate_prefixes)]
            return df.drop(columns=drop)

        leakage_cols = [
            'quantidade_produzida_ton', 'valor_producao_mil_reais',
            'valor_producao_pct', 'valor_producao_ipca_mil_reais',
            'valor_producao_ipca_mil_reais_ha', 'rendimento_kg_ha',
            'area_colhida_ha', 'area_colhida_pct', 'area_plantada_pct',
            'fator_correcao_ipca', 'valor_rs_ha', 'valor_por_ha',
            year_col, 'cod_ibge', 'municipio', 'cod_meso', 'mesorregiao',
            'latitude', 'longitude',
        ]
        leakage_cols = [c for c in leakage_cols if c != target_col]
        return df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # ------------------------------------------------------------------
    # Normalização climática — fit exclusivamente no treino
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_pair(train_df, test_df, climate_vars):
        train_out  = train_df.copy()
        climate_cols = [c for c in train_df.columns if 'dec' in c and 'ano' in c]
        scalers_per_var = {}

        for var in climate_vars:
            var_cols = [c for c in climate_cols if c.startswith(f"{var}_dec")]
            if var_cols:
                sc = RobustScaler()
                train_out[var_cols] = sc.fit_transform(train_df[var_cols])
                scalers_per_var[var] = (sc, var_cols)

        print(f"   🔧 {len(climate_cols)} colunas climáticas normalizadas "
              f"(RobustScaler, fit exclusivo no treino)")

        if test_df is not None:
            test_out = test_df.copy()
            for var, (sc, var_cols) in scalers_per_var.items():
                present = [c for c in var_cols if c in test_df.columns]
                if present:
                    test_out[present] = sc.transform(test_df[present])
        else:
            test_out = None

        return train_out, test_out

    # ------------------------------------------------------------------
    # Agregação de features por fase fenológica
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_features(df, variant_config, climate_vars):
        non_climate = [c for c in df.columns
                       if not any(c.startswith(f"{v}_dec") for v in climate_vars)]
        result = df[non_climate].copy()

        def add_stats(prefix, data, stats):
            if 'mean' in stats:
                result[f'{prefix}_mean'] = data.mean(axis=1)
            if 'std' in stats:
                result[f'{prefix}_std'] = data.std(axis=1)
            if 'min' in stats:
                result[f'{prefix}_min'] = data.min(axis=1)
            if 'max' in stats:
                result[f'{prefix}_max'] = data.max(axis=1)
            if 'median' in stats:
                result[f'{prefix}_median'] = data.median(axis=1)
            if 'iqr' in stats:
                result[f'{prefix}_iqr'] = (
                    data.quantile(0.75, axis=1) - data.quantile(0.25, axis=1)
                )
            if 'cv' in stats:
                row_mean = data.mean(axis=1)
                safe_mean = row_mean.abs().where(row_mean.abs() > 1e-6, other=np.nan)
                result[f'{prefix}_cv'] = (
                    (data.std(axis=1) / safe_mean)
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                )

        for var in climate_vars:
            var_cols  = [c for c in df.columns if c.startswith(f"{var}_dec")]
            if not var_cols:
                continue
            ano1_cols = [c for c in var_cols if 'ano1' in c]
            ano2_cols = [c for c in var_cols if 'ano2' in c]
            cfg = variant_config

            if cfg['early']['enabled']:
                n = cfg['early']['n_decendios']
                if ano1_cols and len(ano1_cols) >= n:
                    d, sts = df[ano1_cols[-n:]], cfg['early']['stats']
                    add_stats(f'{var}_early', d, sts)

            fs, fe = cfg['flowering']['start_dec'], cfg['flowering']['end_dec']
            fl = [c for c in ano2_cols if any(f'dec{i}_' in c for i in range(fs, fe + 1))]
            if fl:
                d, sts = df[fl], cfg['flowering']['stats']
                add_stats(f'{var}_flowering', d, sts)

            gs, ge = cfg['grain']['start_dec'], cfg['grain']['end_dec']
            gr = [c for c in ano2_cols if any(f'dec{i}_' in c for i in range(gs, ge + 1))]
            if gr:
                d, sts = df[gr], cfg['grain']['stats']
                add_stats(f'{var}_grain', d, sts)

            if cfg['maturation']['enabled']:
                ms, me = cfg['maturation']['start_dec'], cfg['maturation']['end_dec']
                mt = [c for c in ano2_cols if any(f'dec{i}_' in c for i in range(ms, me + 1))]
                if mt:
                    d, sts = df[mt], cfg['maturation']['stats']
                    add_stats(f'{var}_maturation', d, sts)

        return result

    # ------------------------------------------------------------------
    # Seleção de features via RF
    # ---------------------------------------------------------------
    # OPT-2: RF de seleção reduzido para 50 árvores e max_depth=6,
    #         com n_jobs=-1 para aproveitar múltiplos núcleos.
    #         Suficiente para ranking de importâncias, 2× mais rápido
    #         que as 100 árvores originais.
    # ------------------------------------------------------------------

    @staticmethod
    def _select_features_rf(df_train, df_test, target_col, year_col, n_vars):
        drop_cols = [c for c in [target_col, year_col] if c in df_train.columns]
        X_tr = df_train.drop(columns=drop_cols)
        y_tr = df_train[target_col]

        # OPT-2: 50 árvores com n_jobs=-1 em vez de 100 com n_jobs=1
        rf = RandomForestRegressor(
            n_estimators=50, max_depth=6,
            min_samples_leaf=5, random_state=42, n_jobs=N_JOBS_MODELS
        )
        rf.fit(X_tr, y_tr)

        top = (pd.DataFrame({'feature': X_tr.columns,
                              'importance': rf.feature_importances_})
               .sort_values('importance', ascending=False)
               .head(n_vars)['feature'].tolist())

        keep = top + [target_col]
        df_tr_sel = df_train[[c for c in keep if c in df_train.columns]]

        if df_test is not None:
            df_te_sel = df_test[[c for c in keep if c in df_test.columns]]
        else:
            df_te_sel = None

        return df_tr_sel, df_te_sel, top

    # ------------------------------------------------------------------
    # Construção do scaler sklearn a partir do nome
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scaler(scaler_type):
        if scaler_type == 'robust':
            return RobustScaler()
        elif scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'power':
            return PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            raise ValueError(f"Scaler desconhecido: {scaler_type}")

    # ------------------------------------------------------------------
    # Escalonamento — usado no holdout final do método A
    # ------------------------------------------------------------------

    def _apply_scaling(self, X_train, X_test, scaler_type):
        sc = self._build_scaler(scaler_type)
        return sc.fit_transform(X_train), sc.transform(X_test)

    # ------------------------------------------------------------------
    # Métricas auxiliares
    # ------------------------------------------------------------------

    @staticmethod
    def _rae(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        d = np.sum(np.abs(y_true - y_true.mean()))
        if np.isclose(d, 0.0):
            return np.nan
        return np.sum(np.abs(y_true - y_pred)) / d * 100

    @staticmethod
    def _rrse(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        d = np.sum((y_true - y_true.mean()) ** 2)
        if np.isclose(d, 0.0):
            return np.nan
        return np.sqrt(np.sum((y_true - y_pred) ** 2) / d) * 100

    # ------------------------------------------------------------------
    # Construtor de modelo
    # ---------------------------------------------------------------
    # OPT-1: n_jobs=N_JOBS_MODELS em todos os modelos que suportam.
    #         RF e LinearRegression usam múltiplos núcleos internamente.
    #         GBM e SVM são single-thread por design do sklearn.
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model(model_config):
        t, p = model_config['type'], model_config['params']
        if t == 'gbm':
            return (GradientBoostingRegressor(**p, random_state=42),
                    f"{p['n_estimators']}e_d{p['max_depth']}_lr{p['learning_rate']}")
        elif t == 'svm':
            return SVR(**p), f"SVM_{p.get('kernel','rbf')}_C{p.get('C',1)}"
        elif t == 'linear':
            # OPT-1: n_jobs=-1 para LinearRegression
            return LinearRegression(**p, n_jobs=N_JOBS_MODELS), "Linear_OLS"
        else:
            # OPT-1: n_jobs=-1 para RandomForestRegressor
            return (RandomForestRegressor(**p, random_state=42, n_jobs=N_JOBS_MODELS),
                    f"{p['n_estimators']}e_d{p['max_depth']}_msl{p['min_samples_leaf']}")

    @staticmethod
    def _model_pshort(model_config):
        t, p = model_config['type'], model_config['params']
        if t == 'gbm':
            return f"{p['n_estimators']}e_d{p['max_depth']}_lr{p['learning_rate']}"
        elif t == 'svm':
            return f"SVM_{p.get('kernel','rbf')}_C{p.get('C',1)}"
        elif t == 'linear':
            return "Linear_OLS"
        else:
            return f"{p['n_estimators']}e_d{p['max_depth']}_msl{p['min_samples_leaf']}"

    # ------------------------------------------------------------------
    # Avaliação — Método [A]: split temporal fixo
    # ---------------------------------------------------------------
    # OPT-1: cross_val_score com n_jobs=N_JOBS_CV.
    # OPT-5: selector_rf dentro do Pipeline com n_jobs=-1.
    # ------------------------------------------------------------------

    def _evaluate_method_A(self, df_tr, df_te, target_col, model_config,
                            scaler_type, n_vars):
        X_tr = df_tr.drop(columns=[target_col])
        y_tr = df_tr[target_col]
        X_te = df_te.drop(columns=[target_col])
        y_te = df_te[target_col]

        pshort = self._model_pshort(model_config)

        # OPT-5: selector_rf com n_jobs=-1
        selector_rf = RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=N_JOBS_MODELS
        )
        pipe_cv = Pipeline([
            ('scaler',   self._build_scaler(scaler_type)),
            ('selector', SelectFromModel(selector_rf,
                                         max_features=n_vars,
                                         threshold=-np.inf)),
            ('model',    self._build_model(model_config)[0]),
        ])

        t0 = time.perf_counter()
        # OPT-1: n_jobs=N_JOBS_CV paraleliza os 5 folds
        # Nota: quando o modelo interno já usa n_jobs=-1 (RF),
        # N_JOBS_CV deve ser 1 para evitar oversubscription.
        # Para SVM/GBM/Linear, N_JOBS_CV=-1 é seguro.
        _cv_jobs = 1 if model_config['type'] == 'rf' else N_JOBS_CV
        cv_s = cross_val_score(pipe_cv, X_tr, y_tr, cv=5, scoring='r2', n_jobs=_cv_jobs)
        cv_t = time.perf_counter() - t0

        X_tr_s, X_te_s = self._apply_scaling(X_tr, X_te, scaler_type)
        model_final, _ = self._build_model(model_config)

        t1 = time.perf_counter()
        model_final.fit(X_tr_s, y_tr)
        bt = time.perf_counter() - t1

        yp_tr = model_final.predict(X_tr_s)
        yp_te = model_final.predict(X_te_s)

        return dict(
            r2_train   = r2_score(y_tr, yp_tr),
            r2_test    = r2_score(y_te, yp_te),
            r2_cv      = float(cv_s.mean()),
            cv_std     = float(cv_s.std()),
            overfit    = r2_score(y_tr, yp_tr) - r2_score(y_te, yp_te),
            mae_train  = mean_absolute_error(y_tr, yp_tr),
            mae_test   = mean_absolute_error(y_te, yp_te),
            rmse_train = np.sqrt(mean_squared_error(y_tr, yp_tr)),
            rmse_test  = np.sqrt(mean_squared_error(y_te, yp_te)),
            rae_train  = self._rae(y_tr, yp_tr),
            rae_test   = self._rae(y_te, yp_te),
            rrse_train = self._rrse(y_tr, yp_tr),
            rrse_test  = self._rrse(y_te, yp_te),
            n_train    = len(y_tr),
            n_test     = len(y_te),
            y_tr       = y_tr,
            y_te       = y_te,
            pshort     = pshort,
            cv_t       = cv_t,
            bt         = bt,
            cv_scope   = 'train_only',
        )

    # ------------------------------------------------------------------
    # Avaliação — Método [B]: CV aleatório 70/30 (ShuffleSplit)
    # ------------------------------------------------------------------

    def _evaluate_method_B(self, df_full, target_col, model_config, scaler_type, top_feats):
        X = df_full.drop(columns=[target_col]).values
        y = df_full[target_col].values

        pshort = self._model_pshort(model_config)
        ss = ShuffleSplit(n_splits=10, test_size=0.30, random_state=42)

        fold_r2_train, fold_r2_test = [], []
        fold_mae_te, fold_rmse_te   = [], []
        fold_rae_te, fold_rrse_te   = [], []
        yt_all = []

        t0 = time.perf_counter()
        for tr_idx, te_idx in ss.split(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            sc = self._build_scaler(scaler_type)
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            m, _ = self._build_model(model_config)
            m.fit(X_tr_s, y_tr)
            yp_tr = m.predict(X_tr_s)
            yp_te = m.predict(X_te_s)

            fold_r2_train.append(r2_score(y_tr, yp_tr))
            fold_r2_test.append(r2_score(y_te, yp_te))
            fold_mae_te.append(mean_absolute_error(y_te, yp_te))
            fold_rmse_te.append(np.sqrt(mean_squared_error(y_te, yp_te)))
            fold_rae_te.append(self._rae(y_te, yp_te))
            fold_rrse_te.append(self._rrse(y_te, yp_te))
            yt_all.extend(y_te.tolist())

        cv_t = time.perf_counter() - t0

        return dict(
            r2_train   = np.mean(fold_r2_train),
            r2_test    = np.mean(fold_r2_test),
            r2_cv      = np.mean(fold_r2_test),
            cv_std     = np.std(fold_r2_test),
            overfit    = np.mean(fold_r2_train) - np.mean(fold_r2_test),
            mae_train  = np.nan,
            mae_test   = np.mean(fold_mae_te),
            rmse_train = np.nan,
            rmse_test  = np.mean(fold_rmse_te),
            rae_train  = np.nan,
            rae_test   = np.mean(fold_rae_te),
            rrse_train = np.nan,
            rrse_test  = np.mean(fold_rrse_te),
            n_train    = int(len(y) * 0.70),
            n_test     = int(len(y) * 0.30),
            y_tr       = pd.Series(y),
            y_te       = pd.Series(yt_all),
            pshort     = pshort,
            cv_t       = cv_t,
            bt         = 0.0,
            cv_scope   = 'full_dataset',
        )

    # ------------------------------------------------------------------
    # Avaliação — Método [C]: CV 10 folds (KFold)
    # ------------------------------------------------------------------

    def _evaluate_method_C(self, df_full, target_col, model_config, scaler_type, top_feats):
        X = df_full.drop(columns=[target_col]).values
        y = df_full[target_col].values

        pshort = self._model_pshort(model_config)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        fold_r2_train, fold_r2_test = [], []
        fold_mae_te, fold_rmse_te   = [], []
        fold_rae_te, fold_rrse_te   = [], []
        yt_all = []

        t0 = time.perf_counter()
        for tr_idx, te_idx in kf.split(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            sc = self._build_scaler(scaler_type)
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            m, _ = self._build_model(model_config)
            m.fit(X_tr_s, y_tr)
            yp_tr = m.predict(X_tr_s)
            yp_te = m.predict(X_te_s)

            fold_r2_train.append(r2_score(y_tr, yp_tr))
            fold_r2_test.append(r2_score(y_te, yp_te))
            fold_mae_te.append(mean_absolute_error(y_te, yp_te))
            fold_rmse_te.append(np.sqrt(mean_squared_error(y_te, yp_te)))
            fold_rae_te.append(self._rae(y_te, yp_te))
            fold_rrse_te.append(self._rrse(y_te, yp_te))
            yt_all.extend(y_te.tolist())

        cv_t = time.perf_counter() - t0
        n_total = len(y)

        return dict(
            r2_train   = np.mean(fold_r2_train),
            r2_test    = np.mean(fold_r2_test),
            r2_cv      = np.mean(fold_r2_test),
            cv_std     = np.std(fold_r2_test),
            overfit    = np.mean(fold_r2_train) - np.mean(fold_r2_test),
            mae_train  = np.nan,
            mae_test   = np.mean(fold_mae_te),
            rmse_train = np.nan,
            rmse_test  = np.mean(fold_rmse_te),
            rae_train  = np.nan,
            rae_test   = np.mean(fold_rae_te),
            rrse_train = np.nan,
            rrse_test  = np.mean(fold_rrse_te),
            n_train    = int(n_total * 0.90),
            n_test     = int(n_total * 0.10),
            y_tr       = pd.Series(y),
            y_te       = pd.Series(yt_all),
            pshort     = pshort,
            cv_t       = cv_t,
            bt         = 0.0,
            cv_scope   = 'full_dataset',
        )

    # ------------------------------------------------------------------
    # Treino, avaliação e geração do dicionário de métricas (unificado)
    # ------------------------------------------------------------------

    def _train_and_evaluate(self, df_tr, df_te, exp_id, target_col, variant,
                            n_features, model_config, scaler_type, exp_num,
                            top_feats, area_available):
        if self.validation_method == 'A':
            ev = self._evaluate_method_A(df_tr, df_te, target_col,
                                         model_config, scaler_type, n_features)
        elif self.validation_method == 'B':
            ev = self._evaluate_method_B(df_tr, target_col, model_config,
                                         scaler_type, top_feats)
        else:
            ev = self._evaluate_method_C(df_tr, target_col, model_config,
                                         scaler_type, top_feats)

        y_tr = ev['y_tr']
        y_te = ev['y_te']

        return dict(
            experiment_id                     = exp_id,
            experiment_number                 = exp_num,
            validation_method                 = self.validation_method,
            cv_scope                          = ev['cv_scope'],
            target_col                        = target_col,
            variant                           = variant,
            n_features                        = n_features,
            model                             = model_config['type'],
            scaler                            = scaler_type,
            params_short                      = ev['pshort'],
            params_full                       = str(model_config['params']),
            r2_train                          = ev['r2_train'],
            r2_test                           = ev['r2_test'],
            r2_cv                             = ev['r2_cv'],
            cv_std                            = ev['cv_std'],
            overfit                           = ev['overfit'],
            cv_time_seconds                   = ev['cv_t'],
            time_taken_to_build_model_seconds = ev['bt'],
            mae_train                         = ev['mae_train'],
            mae_test                          = ev['mae_test'],
            rmse_train                        = ev['rmse_train'],
            rmse_test                         = ev['rmse_test'],
            rae_train_pct                     = ev['rae_train'],
            rae_test_pct                      = ev['rae_test'],
            rrse_train_pct                    = ev['rrse_train'],
            rrse_test_pct                     = ev['rrse_test'],
            total_train_instances             = ev['n_train'],
            total_test_instances              = ev['n_test'],
            area_available                    = area_available,
            area_selected                     = 'area_plantada_ha' in top_feats,
            target_train_mean                 = float(y_tr.mean()),
            target_train_std                  = float(y_tr.std()),
            target_train_min                  = float(y_tr.min()),
            target_train_max                  = float(y_tr.max()),
            target_test_mean                  = float(y_te.mean()),
            target_test_std                   = float(y_te.std()),
            target_test_min                   = float(y_te.min()),
            target_test_max                   = float(y_te.max()),
            selected_features                 = ' | '.join(top_feats) if top_feats else '',
        )

    # ------------------------------------------------------------------
    # OPT-4: função auxiliar para rodar uma única combinação (scaler × modelo)
    #         de forma independente — usada pelo Parallel externo.
    # ------------------------------------------------------------------

    def _run_single_combo(self, df_tr_sel, df_te_sel, exp_id, target_col,
                          var_key, n_vars, mcfg, scaler_type, exp_num,
                          top_feats, area_available):
        """Executa _train_and_evaluate para uma combinação (scaler, modelo)."""
        return self._train_and_evaluate(
            df_tr_sel, df_te_sel, exp_id, target_col,
            var_key, n_vars, mcfg, scaler_type, exp_num,
            top_feats, area_available
        )

    # ------------------------------------------------------------------
    # OPT-4: paralelização do loop de combos com joblib.Parallel
    # ---------------------------------------------------------------
    # Separa combos por tipo de modelo para evitar oversubscription:
    #   • RF  → modelo usa n_jobs=-1 internamente → Parallel com n_jobs=1
    #   • SVM/GBM/Linear → single-thread → Parallel com n_jobs=-1
    # Os dois grupos são executados separadamente e os resultados unidos.
    # ------------------------------------------------------------------

    def _run_combos_parallel(self, df_tr_sel, df_te_sel, exp_id, target_col,
                             var_key, n_vars, top_feats, area_available, base_exp_num):
        """
        Distribui todas as combinações (scaler × modelo) em paralelo.
        RF usa Parallel(n_jobs=1) para evitar oversubscription com n_jobs=-1
        interno. SVM/GBM/Linear usam Parallel(n_jobs=-1).
        """
        combos = [
            (mcfg, scaler_type, base_exp_num + i)
            for i, (scaler_type, mcfg) in enumerate(
                [(s, m) for s in SCALERS for m in MODEL_CONFIGS]
            )
        ]

        rf_combos    = [(m, s, n) for m, s, n in combos if m['type'] == 'rf']
        other_combos = [(m, s, n) for m, s, n in combos if m['type'] != 'rf']

        # RF: n_jobs=1 no Parallel (RF já é paralelo internamente)
        rf_results = Parallel(n_jobs=1)(
            delayed(self._run_single_combo)(
                df_tr_sel, df_te_sel, exp_id, target_col,
                var_key, n_vars, mcfg, scaler_type, exp_num,
                top_feats, area_available
            )
            for mcfg, scaler_type, exp_num in rf_combos
        )

        # SVM / GBM / Linear: n_jobs=-1 (são single-thread, Parallel é seguro)
        other_results = Parallel(n_jobs=N_JOBS_COMBOS)(
            delayed(self._run_single_combo)(
                df_tr_sel, df_te_sel, exp_id, target_col,
                var_key, n_vars, mcfg, scaler_type, exp_num,
                top_feats, area_available
            )
            for mcfg, scaler_type, exp_num in other_combos
        )

        return rf_results + other_results

    # ------------------------------------------------------------------
    # Loop de um experimento
    # ---------------------------------------------------------------
    # OPT-3: cache de agregação por var_key — _aggregate_features é
    #         chamado UMA VEZ por variante, não repetido para cada n_vars.
    # OPT-4: _run_combos_parallel substitui o loop sequencial de combos.
    # ------------------------------------------------------------------

    def _run_single_experiment(self, exp):
        exp_t0 = time.perf_counter()
        train_df, test_df, target_col = self._prepare_experiment(exp)
        climate_vars = self._identify_climate_vars(train_df)
        print(f"  ✓ {len(climate_vars)} variáveis climáticas identificadas\n")

        train_clean = self._drop_leakage(train_df, target_col, exp['preditores'],
                                         climate_vars, self.year_col)
        test_clean  = (self._drop_leakage(test_df, target_col, exp['preditores'],
                                          climate_vars, self.year_col)
                       if test_df is not None else None)

        if self.validation_method == 'A':
            train_norm, test_norm = self._normalize_pair(train_clean, test_clean, climate_vars)
        else:
            train_norm, test_norm = train_clean.copy(), None
            print("   🔧 Normalização climática pré-CV ignorada; scaler aplicado dentro dos folds")

        exp_results = []
        exp_num     = 0
        best_r2     = -np.inf
        total_possible = len(VARIANTS) * len(FEATURE_COUNTS) * len(SCALERS) * len(MODEL_CONFIGS)

        # OPT-3: cache de agregação — chave = var_key
        agg_cache = {}

        for var_key, var_cfg in VARIANTS.items():
            variant_t0 = time.perf_counter()
            print(f"  🔬 {var_cfg['name']}")

            # OPT-3: agrega UMA VEZ por variante e reutiliza para todos os n_vars
            if var_key not in agg_cache:
                agg_t0 = time.perf_counter()
                tr_agg = self._aggregate_features(train_norm, var_cfg, climate_vars)
                te_agg = (self._aggregate_features(test_norm, var_cfg, climate_vars)
                          if test_norm is not None else None)

                # Correção #2/#6: drop_leakage após agregação
                tr_agg = self._drop_leakage(tr_agg, target_col, exp['preditores'],
                                             climate_vars, self.year_col)
                if te_agg is not None:
                    te_agg = self._drop_leakage(te_agg, target_col, exp['preditores'],
                                                 climate_vars, self.year_col)

                agg_cache[var_key] = (tr_agg, te_agg)
                agg_elapsed = time.perf_counter() - agg_t0
                area_available = 'area_plantada_ha' in tr_agg.columns
                print(f"     Features disponíveis: {tr_agg.shape[1] - 1} | "
                      f"agregação: {format_duration(agg_elapsed)}")
            else:
                # OPT-3: reutiliza resultado em cache sem recalcular
                tr_agg, te_agg = agg_cache[var_key]
                area_available = 'area_plantada_ha' in tr_agg.columns
                print(f"     Features disponíveis: {tr_agg.shape[1] - 1} | "
                      f"agregação: [cache]")

            for n_vars in FEATURE_COUNTS:
                if n_vars > tr_agg.shape[1] - 1:
                    continue

                print(f"     Selecionando top {n_vars} features via RF (50 árvores)...", flush=True)
                select_t0 = time.perf_counter()
                df_tr_sel, df_te_sel, top_feats = self._select_features_rf(
                    tr_agg, te_agg, target_col, self.year_col, n_vars
                )
                print(f"     Top {n_vars} selecionadas em "
                      f"{format_duration(time.perf_counter() - select_t0)}",
                      flush=True)

                # OPT-4: distribui combos (scaler × modelo) em paralelo
                combo_t0 = time.perf_counter()
                print(f"     Rodando {len(SCALERS) * len(MODEL_CONFIGS)} combos em paralelo...",
                      flush=True)

                batch_results = self._run_combos_parallel(
                    df_tr_sel, df_te_sel, exp['id'], target_col,
                    var_key, n_vars, top_feats, area_available, exp_num
                )
                exp_num += len(batch_results)

                # Log e rastreamento do melhor resultado do batch
                for res in batch_results:
                    exp_results.append(res)
                    r2t = res['r2_test']
                    if r2t > best_r2:
                        best_r2 = r2t
                        print(f"   ⭐ {res['model']:<6} | {res['scaler']:<8} | "
                              f"n={n_vars} | R²Test={r2t:.4f} | "
                              f"R²CV={res['r2_cv']:.4f} | Overfit={res['overfit']:.3f} | "
                              f"t_batch={format_duration(time.perf_counter() - combo_t0)} "
                              f"- {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", flush=True)

                print(f"     Batch concluído em {format_duration(time.perf_counter() - combo_t0)}",
                      flush=True)

            print(f"     Tempo da variante: {format_duration(time.perf_counter() - variant_t0)}\n",
                  flush=True)

        print(f"  Tempo total do experimento {exp['id']}: "
              f"{format_duration(time.perf_counter() - exp_t0)}\n", flush=True)
        return exp_results

    # ------------------------------------------------------------------
    # Relatório por experimento
    # ------------------------------------------------------------------

    def _display_experiment_report(self, exp, exp_results):
        df_res  = pd.DataFrame(exp_results).sort_values('r2_test', ascending=False)
        exp_id  = exp['id']
        unidade = exp['unidade']
        method_label = {
            'A': '[A] Split temporal 2018-2022 / 2023-2024',
            'B': '[B] CV 70/30 ShuffleSplit (10 iter)',
            'C': '[C] CV 10 folds KFold',
        }[self.validation_method]

        print("=" * 120)
        print(f"🏆 [{exp_id}] TOP 30 RESULTADOS — {exp['descricao']}")
        print(f"   Método de validação: {method_label}")
        print("=" * 120 + "\n")

        df_res.to_csv(f'relatorio_{exp_id}_metodo{self.validation_method}_detalhado.csv',
                      index=False, encoding='utf-8-sig')
        df_res.head(1).to_csv(f'relatorio_{exp_id}_metodo{self.validation_method}_melhor.csv',
                               index=False, encoding='utf-8-sig')
        print(f"  CSVs salvos: relatorio_{exp_id}_metodo{self.validation_method}_*.csv\n")

        hdr = (f"{'#':<4} {'Variante':<22} {'n':<5} {'Modelo':<6} "
               f"{'Scaler':<10} {'Config':<28} {'R²Test':<9} {'R²CV':<9} {'Overfit':<8}")
        print(hdr)
        print("-" * 130)
        for i, (_, row) in enumerate(df_res.head(30).iterrows(), 1):
            print(f"{i:<4} {row['variant']:<22} {row['n_features']:<5} {row['model']:<6} "
                  f"{row['scaler']:<10} {row['params_short']:<28} "
                  f"{row['r2_test']:<9.4f} {row['r2_cv']:<9.4f} {row['overfit']:<8.3f}")

        best = df_res.iloc[0]
        print("\n" + "=" * 120)
        print(f"🎯 [{exp_id}] MELHOR CONFIGURAÇÃO")
        print("=" * 120)
        print(f"  Variante:  {best['variant']}")
        print(f"  Features:  {best['n_features']}")
        print(f"  Modelo:    {best['model']}  ({best['params_short']})")
        print(f"  Scaler:    {best['scaler']}")
        print(f"\n  R² Teste:    {best['r2_test']:.4f} ⭐")
        print(f"  R² CV:       {best['r2_cv']:.4f} ± {best['cv_std']:.4f}"
              f"  (escopo: {best['cv_scope']})")
        print(f"  R² Treino:   {best['r2_train']:.4f}")
        print(f"  RMSE:        {best['rmse_test']:.2f} {unidade}")
        print(f"  MAE:         {best['mae_test']:.2f} {unidade}")
        print(f"  RAE:         {best['rae_test_pct']:.2f}%")
        print(f"  RRSE:        {best['rrse_test_pct']:.2f}%")
        print(f"  Overfitting: {best['overfit']:.4f}")

        print(f"\n📊 POR VARIANTE:")
        print(df_res.groupby('variant').agg(
            R2_medio=('r2_test', 'mean'), R2_max=('r2_test', 'max'),
            N=('r2_test', 'count')).round(4).sort_values('R2_max', ascending=False))

        print(f"\n📊 POR MODELO:")
        print(df_res.groupby('model').agg(
            R2_medio=('r2_test', 'mean'), R2_max=('r2_test', 'max'),
            N=('r2_test', 'count')).round(4).sort_values('R2_max', ascending=False))

        print(f"\n📊 POR SCALER:")
        print(df_res.groupby('scaler').agg(
            R2_medio=('r2_test', 'mean'), R2_max=('r2_test', 'max'),
            N=('r2_test', 'count')).round(4).sort_values('R2_max', ascending=False))

        print()
        return df_res

    # ------------------------------------------------------------------
    # Salvamento incremental
    # ------------------------------------------------------------------

    def _save_incremental_results(self, exp_id):
        if not self.all_results:
            return

        df_partial = pd.DataFrame(self.all_results).sort_values('r2_test', ascending=False)
        base = f'relatorio_consolidado_metodo{self.validation_method}_ate_{exp_id}'
        df_partial.to_csv(f'{base}_detalhado.csv', index=False, encoding='utf-8-sig')
        df_partial.head(1).to_csv(f'{base}_melhor.csv', index=False, encoding='utf-8-sig')
        print(f"  Salvamento incremental: {base}_*.csv\n")

    # ------------------------------------------------------------------
    # Checkpoints por experimento
    # ------------------------------------------------------------------

    def _checkpoint_path(self, exp_id):
        return os.path.join(CHECKPOINT_DIR, f'checkpoint_{exp_id}_metodo{self.validation_method}.csv')

    def _load_experiment_checkpoint(self, exp_id):
        path = self._checkpoint_path(exp_id)
        try:
            df_checkpoint = pd.read_csv(path, encoding='utf-8-sig')
        except FileNotFoundError:
            return None

        if df_checkpoint.empty:
            print(f"  ⚠ Checkpoint vazio encontrado para {exp_id}: {path}. Reprocessando EXP.")
            return None

        print(f"  ✓ Checkpoint encontrado para {exp_id}: {path}")
        print(f"    {len(df_checkpoint)} resultados carregados; EXP será pulado.\n")
        return df_checkpoint.to_dict('records')

    def _save_experiment_checkpoint(self, exp_id, exp_results):
        if not exp_results:
            return

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = self._checkpoint_path(exp_id)
        df_checkpoint = pd.DataFrame(exp_results).sort_values('r2_test', ascending=False)
        df_checkpoint.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  Checkpoint salvo: {path}\n")

    # ------------------------------------------------------------------
    # Relatório consolidado
    # ------------------------------------------------------------------

    def _display_consolidated_report(self):
        method_label = {
            'A': '[A] Split temporal 2018-2022 / 2023-2024',
            'B': '[B] CV 70/30 ShuffleSplit (10 iter)',
            'C': '[C] CV 10 folds KFold',
        }[self.validation_method]

        print("=" * 120)
        print("🌐 RELATÓRIO CONSOLIDADO — TODOS OS EXPERIMENTOS")
        print(f"   Método de validação: {method_label}")
        print("=" * 120 + "\n")

        df_all = pd.DataFrame(self.all_results).sort_values('r2_test', ascending=False)
        df_all.to_csv(f'relatorio_consolidado_metodo{self.validation_method}_detalhado.csv',
                      index=False, encoding='utf-8-sig')
        df_all.head(1).to_csv(f'relatorio_consolidado_metodo{self.validation_method}_melhor.csv',
                               index=False, encoding='utf-8-sig')
        print("CSVs consolidados salvos:")
        print(f"  relatorio_consolidado_metodo{self.validation_method}_detalhado.csv")
        print(f"  relatorio_consolidado_metodo{self.validation_method}_melhor.csv\n")

        if df_all['validation_method'].nunique() > 1:
            print("⚠  ATENÇÃO: r2_cv NÃO é comparável entre métodos A e B/C.")
            print("   Método A: cv_scope='train_only' (apenas 2018–2022)")
            print("   Métodos B/C: cv_scope='full_dataset' (base inteira)\n")

        print("📊 TOP 30 GLOBAL:")
        hdr = (f"{'#':<4} {'ExpID':<7} {'Target':<28} {'Variante':<22} "
               f"{'n':<5} {'Modelo':<6} {'Scaler':<10} {'R²Test':<9} {'R²CV':<9} {'Overfit':<8}")
        print(hdr)
        print("-" * 155)
        for i, (_, row) in enumerate(df_all.head(30).iterrows(), 1):
            print(f"{i:<4} {row['experiment_id']:<7} {str(row['target_col']):<28} "
                  f"{row['variant']:<22} {row['n_features']:<5} {row['model']:<6} "
                  f"{row['scaler']:<10} {row['r2_test']:<9.4f} "
                  f"{row['r2_cv']:<9.4f} {row['overfit']:<8.3f}")

        best = df_all.iloc[0]
        print("\n" + "=" * 120)
        print("🎯 MELHOR CONFIGURAÇÃO GLOBAL")
        print("=" * 120)
        print(f"  Experimento: {best['experiment_id']}  ({best['target_col']})")
        print(f"  Variante:    {best['variant']}")
        print(f"  Features:    {best['n_features']}")
        print(f"  Modelo:      {best['model']}  ({best['params_short']})")
        print(f"  Scaler:      {best['scaler']}")
        print(f"\n  R² Teste:    {best['r2_test']:.4f} ⭐")
        print(f"  R² CV:       {best['r2_cv']:.4f} ± {best['cv_std']:.4f}"
              f"  (escopo: {best['cv_scope']})")
        print(f"  R² Treino:   {best['r2_train']:.4f}")
        print(f"  RMSE:        {best['rmse_test']:.2f}")
        print(f"  MAE:         {best['mae_test']:.2f}")
        print(f"  RAE:         {best['rae_test_pct']:.2f}%")
        print(f"  RRSE:        {best['rrse_test_pct']:.2f}%")
        print(f"  Overfitting: {best['overfit']:.4f}")

        print(f"\n📊 RANKING DE TARGETS (R² máximo):")
        print(df_all.groupby('target_col').agg(
            R2_max=('r2_test', 'max'), R2_medio=('r2_test', 'mean'),
            N=('r2_test', 'count')).round(4).sort_values('R2_max', ascending=False))

        print(f"\n📊 POR MODELO (global):")
        print(df_all.groupby('model').agg(
            R2_medio=('r2_test', 'mean'), R2_max=('r2_test', 'max'),
            N=('r2_test', 'count')).round(4).sort_values('R2_max', ascending=False))

        print(f"\n📊 POR SCALER (global):")
        print(df_all.groupby('scaler').agg(
            R2_medio=('r2_test', 'mean'), R2_max=('r2_test', 'max'),
            N=('r2_test', 'count')).round(4).sort_values('R2_max', ascending=False))

        print(f"\n📊 POR VARIANTE (global):")
        print(df_all.groupby('variant').agg(
            R2_medio=('r2_test', 'mean'), R2_max=('r2_test', 'max'),
            N=('r2_test', 'count')).round(4).sort_values('R2_max', ascending=False))

        return df_all

    # ------------------------------------------------------------------
    # Ponto de entrada
    # ------------------------------------------------------------------

    def run(self):
        method_t0 = time.perf_counter()
        method_label = {
            'A': '[A] Split temporal fixo — treino 2018–2022 / teste 2023–2024',
            'B': '[B] CV aleatório 70/30 — ShuffleSplit, base inteira',
            'C': '[C] CV 10 folds — KFold, base inteira',
        }[self.validation_method]

        print("\n" + "=" * 120)
        print(f"📋 MÉTODO DE VALIDAÇÃO: {method_label}")
        print("=" * 120)

        self.load_raw()

        for exp in EXPERIMENTOS:
            checkpoint_results = self._load_experiment_checkpoint(exp['id'])
            if checkpoint_results is not None:
                self.all_results.extend(checkpoint_results)
                self._save_incremental_results(exp['id'])
                continue

            print("\n" + "=" * 120)
            print(f"🚀 INICIANDO {exp['id']}: {exp['descricao']}")
            if self.validation_method == 'A':
                print(f"   Split: treino {self.train_years[0]}–{self.train_years[-1]} "
                      f"/ teste {self.test_years[0]}–{self.test_years[-1]}")
            print("=" * 120)

            exp_results = self._run_single_experiment(exp)
            self.all_results.extend(exp_results)
            self._display_experiment_report(exp, exp_results)
            self._save_experiment_checkpoint(exp['id'], exp_results)
            self._save_incremental_results(exp['id'])

        df_report = self._display_consolidated_report()
        print(f"\n⏱ Tempo total do método {self.validation_method}: "
              f"{format_duration(time.perf_counter() - method_t0)}")
        return df_report


# ===============================================================================
# EXECUÇÃO MULTIMÉTODO
# ===============================================================================

def run_all_validation_methods(methods=None):
    if methods is None:
        methods = [VALIDATION_METHOD]

    all_t0 = time.perf_counter()
    all_method_results = []

    for method in methods:
        tester = SupervisedUnified(
            parquet_path      = PARQUET_PATH,
            year_col          = YEAR_COL,
            train_years       = TRAIN_YEARS,
            test_years        = TEST_YEARS,
            validation_method = method,
        )
        df_method = tester.run()
        all_method_results.append(df_method)

    if not all_method_results:
        return pd.DataFrame()

    df_all_methods = pd.concat(all_method_results, ignore_index=True)
    df_all_methods = df_all_methods.sort_values('r2_test', ascending=False)
    methods_label = ''.join(methods)
    df_all_methods.to_csv(f'relatorio_consolidado_metodo{methods_label}_detalhado.csv',
                          index=False, encoding='utf-8-sig')
    df_all_methods.head(1).to_csv(f'relatorio_consolidado_metodo{methods_label}_melhor.csv',
                                  index=False, encoding='utf-8-sig')

    print("\n" + "=" * 120)
    print(f"RELATÓRIO CONSOLIDADO — MÉTODO(S) {', '.join(methods)}")
    print("=" * 120)
    print("CSVs consolidados salvos:")
    print(f"  relatorio_consolidado_metodo{methods_label}_detalhado.csv")
    print(f"  relatorio_consolidado_metodo{methods_label}_melhor.csv")
    print(f"Tempo total da execução: {format_duration(time.perf_counter() - all_t0)}")

    return df_all_methods


def _methods_from_argv(argv):
    if len(argv) <= 1:
        return [VALIDATION_METHOD]

    raw_args = [arg.strip().upper() for arg in argv[1:] if arg.strip()]
    if any(arg in ('ALL', 'TODOS') for arg in raw_args):
        return VALIDATION_METHODS

    methods = []
    for arg in raw_args:
        methods.extend(part.strip().upper() for part in arg.split(',') if part.strip())

    invalid = [method for method in methods if method not in VALIDATION_METHODS]
    if invalid:
        raise ValueError(
            f"Método(s) inválido(s): {', '.join(invalid)}. "
            f"Use A, B, C ou ALL."
        )

    return methods or [VALIDATION_METHOD]


# ===============================================================================
# EXECUÇÃO
# ===============================================================================

if __name__ == "__main__":
    script_t0 = time.perf_counter()
    results = run_all_validation_methods(_methods_from_argv(sys.argv))
    print("\n" + "=" * 120)
    print("✅ CONCLUÍDO — TODOS OS EXPERIMENTOS FINALIZADOS!")
    print(f"⏱ Tempo total do script: {format_duration(time.perf_counter() - script_t0)}")
    print("=" * 120)