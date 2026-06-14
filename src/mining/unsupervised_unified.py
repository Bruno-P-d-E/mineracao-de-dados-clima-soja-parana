from __future__ import annotations

import argparse
import os
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# Configuracao
# =============================================================================

PARQUET_PATH = Path(r"C:\Users\bruno\Desktop\Pipeline_TCC\data\processed\dataset_final.parquet")
OUTPUT_DIR = Path("reports_unsupervised")

ID_COL = "cod_ibge"
NAME_COL = "municipio"
YEAR_COL = "ano"
TARGET_COL = "rendimento_kg_ha"
LAT_COL = "latitude"
LON_COL = "longitude"

K_MIN = 2
K_MAX = 10
REFERENCE_QUANTILES = 4
PCA_VARIANCE = 0.85
MIN_PCA_COMPONENTS = 5
RANDOM_STATE = 42
STABILITY_TOP_N = 24
STABILITY_SEEDS = list(range(10))
BOOTSTRAPS = 20
BOOTSTRAP_FRACTION = 0.80


def format_duration(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f}s"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def log(message: str) -> None:
    print(message, flush=True)


def safe_slug(value: str) -> str:
    return (
        str(value)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


# =============================================================================
# Preparacao dos dados
# =============================================================================


def identify_climate_features(df: pd.DataFrame) -> list[str]:
    climate_cols = [c for c in df.columns if "dec" in c.lower()]
    return [
        c
        for c in climate_cols
        if "lat" not in c.lower()
        and "lon" not in c.lower()
        and c not in {ID_COL, NAME_COL, YEAR_COL, TARGET_COL, LAT_COL, LON_COL}
    ]


def identify_climate_vars(columns: list[str]) -> list[str]:
    climate_vars = set()
    for col in columns:
        if "dec" in col and "ano" in col:
            climate_vars.add(col.split("dec")[0].rstrip("_"))
    return sorted(climate_vars)


def build_municipality_dataset(df: pd.DataFrame, climate_cols: list[str]) -> pd.DataFrame:
    required = [ID_COL, TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas obrigatorias ausentes: {missing}")

    agg = {c: "mean" for c in climate_cols}
    agg[TARGET_COL] = "mean"

    optional_first = [NAME_COL, LAT_COL, LON_COL]
    for col in optional_first:
        if col in df.columns:
            agg[col] = "first"

    mun = df.groupby(ID_COL).agg(agg).reset_index()

    coverage = mun[climate_cols].notna().mean(axis=1)
    mun = mun[coverage >= 0.70].copy()
    mun["feature_coverage"] = coverage.loc[mun.index].values

    return mun.reset_index(drop=True)


def make_reference_classes(target: pd.Series, q: int = REFERENCE_QUANTILES) -> np.ndarray:
    return pd.qcut(target, q=q, labels=False, duplicates="drop").astype(int).to_numpy()


def impute_and_scale(
    X: pd.DataFrame,
    scaler_type: str = "standard",
) -> tuple[np.ndarray, list[str], SimpleImputer, StandardScaler | RobustScaler]:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    if scaler_type == "robust":
        scaler: StandardScaler | RobustScaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, list(X.columns), imputer, scaler


def aggregate_phase_features(df: pd.DataFrame, climate_cols: list[str]) -> pd.DataFrame:
    climate_vars = identify_climate_vars(climate_cols)
    result = pd.DataFrame(index=df.index)

    phases = {
        "previous_late": ("ano1", range(26, 37)),
        "early": ("ano2", range(1, 5)),
        "flowering": ("ano2", range(5, 11)),
        "grain": ("ano2", range(11, 16)),
        "maturation": ("ano2", range(16, 19)),
        "late": ("ano2", range(19, 26)),
    }

    for var in climate_vars:
        var_cols = [c for c in climate_cols if c.startswith(f"{var}_dec")]
        if not var_cols:
            continue

        for phase_name, (year_token, dec_range) in phases.items():
            cols = [
                c
                for c in var_cols
                if year_token in c and any(f"dec{dec}_" in c for dec in dec_range)
            ]
            if not cols:
                continue

            data = df[cols]
            prefix = f"{var}_{phase_name}"
            mean = data.mean(axis=1)
            std = data.std(axis=1)
            q75 = data.quantile(0.75, axis=1)
            q25 = data.quantile(0.25, axis=1)

            result[f"{prefix}_mean"] = mean
            result[f"{prefix}_std"] = std
            result[f"{prefix}_iqr"] = q75 - q25

    return result


def build_representations(
    mun: pd.DataFrame,
    climate_cols: list[str],
    quick: bool = False,
) -> dict[str, dict[str, object]]:
    reps: dict[str, dict[str, object]] = {}

    raw_df = mun[climate_cols].copy()
    X_raw, raw_features, _, _ = impute_and_scale(raw_df, scaler_type="standard")
    reps["raw_dec_scaled"] = {
        "X": X_raw,
        "features": raw_features,
        "description": "Todas as features decendiais, imputadas por mediana e padronizadas.",
        "interpretable": True,
    }

    pca_probe = PCA(random_state=RANDOM_STATE).fit(X_raw)
    cum_var = np.cumsum(pca_probe.explained_variance_ratio_)
    n_comp = max(int(np.searchsorted(cum_var, PCA_VARIANCE)) + 1, MIN_PCA_COMPONENTS)
    n_comp = min(n_comp, X_raw.shape[0] - 1, X_raw.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_raw)
    reps["pca_85_raw_dec"] = {
        "X": X_pca,
        "features": [f"PC{i + 1}" for i in range(n_comp)],
        "description": f"PCA sobre decendiais padronizadas, {n_comp} componentes.",
        "interpretable": False,
        "pca_variance": float(cum_var[n_comp - 1]),
    }

    phase_df = aggregate_phase_features(mun, climate_cols)
    X_phase, phase_features, _, _ = impute_and_scale(phase_df, scaler_type="standard")
    reps["phenology_stats_scaled"] = {
        "X": X_phase,
        "features": phase_features,
        "description": "Estatisticas por fase fenologica: mean, std e iqr.",
        "interpretable": True,
    }

    phase_pca_probe = PCA(random_state=RANDOM_STATE).fit(X_phase)
    phase_cum_var = np.cumsum(phase_pca_probe.explained_variance_ratio_)
    phase_n_comp = max(int(np.searchsorted(phase_cum_var, PCA_VARIANCE)) + 1, MIN_PCA_COMPONENTS)
    phase_n_comp = min(phase_n_comp, X_phase.shape[0] - 1, X_phase.shape[1])
    phase_pca = PCA(n_components=phase_n_comp, random_state=RANDOM_STATE)
    X_phase_pca = phase_pca.fit_transform(X_phase)
    reps["pca_85_phenology"] = {
        "X": X_phase_pca,
        "features": [f"PC{i + 1}" for i in range(phase_n_comp)],
        "description": f"PCA sobre estatisticas fenologicas, {phase_n_comp} componentes.",
        "interpretable": False,
        "pca_variance": float(phase_cum_var[phase_n_comp - 1]),
    }

    if quick:
        return {
            "pca_85_raw_dec": reps["pca_85_raw_dec"],
            "phenology_stats_scaled": reps["phenology_stats_scaled"],
        }

    return reps


# =============================================================================
# Modelos, metricas e estabilidade
# =============================================================================


def fit_cluster_model(
    X: np.ndarray,
    algorithm: str,
    k: int,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, object, dict[str, float]]:
    extras: dict[str, float] = {}

    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=30, max_iter=500)
        labels = model.fit_predict(X)
        extras["inertia"] = float(model.inertia_)
        return labels, model, extras

    if algorithm == "agglomerative_ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        return labels, model, extras

    if algorithm == "gmm_diag":
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=seed,
            n_init=10,
            max_iter=500,
            reg_covar=1e-6,
        )
        labels = model.fit_predict(X)
        extras["bic"] = float(model.bic(X))
        extras["aic"] = float(model.aic(X))
        return labels, model, extras

    raise ValueError(f"Algoritmo desconhecido: {algorithm}")


def compute_internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "min_cluster_share": np.nan,
            "cluster_size_cv": np.nan,
        }

    counts = pd.Series(labels).value_counts().sort_index()
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "min_cluster_share": float(counts.min() / len(labels)),
        "cluster_size_cv": float(counts.std(ddof=0) / counts.mean()),
    }


def compute_external_metrics(
    labels: np.ndarray,
    reference_classes: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    groups = [target[labels == lab] for lab in sorted(np.unique(labels))]
    if len(groups) > 1 and all(len(g) > 0 for g in groups):
        h_stat, p_value = kruskal(*groups)
        epsilon_sq = (h_stat - len(groups) + 1) / max(len(target) - len(groups), 1)
        epsilon_sq = max(0.0, min(float(epsilon_sq), 1.0))
    else:
        h_stat, p_value, epsilon_sq = np.nan, np.nan, np.nan

    return {
        "ari_vs_y_quartile": float(adjusted_rand_score(reference_classes, labels)),
        "nmi_vs_y_quartile": float(normalized_mutual_info_score(reference_classes, labels)),
        "ami_vs_y_quartile": float(adjusted_mutual_info_score(reference_classes, labels)),
        "kruskal_h_target": float(h_stat),
        "kruskal_p_target": float(p_value),
        "kruskal_epsilon_sq": float(epsilon_sq),
    }


def mean_pairwise_ari(label_sets: list[np.ndarray]) -> tuple[float, float]:
    if len(label_sets) < 2:
        return np.nan, np.nan
    values = [adjusted_rand_score(a, b) for a, b in combinations(label_sets, 2)]
    return float(np.mean(values)), float(np.std(values))


def compute_seed_stability(
    X: np.ndarray,
    algorithm: str,
    k: int,
    seeds: list[int],
) -> tuple[float, float]:
    if algorithm == "agglomerative_ward":
        return 1.0, 0.0

    label_sets = [fit_cluster_model(X, algorithm, k, seed=seed)[0] for seed in seeds]
    return mean_pairwise_ari(label_sets)


def compute_bootstrap_stability(
    X: np.ndarray,
    algorithm: str,
    k: int,
    base_labels: np.ndarray,
    n_bootstraps: int,
    fraction: float,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    sample_size = max(k + 1, int(round(n * fraction)))
    values = []

    for i in range(n_bootstraps):
        idx = np.sort(rng.choice(n, size=sample_size, replace=False))
        labels_sample, _, _ = fit_cluster_model(X[idx], algorithm, k, seed=seed + i + 1)
        values.append(adjusted_rand_score(base_labels[idx], labels_sample))

    return float(np.mean(values)), float(np.std(values))


def evaluate_grid(
    representations: dict[str, dict[str, object]],
    reference_classes: np.ndarray,
    target: np.ndarray,
    k_values: range,
    algorithms: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    labels_cache: dict[str, np.ndarray] = {}

    for rep_name, rep in representations.items():
        X = rep["X"]
        log(f"\nRepresentacao: {rep_name} | shape={X.shape}")

        for algorithm in algorithms:
            for k in k_values:
                t0 = time.perf_counter()
                labels, _, extras = fit_cluster_model(X, algorithm, k, seed=RANDOM_STATE)
                cache_key = f"{rep_name}__{algorithm}__k{k}"
                labels_cache[cache_key] = labels

                internal = compute_internal_metrics(X, labels)
                external = compute_external_metrics(labels, reference_classes, target)
                row = {
                    "solution_id": cache_key,
                    "representation": rep_name,
                    "algorithm": algorithm,
                    "k": k,
                    "n_features": X.shape[1],
                    "fit_seconds": time.perf_counter() - t0,
                    **internal,
                    **external,
                    **extras,
                }
                rows.append(row)

                log(
                    "  "
                    f"{algorithm:<18} k={k:<2} "
                    f"sil={row['silhouette']:.4f} "
                    f"db={row['davies_bouldin']:.4f} "
                    f"ch={row['calinski_harabasz']:.1f} "
                    f"ARIext={row['ari_vs_y_quartile']:.4f} "
                    f"t={format_duration(row['fit_seconds'])}"
                )

    return pd.DataFrame(rows), labels_cache


def add_internal_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank_silhouette"] = out["silhouette"].rank(ascending=False, method="min")
    out["rank_davies"] = out["davies_bouldin"].rank(ascending=True, method="min")
    out["rank_calinski"] = out["calinski_harabasz"].rank(ascending=False, method="min")
    out["rank_balance"] = out["min_cluster_share"].rank(ascending=False, method="min")
    out["internal_rank_sum"] = (
        out["rank_silhouette"]
        + out["rank_davies"]
        + out["rank_calinski"]
        + out["rank_balance"]
    )
    return out


def add_stability(
    df: pd.DataFrame,
    labels_cache: dict[str, np.ndarray],
    representations: dict[str, dict[str, object]],
    top_n: int,
    seeds: list[int],
    bootstraps: int,
    bootstrap_fraction: float,
) -> pd.DataFrame:
    out = df.copy()
    out["seed_stability_ari_mean"] = np.nan
    out["seed_stability_ari_std"] = np.nan
    out["bootstrap_stability_ari_mean"] = np.nan
    out["bootstrap_stability_ari_std"] = np.nan

    candidates = (
        out.sort_values("internal_rank_sum", ascending=True)
        .head(top_n)
        .copy()
    )

    log(f"\nCalculando estabilidade para top {len(candidates)} solucoes internas...")
    for _, row in candidates.iterrows():
        rep_name = row["representation"]
        algorithm = row["algorithm"]
        k = int(row["k"])
        solution_id = row["solution_id"]
        X = representations[rep_name]["X"]
        labels = labels_cache[solution_id]

        t0 = time.perf_counter()
        seed_mean, seed_std = compute_seed_stability(X, algorithm, k, seeds)
        boot_mean, boot_std = compute_bootstrap_stability(
            X,
            algorithm,
            k,
            labels,
            n_bootstraps=bootstraps,
            fraction=bootstrap_fraction,
        )

        mask = out["solution_id"] == solution_id
        out.loc[mask, "seed_stability_ari_mean"] = seed_mean
        out.loc[mask, "seed_stability_ari_std"] = seed_std
        out.loc[mask, "bootstrap_stability_ari_mean"] = boot_mean
        out.loc[mask, "bootstrap_stability_ari_std"] = boot_std

        log(
            "  "
            f"{solution_id:<40} "
            f"seedARI={seed_mean:.4f} "
            f"bootARI={boot_mean:.4f} "
            f"t={format_duration(time.perf_counter() - t0)}"
        )

    stable_mask = out["bootstrap_stability_ari_mean"].notna()
    out.loc[stable_mask, "rank_seed_stability"] = out.loc[
        stable_mask, "seed_stability_ari_mean"
    ].rank(ascending=False, method="min")
    out.loc[stable_mask, "rank_bootstrap_stability"] = out.loc[
        stable_mask, "bootstrap_stability_ari_mean"
    ].rank(ascending=False, method="min")
    out.loc[stable_mask, "consensus_rank_sum"] = (
        out.loc[stable_mask, "internal_rank_sum"]
        + out.loc[stable_mask, "rank_seed_stability"]
        + out.loc[stable_mask, "rank_bootstrap_stability"]
    )

    return out


# =============================================================================
# Relatorios, perfis e figuras
# =============================================================================


def cluster_summary(
    mun: pd.DataFrame,
    labels: np.ndarray,
    reference_classes: np.ndarray,
) -> pd.DataFrame:
    tmp = mun[[ID_COL, TARGET_COL, "feature_coverage"]].copy()
    tmp["cluster"] = labels
    tmp["reference_y_quartile"] = reference_classes

    if NAME_COL in mun.columns:
        tmp[NAME_COL] = mun[NAME_COL].values

    summary = (
        tmp.groupby("cluster")
        .agg(
            n=(ID_COL, "count"),
            target_mean=(TARGET_COL, "mean"),
            target_median=(TARGET_COL, "median"),
            target_std=(TARGET_COL, "std"),
            target_q25=(TARGET_COL, lambda s: s.quantile(0.25)),
            target_q75=(TARGET_COL, lambda s: s.quantile(0.75)),
            coverage_mean=("feature_coverage", "mean"),
        )
        .reset_index()
        .sort_values("target_median")
    )
    summary["target_iqr"] = summary["target_q75"] - summary["target_q25"]
    return summary


def feature_profile(
    X: np.ndarray,
    features: list[str],
    labels: np.ndarray,
    top_n: int = 30,
) -> pd.DataFrame:
    global_mean = X.mean(axis=0)
    rows = []

    for cluster in sorted(np.unique(labels)):
        idx = labels == cluster
        cluster_mean = X[idx].mean(axis=0)
        diff = cluster_mean - global_mean
        top_idx = np.argsort(np.abs(diff))[::-1][:top_n]

        for rank, feature_idx in enumerate(top_idx, 1):
            rows.append(
                {
                    "cluster": int(cluster),
                    "rank_abs_diff": rank,
                    "feature": features[feature_idx],
                    "cluster_mean_scaled": float(cluster_mean[feature_idx]),
                    "global_mean_scaled": float(global_mean[feature_idx]),
                    "diff_scaled": float(diff[feature_idx]),
                    "abs_diff_scaled": float(abs(diff[feature_idx])),
                }
            )

    return pd.DataFrame(rows)


def plot_internal_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    metrics = [
        ("silhouette", "Silhouette (maior melhor)"),
        ("davies_bouldin", "Davies-Bouldin (menor melhor)"),
        ("calinski_harabasz", "Calinski-Harabasz (maior melhor)"),
    ]

    for metric, title in metrics:
        g = sns.relplot(
            data=df,
            x="k",
            y=metric,
            hue="algorithm",
            col="representation",
            kind="line",
            marker="o",
            col_wrap=2,
            facet_kws={"sharey": False},
            height=4,
            aspect=1.35,
        )
        g.fig.suptitle(title, y=1.02)
        g.set_titles("{col_name}")
        g.savefig(output_dir / f"metric_{metric}.png", dpi=160, bbox_inches="tight")
        plt.close(g.fig)


def plot_target_by_cluster(
    mun: pd.DataFrame,
    labels: np.ndarray,
    output_dir: Path,
    solution_id: str,
) -> None:
    tmp = pd.DataFrame({"cluster": labels, TARGET_COL: mun[TARGET_COL].values})
    order = (
        tmp.groupby("cluster")[TARGET_COL]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    plt.figure(figsize=(10, 5.5))
    sns.boxplot(data=tmp, x="cluster", y=TARGET_COL, order=order, color="#8fb9d9")
    sns.stripplot(
        data=tmp,
        x="cluster",
        y=TARGET_COL,
        order=order,
        color="black",
        alpha=0.35,
        size=3,
    )
    plt.title(f"Rendimento por cluster - {solution_id}")
    plt.xlabel("Cluster")
    plt.ylabel("Rendimento medio (kg/ha)")
    plt.tight_layout()
    plt.savefig(output_dir / "best_solution_target_boxplot.png", dpi=170)
    plt.close()


def plot_cluster_map(
    mun: pd.DataFrame,
    labels: np.ndarray,
    output_dir: Path,
    solution_id: str,
) -> None:
    if LAT_COL not in mun.columns or LON_COL not in mun.columns:
        return

    tmp = mun[[LAT_COL, LON_COL]].copy()
    tmp["cluster"] = labels.astype(str)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=tmp,
        x=LON_COL,
        y=LAT_COL,
        hue="cluster",
        palette="tab10",
        s=55,
        edgecolor="black",
        linewidth=0.35,
    )
    plt.title(f"Clusters por municipio - {solution_id}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "best_solution_spatial_scatter.png", dpi=170)
    plt.close()


def save_best_solution_outputs(
    output_dir: Path,
    df_ranked: pd.DataFrame,
    labels_cache: dict[str, np.ndarray],
    representations: dict[str, dict[str, object]],
    mun: pd.DataFrame,
    reference_classes: np.ndarray,
    make_plots: bool,
) -> pd.Series:
    candidates = df_ranked[df_ranked["consensus_rank_sum"].notna()].copy()
    if candidates.empty:
        candidates = df_ranked.copy()
        sort_cols = ["internal_rank_sum"]
    else:
        sort_cols = ["consensus_rank_sum", "internal_rank_sum"]

    best = candidates.sort_values(sort_cols, ascending=True).iloc[0]
    solution_id = best["solution_id"]
    labels = labels_cache[solution_id]
    rep = representations[best["representation"]]

    labels_df = mun[[ID_COL, TARGET_COL, "feature_coverage"]].copy()
    if NAME_COL in mun.columns:
        labels_df[NAME_COL] = mun[NAME_COL].values
    if LAT_COL in mun.columns and LON_COL in mun.columns:
        labels_df[LAT_COL] = mun[LAT_COL].values
        labels_df[LON_COL] = mun[LON_COL].values

    labels_df["reference_y_quartile"] = reference_classes
    labels_df["cluster"] = labels
    labels_df.to_csv(output_dir / "best_solution_labels.csv", index=False, encoding="utf-8-sig")

    summary = cluster_summary(mun, labels, reference_classes)
    summary.to_csv(output_dir / "best_solution_cluster_summary.csv", index=False, encoding="utf-8-sig")

    profile = feature_profile(rep["X"], rep["features"], labels, top_n=30)
    profile.to_csv(output_dir / "best_solution_feature_profile.csv", index=False, encoding="utf-8-sig")

    if make_plots:
        plot_target_by_cluster(mun, labels, output_dir, solution_id)
        plot_cluster_map(mun, labels, output_dir, solution_id)

    return best


def write_method_note(output_dir: Path, best: pd.Series) -> None:
    text = f"""# Relatorio metodologico - clustering nao supervisionado

Objetivo principal:
- Agrupar municipios por padroes climaticos decendiais, sem usar rendimento,
  area, producao, coordenadas ou identificadores como features de clustering.

Validacao principal:
- Metricas internas: silhouette, Davies-Bouldin, Calinski-Harabasz e equilibrio
  de tamanho dos clusters.
- Estabilidade: ARI entre seeds e ARI entre solucao completa e subamostras
  bootstrap sem reposicao.

Validacao externa/post-hoc:
- O rendimento medio entra apenas depois do clustering.
- Quartis de rendimento sao classes externas de referencia, nao ground truth.
- ARI/NMI/AMI contra quartis medem associacao, nao causalidade.
- Kruskal-Wallis e epsilon quadrado medem separacao de rendimento entre clusters.

Escolha final:
- Solucao escolhida por ranking consensual interno + estabilidade.
- O rendimento nao foi usado para selecionar a solucao final.

Melhor solucao:
- solution_id: {best['solution_id']}
- representacao: {best['representation']}
- algoritmo: {best['algorithm']}
- k: {int(best['k'])}
- silhouette: {best['silhouette']:.4f}
- Davies-Bouldin: {best['davies_bouldin']:.4f}
- Calinski-Harabasz: {best['calinski_harabasz']:.2f}
- estabilidade bootstrap ARI: {best.get('bootstrap_stability_ari_mean', np.nan):.4f}
- ARI externo vs quartis de rendimento: {best['ari_vs_y_quartile']:.4f}
- Kruskal p-valor rendimento: {best['kruskal_p_target']:.6g}

Limitacoes:
- Clustering descreve estrutura dos dados climaticos; nao prova causalidade.
- Rendimento tambem depende de solo, manejo, cultivar, tecnologia, pragas e mercado.
- Resultados externos devem ser apresentados como associacao entre clima e produtividade.
"""
    (output_dir / "methodological_note.md").write_text(text, encoding="utf-8")


# =============================================================================
# Pipeline principal
# =============================================================================


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    t0 = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = range(args.k_min, args.k_max + 1)
    algorithms = ["kmeans", "agglomerative_ward", "gmm_diag"]

    if args.quick:
        algorithms = ["kmeans", "agglomerative_ward"]
        k_values = range(args.k_min, min(args.k_max, 5) + 1)
        args.stability_top_n = min(args.stability_top_n, 8)
        args.bootstraps = min(args.bootstraps, 5)
        args.stability_seeds = min(args.stability_seeds, 4)

    log("=" * 100)
    log("CLUSTERING NAO SUPERVISIONADO - PIPELINE ACADEMICO")
    log("=" * 100)
    log(f"Dataset: {PARQUET_PATH}")
    log(f"Saida:   {output_dir.resolve()}")

    df = pd.read_parquet(PARQUET_PATH)
    climate_cols = identify_climate_features(df)
    mun = build_municipality_dataset(df, climate_cols)
    reference_classes = make_reference_classes(mun[TARGET_COL], q=REFERENCE_QUANTILES)
    target = mun[TARGET_COL].to_numpy(dtype=float)

    log(f"\nLinhas originais: {len(df):,}")
    log(f"Municipios validos: {len(mun):,}")
    log(f"Features climaticas decendiais: {len(climate_cols):,}")
    log(f"Target externo/post-hoc: {TARGET_COL}")

    representations = build_representations(mun, climate_cols, quick=args.quick)
    rep_rows = [
        {
            "representation": name,
            "n_features": rep["X"].shape[1],
            "description": rep["description"],
            "pca_variance": rep.get("pca_variance", np.nan),
        }
        for name, rep in representations.items()
    ]
    pd.DataFrame(rep_rows).to_csv(
        output_dir / "representations_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grid_df, labels_cache = evaluate_grid(
        representations,
        reference_classes=reference_classes,
        target=target,
        k_values=k_values,
        algorithms=algorithms,
    )
    grid_df = add_internal_ranks(grid_df)
    grid_df.to_csv(output_dir / "cluster_model_selection_pre_stability.csv", index=False, encoding="utf-8-sig")

    seeds = list(range(args.stability_seeds))
    ranked_df = add_stability(
        grid_df,
        labels_cache=labels_cache,
        representations=representations,
        top_n=args.stability_top_n,
        seeds=seeds,
        bootstraps=args.bootstraps,
        bootstrap_fraction=args.bootstrap_fraction,
    )
    ranked_df = ranked_df.sort_values(
        ["consensus_rank_sum", "internal_rank_sum"],
        ascending=[True, True],
        na_position="last",
    )
    ranked_df.to_csv(output_dir / "cluster_model_selection.csv", index=False, encoding="utf-8-sig")

    external_cols = [
        "solution_id",
        "representation",
        "algorithm",
        "k",
        "ari_vs_y_quartile",
        "nmi_vs_y_quartile",
        "ami_vs_y_quartile",
        "kruskal_h_target",
        "kruskal_p_target",
        "kruskal_epsilon_sq",
    ]
    ranked_df[external_cols].to_csv(
        output_dir / "external_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not args.no_plots:
        plot_internal_metrics(ranked_df, output_dir)

    best = save_best_solution_outputs(
        output_dir,
        ranked_df,
        labels_cache=labels_cache,
        representations=representations,
        mun=mun,
        reference_classes=reference_classes,
        make_plots=not args.no_plots,
    )
    write_method_note(output_dir, best)

    log("\n" + "=" * 100)
    log("MELHOR SOLUCAO POR CRITERIO INTERNO + ESTABILIDADE")
    log("=" * 100)
    log(f"solution_id: {best['solution_id']}")
    log(f"representacao: {best['representation']}")
    log(f"algoritmo: {best['algorithm']}")
    log(f"k: {int(best['k'])}")
    log(f"silhouette: {best['silhouette']:.4f}")
    log(f"Davies-Bouldin: {best['davies_bouldin']:.4f}")
    log(f"Calinski-Harabasz: {best['calinski_harabasz']:.2f}")
    log(f"bootstrap stability ARI: {best.get('bootstrap_stability_ari_mean', np.nan):.4f}")
    log(f"ARI externo vs quartis rendimento: {best['ari_vs_y_quartile']:.4f}")
    log(f"Kruskal p-valor rendimento: {best['kruskal_p_target']:.6g}")
    log(f"\nArquivos salvos em: {output_dir.resolve()}")
    log(f"Tempo total: {format_duration(time.perf_counter() - t0)}")

    return ranked_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline academico para clustering nao supervisionado climatico."
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--k-min", type=int, default=K_MIN)
    parser.add_argument("--k-max", type=int, default=K_MAX)
    parser.add_argument("--stability-top-n", type=int, default=STABILITY_TOP_N)
    parser.add_argument("--stability-seeds", type=int, default=len(STABILITY_SEEDS))
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--bootstrap-fraction", type=float, default=BOOTSTRAP_FRACTION)
    parser.add_argument("--quick", action="store_true", help="Executa uma versao curta para teste.")
    parser.add_argument("--no-plots", action="store_true", help="Nao gera figuras PNG.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
