"""
Визуализация результатов эксперимента ТОЛЬКО для GCR.

Источники данных:
  - results/gcr_only.json          (метрики RAGAS для одной конфигурации)
  - results/gcr_sweep_no_ragas.csv (сравнение конфигураций без RAGAS)

Выход:
  - results/figures/*.png          (6 графиков)
  - results/tables/*.csv           (2 таблицы)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Настройки ---
sns.set_theme(style="whitegrid")
FIG_DIR = "results/figures"
TAB_DIR = "results/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Загрузка данных
# ----------------------------------------------------------------------
def load_gcr_results() -> dict:
    """Загружает gcr_only.json."""
    path = "results/gcr_only_2.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} не найден. Запустите run_experiment.py.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sweep_results() -> pd.DataFrame | None:
    """Загружает gcr_sweep_no_ragas.csv (если есть)."""
    path = "results/gcr_sweep_no_ragas.csv"
    if not os.path.exists(path):
        print(f"⚠ {path} не найден. Графики sweep будут пропущены.")
        return None
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# Таблицы
# ----------------------------------------------------------------------
def generate_tables(gcr_data: dict, sweep_df: pd.DataFrame | None):
    """Сохраняет CSV-таблицы для отчёта."""
    # Таблица 1: метрики GCR
    metrics = gcr_data["gcr_metrics"]
    table1 = pd.DataFrame({
        "Metric": [
            "Faithfulness", "Answer Relevancy",
            "Context Precision", "Hallucination Rate",
        ],
        "GCR (GraphRAG)": [
            metrics["faithfulness"],
            metrics["answer_relevancy"],
            metrics["context_precision"],
            metrics["hallucination_rate"],
        ],
    })
    table1["Value"] = table1["GCR (GraphRAG)"].apply(lambda x: f"{x:.4f}")
    table1.to_csv(f"{TAB_DIR}/table1_gcr_metrics.csv", index=False)
    print(f"  ✓ {TAB_DIR}/table1_gcr_metrics.csv")

    # Таблица 2: timing и rejection stats
    table2 = pd.DataFrame([{
        "GCR Time (s)": gcr_data.get("gcr_time", 0),
        "Total Queries": len(gcr_data.get("gcr_metadata", [])),
        "Rejected Queries": gcr_data.get("gcr_rejected", 0),
        "Rejection Rate (%)": round(
            gcr_data.get("gcr_rejected", 0)
            / max(len(gcr_data.get("gcr_metadata", [])), 1) * 100, 1
        ),
    }])
    table2.to_csv(f"{TAB_DIR}/table2_timing_stats.csv", index=False)
    print(f"  ✓ {TAB_DIR}/table2_timing_stats.csv")

    # Таблица 3 (если есть sweep): сравнение конфигураций
    if sweep_df is not None:
        sweep_df.to_csv(f"{TAB_DIR}/table3_sweep_configs.csv", index=False)
        print(f"  ✓ {TAB_DIR}/table3_sweep_configs.csv")


# ----------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------
def plot_gcr_metrics_bar(gcr_data: dict):
    """Fig 1: столбчатая диаграмма метрик GCR."""
    metrics = gcr_data["gcr_metrics"]
    names = ["Faithfulness", "Answer Relevancy", "Context Precision", "Hallucination Rate"]
    values = [
        metrics["faithfulness"],
        metrics["answer_relevancy"],
        metrics["context_precision"],
        metrics["hallucination_rate"],
    ]
    colors = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{v:.3f}", ha="center", fontsize=11, fontweight="bold",
        )
    ax.set_ylabel("Score")
    ax.set_title("GCR: RAGAS Metrics", fontsize=14)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig1_gcr_metrics.png", dpi=300)
    plt.close()
    print(f"  ✓ {FIG_DIR}/fig1_gcr_metrics.png")


def plot_gcr_components_scatter(gcr_data: dict):
    """Fig 2: Scatter Se vs Sc, окраска по accepted/rejected."""
    metadata = gcr_data.get("gcr_metadata", [])
    rows = []
    for m in metadata:
        if not isinstance(m, dict):
            continue
        if m.get("gcr_se") is None:
            continue
        rows.append({
            "Se": m.get("gcr_se", 0),
            "Sc": m.get("gcr_sc", 0),
            "GCR": m.get("gcr", 0),
            "Status": "Rejected" if m.get("rejected") else "Accepted",
        })
    if not rows:
        print("  ⚠ Нет данных для fig2")
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df, x="Se", y="Sc", hue="Status",
        palette={"Accepted": "#2ca02c", "Rejected": "#d62728"},
        s=100, alpha=0.7, edgecolor="white", ax=ax,
    )
    ax.set_xlabel("Entity Overlap (Se)")
    ax.set_ylabel("Structural Connectivity (Sc)")
    ax.set_title("GCR Components: Se vs Sc", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig2_gcr_components.png", dpi=300)
    plt.close()
    print(f"  ✓ {FIG_DIR}/fig2_gcr_components.png")


def plot_gcr_distribution(gcr_data: dict):
    """Fig 3: гистограмма распределения GCR."""
    metadata = gcr_data.get("gcr_metadata", [])
    values = [
        m.get("gcr") for m in metadata
        if isinstance(m, dict) and m.get("gcr") is not None
    ]
    if not values:
        print("  ⚠ Нет данных для fig3")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(values, bins=20, kde=True, color="purple", ax=ax)
    ax.axvline(0.10, color="red", linestyle="--", label="Threshold (0.10)")
    ax.set_xlabel("GCR Score")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Distribution of GCR Scores", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig3_gcr_distribution.png", dpi=300)
    plt.close()
    print(f"  ✓ {FIG_DIR}/fig3_gcr_distribution.png")


def plot_rejection_pie(gcr_data: dict):
    """Fig 4: круговая диаграмма accepted vs rejected."""
    metadata = gcr_data.get("gcr_metadata", [])
    total = len(metadata)
    rejected = gcr_data.get("gcr_rejected", 0)
    accepted = total - rejected
    if total == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        [accepted, rejected],
        labels=["Accepted", "Rejected"],
        colors=["#2ca02c", "#d62728"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax.set_title(f"GCR Context Acceptance (Total: {total})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig4_rejection_pie.png", dpi=300)
    plt.close()
    print(f"  ✓ {FIG_DIR}/fig4_rejection_pie.png")


def plot_sweep_acceptance(sweep_df: pd.DataFrame):
    """Fig 5: bar chart acceptance_rate по конфигурациям."""
    if "acceptance_rate" not in sweep_df.columns:
        return

    df = sweep_df.dropna(subset=["acceptance_rate"]).sort_values(
        "acceptance_rate", ascending=False
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        df["config_name"], df["acceptance_rate"],
        color="#1f77b4", edgecolor="white", linewidth=1.5,
    )
    for bar, v in zip(bars, df["acceptance_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{v*100:.1f}%", ha="center", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Acceptance Rate by GCR Configuration", fontsize=14)
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig5_sweep_acceptance.png", dpi=300)
    plt.close()
    print(f"  ✓ {FIG_DIR}/fig5_sweep_acceptance.png")


def plot_sweep_gcr_components(sweep_df: pd.DataFrame):
    """Fig 6: grouped bar chart avg_se/avg_sc/avg_ph по конфигурациям."""
    required = ["avg_se", "avg_sc", "avg_ph"]
    if not all(c in sweep_df.columns for c in required):
        return

    df = sweep_df.dropna(subset=required).copy()
    if df.empty:
        return

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, df["avg_se"], width, label="Avg Se", color="#1f77b4")
    ax.bar(x, df["avg_sc"], width, label="Avg Sc", color="#ff7f0e")
    ax.bar(x + width, df["avg_ph"], width, label="Avg Ph", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config_name"], rotation=30, ha="right")
    ax.set_ylabel("Average Component Value")
    ax.set_title("Average GCR Components by Configuration", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig6_sweep_components.png", dpi=300)
    plt.close()
    print(f"  ✓ {FIG_DIR}/fig6_sweep_components.png")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("ВИЗУАЛИЗАЦИЯ GCR")
    print("=" * 70)

    # Загрузка
    try:
        gcr_data = load_gcr_results()
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return

    sweep_df = load_sweep_results()

    # Таблицы
    print("\n[Таблицы]")
    generate_tables(gcr_data, sweep_df)

    # Графики
    print("\n[Графики]")
    plot_gcr_metrics_bar(gcr_data)
    plot_gcr_components_scatter(gcr_data)
    plot_gcr_distribution(gcr_data)
    plot_rejection_pie(gcr_data)

    if sweep_df is not None:
        plot_sweep_acceptance(sweep_df)
        plot_sweep_gcr_components(sweep_df)

    print("\n" + "=" * 70)
    print("ГОТОВО")
    print("=" * 70)
    print(f"  Графики: {FIG_DIR}/")
    print(f"  Таблицы: {TAB_DIR}/")


if __name__ == "__main__":
    main()