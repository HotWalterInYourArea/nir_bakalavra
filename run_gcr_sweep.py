from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import neo4j
import pandas as pd

from config import CONFIG
from eval_dataset import EvalDatasetBuilder
from gcr_graph_rag import GCRGraphRAG
from gcr import Neo4jGraphLoader, GraphContextRelevance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sweep")


# ----------------------------------------------------------------------
# Конфигурации для прогона
# ----------------------------------------------------------------------
CONFIGS = [
    {"name": "default",            "alpha": 0.4, "beta": 0.4, "gamma": 0.2, "threshold": 0.15},
    {"name": "optimal_from_grid",  "alpha": 0.6, "beta": 0.6, "gamma": 0.1, "threshold": 0.10},
    {"name": "soft_filter",        "alpha": 0.5, "beta": 0.3, "gamma": 0.2, "threshold": 0.05},
    {"name": "hard_filter",        "alpha": 0.4, "beta": 0.6, "gamma": 0.2, "threshold": 0.20},
    {"name": "semantic_only",      "alpha": 0.8, "beta": 0.1, "gamma": 0.1, "threshold": 0.10},
    {"name": "structural_only",    "alpha": 0.1, "beta": 0.8, "gamma": 0.1, "threshold": 0.10},
    {"name": "strict",             "alpha": 0.5, "beta": 0.5, "gamma": 0.3, "threshold": 0.20},
    {"name": "semantic_priority",  "alpha": 0.5, "beta": 0.3, "gamma": 0.2, "threshold": 0.08},
    {"name": "structure_priority",  "alpha": 0.3, "beta": 0.5, "gamma": 0.2, "threshold": 0.08},
]

# Сколько вопросов использовать для sweep (можно увеличить, это быстро)
SWEEP_SAMPLE_SIZE = 100


def apply_config(cfg: dict):
    """Применяет параметры к CONFIG.gcr."""
    CONFIG.gcr.alpha = cfg["alpha"]
    CONFIG.gcr.beta = cfg["beta"]
    CONFIG.gcr.gamma = cfg["gamma"]
    CONFIG.gcr.threshold = cfg["threshold"]


def run_gcr_sweep(driver, qa_pairs: list, cfg: dict, gcr_engine, graph) -> dict:

    apply_config(cfg)

    logger.info("=" * 70)
    logger.info("ПРОГОН: %s (α=%.2f, β=%.2f, γ=%.2f, thr=%.2f)",
                cfg["name"], cfg["alpha"], cfg["beta"], cfg["gamma"], cfg["threshold"])
    logger.info("=" * 70)

    # Прямой доступ к GCR-ретриверу без LLM
    from baseline_vector_rag import BaselineVectorRAG
    from gcr_retriever import GCRFilterRetriever

    baseline = BaselineVectorRAG(driver)
    gcr_retriever = GCRFilterRetriever(
        vector_retriever=baseline.retriever,
        graph_loader=driver,  # не используется, граф уже загружен
        gcr_threshold=cfg["threshold"],
        enable_expansion=CONFIG.gcr.enable_expansion,
        expansion_hops=CONFIG.gcr.expansion_hops,
        max_expanded_nodes=CONFIG.gcr.max_expanded_nodes,
        graph=graph,             
        gcr_engine=gcr_engine,
    )


    # Собираем метрики
    accepted = 0
    rejected = 0
    total_gcr = 0.0
    total_se = 0.0
    total_sc = 0.0
    total_ph = 0.0
    total_context_size = 0
    times = []

    for i, pair in enumerate(qa_pairs):
        t0 = time.time()
        try:
            result = gcr_retriever.search(query_text=pair.question, top_k=CONFIG.top_k)
            elapsed = time.time() - t0
            times.append(elapsed)

            meta = result.metadata or {}
            gcr_score = meta.get("gcr", 0.0)
            total_gcr += gcr_score
            total_se += meta.get("gcr_se", 0.0)
            total_sc += meta.get("gcr_sc", 0.0)
            total_ph += meta.get("gcr_ph", 0.0)

            context_size = len(result.items) if result.items else 0
            total_context_size += context_size

            if meta.get("rejected"):
                rejected += 1
            else:
                accepted += 1
        except Exception as e:
            logger.error("[%d] error: %s", i, e)
            times.append(0.0)

        if (i + 1) % 20 == 0:
            logger.info("  %d/%d (accepted=%d, rejected=%d)",
                        i + 1, len(qa_pairs), accepted, rejected)

    n = len(qa_pairs)
    return {
        "config_name": cfg["name"],
        "alpha": cfg["alpha"],
        "beta": cfg["beta"],
        "gamma": cfg["gamma"],
        "threshold": cfg["threshold"],
        "acceptance_rate": accepted / n if n else 0.0,
        "rejected_count": rejected,
        "accepted_count": accepted,
        "avg_gcr": total_gcr / n if n else 0.0,
        "avg_se": total_se / n if n else 0.0,
        "avg_sc": total_sc / n if n else 0.0,
        "avg_ph": total_ph / n if n else 0.0,
        "avg_context_size": total_context_size / n if n else 0.0,
        "avg_time_per_query_ms": (sum(times) / len(times) * 1000) if times else 0.0,
        "total_time_sec": sum(times),
        "n_questions": n,
    }


def main():
    t_start = time.time()

    # --- Загрузка датасета ---
    dataset_path = "eval_dataset.json"
    if not Path(dataset_path).exists():
        logger.error("Датасет не найден. Сначала запустите run_experiment.py для генерации.")
        return

    all_pairs = EvalDatasetBuilder.load(dataset_path)
    logger.info("Загружено %d QA-пар", len(all_pairs))

    sweep_pairs = all_pairs[:SWEEP_SAMPLE_SIZE]
    logger.info("Используем %d пар для sweep", len(sweep_pairs))

    # --- Загрузка графа ОДИН раз ---
    logger.info("Загрузка графа в память...")
    loader = Neo4jGraphLoader(
        uri=CONFIG.neo4j.uri,
        auth=(CONFIG.neo4j.user, CONFIG.neo4j.password),
        database=CONFIG.neo4j.database,
    )
    graph = loader.load_graph(node_label="Resource")
    logger.info("Граф загружен: %d узлов, %d рёбер",
                graph.number_of_nodes(), graph.number_of_edges())

    # --- Драйвер Neo4j ---
    driver = neo4j.GraphDatabase.driver(
        CONFIG.neo4j.uri,
        auth=(CONFIG.neo4j.user, CONFIG.neo4j.password),
    )

    results = []
    try:
        for cfg in CONFIGS:
            try:
                # Создаём движок GCR с параметрами этой конфигурации
                from gcr import GCRConfig
                gcr_cfg = GCRConfig(
                    alpha=cfg["alpha"],
                    beta=cfg["beta"],
                    gamma=cfg["gamma"],
                    hub_sigma=CONFIG.gcr.hub_sigma,
                )
                gcr_engine = GraphContextRelevance(
                    graph=graph,
                    config=gcr_cfg,
                    node_text_attr="text",
                )

                result = run_gcr_sweep(driver, sweep_pairs, cfg, gcr_engine, graph)
                results.append(result)
                logger.info(
                    "  ✓ %s: accept=%.1f%%, avg_GCR=%.3f, ctx=%.1f, time=%.1fms",
                    cfg["name"],
                    result["acceptance_rate"] * 100,
                    result["avg_gcr"],
                    result["avg_context_size"],
                    result["avg_time_per_query_ms"],
                )
            except Exception as e:
                logger.error("  ✗ %s упал: %s", cfg["name"], e)
                results.append({
                    "config_name": cfg["name"],
                    "alpha": cfg["alpha"], "beta": cfg["beta"],
                    "gamma": cfg["gamma"], "threshold": cfg["threshold"],
                    "error": str(e),
                })
    finally:
        driver.close()
        loader.close()

    # --- Сохранение ---
    Path(CONFIG.results_dir).mkdir(exist_ok=True)
    df = pd.DataFrame(results)

    # Сортируем по acceptance_rate (или по avg_gcr — на ваш выбор)
    if "acceptance_rate" in df.columns:
        df = df.sort_values("acceptance_rate", ascending=False)

    csv_path = f"{CONFIG.results_dir}/gcr_sweep_no_ragas.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Сводная таблица: %s", csv_path)

    json_path = f"{CONFIG.results_dir}/gcr_sweep_no_ragas.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("JSON: %s", json_path)

    # --- Красивый вывод ---
    print("\n" + "=" * 100)
    print("СРАВНЕНИЕ КОНФИГУРАЦИЙ GCR (без RAGAS)")
    print("=" * 100)

    display_cols = [
        "config_name", "alpha", "beta", "gamma", "threshold",
        "acceptance_rate", "avg_gcr", "avg_se", "avg_sc", "avg_ph",
        "avg_context_size", "avg_time_per_query_ms",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    display_df = df[existing_cols].copy()

    # Форматирование
    if "acceptance_rate" in display_df.columns:
        display_df["acceptance_rate"] = display_df["acceptance_rate"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"
        )
    for col in ["avg_gcr", "avg_se", "avg_sc", "avg_ph"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "-"
            )
    if "avg_context_size" in display_df.columns:
        display_df["avg_context_size"] = display_df["avg_context_size"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "-"
        )
    if "avg_time_per_query_ms" in display_df.columns:
        display_df["avg_time_per_query_ms"] = display_df["avg_time_per_query_ms"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "-"
        )

    print(display_df.to_string(index=False))

    # --- Лучшая конфигурация ---
    print("\n" + "=" * 100)
    print("ИНТЕРПРЕТАЦИЯ")
    print("=" * 100)

    valid = [r for r in results if "error" not in r]
    if valid:
        # Баланс: высокий acceptance + высокий avg_gcr
        best_balance = max(valid, key=lambda r: r["acceptance_rate"] * r["avg_gcr"])
        print(f"\n  Лучший баланс (acceptance × avg_gcr): {best_balance['config_name']}")
        print(f"    acceptance = {best_balance['acceptance_rate']*100:.1f}%")
        print(f"    avg_gcr    = {best_balance['avg_gcr']:.3f}")

        # Максимальный avg_gcr
        best_gcr = max(valid, key=lambda r: r["avg_gcr"])
        print(f"\n  Максимальный avg_gcr: {best_gcr['config_name']} ({best_gcr['avg_gcr']:.3f})")

        # Минимальное время
        best_time = min(valid, key=lambda r: r["avg_time_per_query_ms"])
        print(f"\n  Самая быстрая: {best_time['config_name']} "
              f"({best_time['avg_time_per_query_ms']:.0f} ms/запрос)")

    logger.info("\nГотово за %.1f сек", time.time() - t_start)


if __name__ == "__main__":
    main()