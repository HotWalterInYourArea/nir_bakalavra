from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import neo4j
import pandas as pd
from scipy import stats

from config import CONFIG
from dbpedia_loader import HybridDBpediaLoader
from eval_dataset import EvalDatasetBuilder
from baseline_vector_rag import BaselineVectorRAG
from gcr_graph_rag import GCRGraphRAG
from evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("experiment")

def step0_calibrate_gcr(driver, qa_pairs):
    """Калибровка порога GCR на первых N примерах."""
    logger.info("=" * 70)
    logger.info("ШАГ 0: Калибровка GCR")
    logger.info("=" * 70)

    from baseline_vector_rag import BaselineVectorRAG
    from gcr import Neo4jGraphLoader, GraphContextRelevance
    from config import CONFIG as C

    baseline = BaselineVectorRAG(driver)
    loader = Neo4jGraphLoader(
        C.neo4j.uri, (C.neo4j.user, C.neo4j.password), C.neo4j.database,
    )
    G = loader.load_graph(node_label="Resource")
    gcr = GraphContextRelevance(G)

    scores = []
    for pair in qa_pairs[:20]:
        result = baseline.retriever.search(query_text=pair.question, top_k=5)
        uris = [
            item.metadata.get("uri") for item in result.items
            if item.metadata and item.metadata.get("uri")
        ]
        if not uris:
            continue
        r = gcr.score(uris, pair.question, pair.ground_truth)
        scores.append(r.gcr)
        logger.info(
            "  Q: %s → GCR=%.3f (Se=%.3f Sc=%.3f Ph=%.3f)",
            pair.question[:50], r.gcr, r.entity_overlap,
            r.structural_connectivity, r.hub_noise_penalty,
        )

    loader.close()

    if scores:
        import numpy as np
        logger.info("GCR распределение: min=%.3f median=%.3f max=%.3f",
                    np.min(scores), np.median(scores), np.max(scores))
        suggested = float(np.median(scores)) * 0.7
        logger.info("Рекомендуемый порог: %.3f", suggested)
# ----------------------------------------------------------------------
# Шаг 1
# ----------------------------------------------------------------------
def step1_setup_graph():
    logger.info("=" * 70)
    logger.info("ШАГ 1: Загрузка DBpedia в Neo4j + векторный индекс")
    logger.info("=" * 70)

    loader = HybridDBpediaLoader(entity_limit=100)
    try:
        loader.run()
    finally:
        loader.close()


# ----------------------------------------------------------------------
# Шаг 2
# ----------------------------------------------------------------------
def step2_build_dataset():
    logger.info("=" * 70)
    logger.info("ШАГ 2: Генерация QA-датасета")
    logger.info("=" * 70)

    dataset_path = "eval_dataset.json"
    if Path(dataset_path).exists():
        logger.info("Загрузка существующего датасета: %s", dataset_path)
        return EvalDatasetBuilder.load(dataset_path)

    from neo4j_graphrag.llm import OllamaLLM
    llm = OllamaLLM(
        model_name=CONFIG.llm.model,
        model_params={"temperature": 0.0},
        host=CONFIG.llm.ollama_base_url,
    )

    driver = neo4j.GraphDatabase.driver(
        CONFIG.neo4j.uri,
        auth=(CONFIG.neo4j.user, CONFIG.neo4j.password),
    )
    try:
        builder = EvalDatasetBuilder(driver, llm)
        abstracts = builder.fetch_abstracts(limit=500)
        logger.info("Найдено %d abstracts", len(abstracts))

        pairs = builder.generate_qa_pairs(abstracts, n=CONFIG.eval_dataset_size)
        builder.save(pairs, dataset_path)
        return pairs
    finally:
        driver.close()


# ----------------------------------------------------------------------
# Шаг 3-4
# ----------------------------------------------------------------------
def step3_run_baseline(driver, qa_pairs):
    logger.info("=" * 70)
    logger.info("ШАГ 3: BASELINE — Vector RAG")
    logger.info("=" * 70)

    baseline = BaselineVectorRAG(driver)
    answers, contexts = [], []
    metadata_list = []

    t0 = time.time()
    for i, pair in enumerate(qa_pairs):
        try:
            out = baseline.search(pair.question, top_k=CONFIG.top_k)
            answers.append(out.answer)
            contexts.append(out.contexts)
            metadata_list.append(out.retriever_metadata)
        except Exception as e:
            logger.error("[%d] Baseline error: %s", i, e)
            answers.append("Error")
            contexts.append([])
            metadata_list.append({"error": str(e)})

        if (i + 1) % 10 == 0:
            logger.info("Baseline: %d/%d", i + 1, len(qa_pairs))

    elapsed = time.time() - t0
    logger.info("Baseline завершён за %.1f сек", elapsed)

    return answers, contexts, metadata_list, elapsed


def step4_run_gcr(driver, qa_pairs):
    logger.info("=" * 70)
    logger.info("ШАГ 4: EXPERIMENTAL — GraphRAG + GCR")
    logger.info("=" * 70)

    gcr_rag = GCRGraphRAG(driver)
    answers, contexts = [], []
    metadata_list = []
    rejected = 0

    t0 = time.time()
    try:
        for i, pair in enumerate(qa_pairs):
            try:
                out = gcr_rag.search(pair.question, top_k=CONFIG.top_k)
                answers.append(out.answer)
                contexts.append(out.contexts)
                metadata_list.append(out.retriever_metadata)
                if out.retriever_metadata.get("rejected"):
                    rejected += 1
            except Exception as e:
                logger.error("[%d] GCR error: %s", i, e)
                answers.append("Error")
                contexts.append([])
                metadata_list.append({"error": str(e)})

            if (i + 1) % 10 == 0:
                logger.info("GCR: %d/%d (отбраковано %d)", i + 1, len(qa_pairs), rejected)
    finally:
        gcr_rag.close()

    elapsed = time.time() - t0
    logger.info("GCR завершён за %.1f сек, отбраковано %d", elapsed, rejected)

    return answers, contexts, metadata_list, elapsed, rejected


# ----------------------------------------------------------------------
# Шаг 5
# ----------------------------------------------------------------------
def step5_evaluate(qa_pairs, baseline_data, gcr_data):
    logger.info("=" * 70)
    logger.info("ШАГ 5: RAGAS-оценка")
    logger.info("=" * 70)

    questions = [p.question for p in qa_pairs]
    ground_truths = [p.ground_truth for p in qa_pairs]

    evaluator = Evaluator()

    logger.info("Оценка baseline...")
    baseline_metrics = evaluator.evaluate(
        questions,
        baseline_data["answers"],
        baseline_data["contexts"],
        ground_truths,
    )
    logger.info("Baseline: %s", baseline_metrics.to_dict())

    logger.info("Оценка GCR...")
    gcr_metrics = evaluator.evaluate(
        questions,
        gcr_data["answers"],
        gcr_data["contexts"],
        ground_truths,
    )
    logger.info("GCR: %s", gcr_metrics.to_dict())
    baseline_df = baseline_metrics.df  # RAGAS возвращает DataFrame
    gcr_df = gcr_metrics.df

    return baseline_df, gcr_df


# ----------------------------------------------------------------------
# Шаг 6
# ----------------------------------------------------------------------
def step6_statistical_tests(baseline_metrics_df, gcr_metrics_df):
    """Wilcoxon по per-sample метрикам RAGAS."""
    from scipy import stats

    results = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        b = baseline_metrics_df[metric].values
        g = gcr_metrics_df[metric].values

        # Убираем NaN пары
        mask = ~(pd.isna(b) | pd.isna(g))
        b, g = b[mask], g[mask]

        if len(b) < 5:
            results[metric] = {"stat": None, "p": None, "n": len(b)}
            continue

        stat, p = stats.wilcoxon(b, g)
        results[metric] = {
            "stat": float(stat),
            "p": float(p),
            "n": int(len(b)),
            "significant": p < 0.05,
        }
        logger.info("%s: Wilcoxon stat=%.2f, p=%.4f, n=%d, %s",
                    metric, stat, p, len(b),
                    "SIGNIFICANT" if p < 0.05 else "not significant")
    return results

# ----------------------------------------------------------------------
# Шаг 7
# ----------------------------------------------------------------------
def step7_save_results(baseline_metrics, gcr_metrics, baseline_data, gcr_data, stats_result):
    Path(CONFIG.results_dir).mkdir(exist_ok=True)

    comparison = pd.DataFrame({
        "Metric": [
            "Faithfulness", "Answer Relevancy",
            "Context Precision", "Hallucination Rate",
        ],
        "Baseline (Vector RAG)": [
            baseline_metrics.faithfulness,
            baseline_metrics.answer_relevancy,
            baseline_metrics.context_precision,
            baseline_metrics.hallucination_rate,
        ],
        "GCR (GraphRAG)": [
            gcr_metrics.faithfulness,
            gcr_metrics.answer_relevancy,
            gcr_metrics.context_precision,
            gcr_metrics.hallucination_rate,
        ],
    })
    comparison["Delta"] = (
        comparison["GCR (GraphRAG)"] - comparison["Baseline (Vector RAG)"]
    )
    comparison["Delta %"] = (
        comparison["Delta"] / comparison["Baseline (Vector RAG)"] * 100
    ).round(1)

    print("\n" + "=" * 75)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
    print("=" * 75)
    print(comparison.to_string(index=False))
    print("=" * 75)
    print(f"\nBaseline time: {baseline_data['time']:.1f}s")
    print(f"GCR time:      {gcr_data['time']:.1f}s")
    print(f"GCR отбраковано: {gcr_data['rejected']}/{len(baseline_data['answers'])}")
    print(f"Wilcoxon p-value: {stats_result['wilcoxon_p']}")

    comparison.to_csv(f"{CONFIG.results_dir}/comparison.csv", index=False)

    full_results = {
        "baseline_metrics": baseline_metrics.to_dict(),
        "gcr_metrics": gcr_metrics.to_dict(),
        "baseline_time": baseline_data["time"],
        "gcr_time": gcr_data["time"],
        "gcr_rejected": gcr_data["rejected"],
        "stats": stats_result,
        "baseline_metadata": baseline_data["metadata"],
        "gcr_metadata": gcr_data["metadata"],
    }
    with open(f"{CONFIG.results_dir}/full_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Результаты сохранены в %s/", CONFIG.results_dir)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    t_start = time.time()
    # Шаг 1: setup графа
    step1_setup_graph()

    # Шаг 2: датасет
    qa_pairs = step2_build_dataset()
    if len(qa_pairs) < 20:
        logger.error("Слишком мало QA-пар (%d). Выход.", len(qa_pairs))
        return

    driver = neo4j.GraphDatabase.driver(
        CONFIG.neo4j.uri,
        auth=(CONFIG.neo4j.user, CONFIG.neo4j.password),
    )
    step0_calibrate_gcr(
            driver, qa_pairs
        )
    
    try:
        # Шаг 3: baseline
        b_answers, b_contexts, b_meta, b_time = step3_run_baseline(driver, qa_pairs)
        baseline_data = {
            "answers": b_answers,
            "contexts": b_contexts,
            "metadata": b_meta,
            "time": b_time,
        }

        # Шаг 4: GCR
        g_answers, g_contexts, g_meta, g_time, rejected = step4_run_gcr(driver, qa_pairs)
        gcr_data = {
            "answers": g_answers,
            "contexts": g_contexts,
            "metadata": g_meta,
            "time": g_time,
            "rejected": rejected,
        }

    finally:
        driver.close()

    # Шаг 5: RAGAS
    baseline_metrics, gcr_metrics = step5_evaluate(qa_pairs, baseline_data, gcr_data)

    # Шаг 6: статистика
    stats_result = step6_statistical_tests(baseline_data, gcr_data, qa_pairs)

    # Шаг 7: сохранение
    step7_save_results(baseline_metrics, gcr_metrics, baseline_data, gcr_data, stats_result)

    logger.info("Эксперимент завершён за %.1f сек", time.time() - t_start)


if __name__ == "__main__":
    main()