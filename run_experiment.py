from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import neo4j
import pandas as pd
import numpy as np
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


# ----------------------------------------------------------------------
# Шаг 0: Калибровка GCR
# ----------------------------------------------------------------------
def step0_calibrate_gcr(driver, qa_pairs):
    """Калибровка порога GCR на первых N примерах."""
    logger.info("=" * 70)
    logger.info("ШАГ 0: Калибровка GCR")
    logger.info("=" * 70)

    from baseline_vector_rag import BaselineVectorRAG
    from gcr import Neo4jGraphLoader, GraphContextRelevance

    baseline = BaselineVectorRAG(driver)
    loader = Neo4jGraphLoader(
        CONFIG.neo4j.uri, (CONFIG.neo4j.user, CONFIG.neo4j.password), CONFIG.neo4j.database,
    )
    G = loader.load_graph(node_label="Resource")
    gcr = GraphContextRelevance(G)

    scores = []
    for pair in qa_pairs[:20]:
        result = baseline.retriever.search(query_text=pair.question, top_k=5)
        # В новой версии neo4j-graphrag URI хранится в metadata['id']
        uris = [
            item.metadata.get("id") for item in result.items
            if item.metadata and item.metadata.get("id")
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
        logger.info("GCR распределение: min=%.3f median=%.3f max=%.3f",
                    np.min(scores), np.median(scores), np.max(scores))
        suggested = float(np.median(scores)) * 0.7
        logger.info("Рекомендуемый порог: %.3f", suggested)
    else:
        logger.warning("Не удалось собрать GCR-скоры для калибровки.")


# ----------------------------------------------------------------------
# Шаг 1
# ----------------------------------------------------------------------
def step1_setup_graph():
    logger.info("=" * 70)
    logger.info("ШАГ 1: Загрузка DBpedia в Neo4j + векторный индекс")
    logger.info("=" * 70)

    loader = HybridDBpediaLoader(entity_limit=1000)
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
        pairs = EvalDatasetBuilder.load(dataset_path)
        # Если датасет уже есть, но в нём меньше 100 пар — перегенерируем
        if len(pairs) >= 100:
            return pairs
        logger.warning("Датасет содержит только %d пар. Перегенерируем.", len(pairs))
        Path(dataset_path).unlink()

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
        # Увеличиваем лимит до 1000, чтобы захватить больше abstracts
        abstracts = builder.fetch_abstracts(limit=1000)
        logger.info("Найдено %d abstracts", len(abstracts))

        # Генерируем до 200 QA-пар (или сколько позволяет датасет)
        pairs = builder.generate_qa_pairs(abstracts, n=200)
        builder.save(pairs, dataset_path)
        return pairs
    finally:
        driver.close()


# ----------------------------------------------------------------------
# Шаг 3
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
    baseline_result = evaluator.evaluate(
        questions,
        baseline_data["answers"],
        baseline_data["contexts"],
        ground_truths,
    )
    logger.info("Baseline: %s", baseline_result.to_dict())

    logger.info("Оценка GCR...")
    gcr_result = evaluator.evaluate(
        questions,
        gcr_data["answers"],
        gcr_data["contexts"],
        ground_truths,
    )
    logger.info("GCR: %s", gcr_result.metrics.to_dict())

    # Возвращаем DataFrame с per-sample метриками для Wilcoxon
    return baseline_result.df, gcr_result.df, baseline_result.metrics, gcr_result.metrics


# ----------------------------------------------------------------------
# Шаг 6
# ----------------------------------------------------------------------
def step6_statistical_tests(baseline_df: pd.DataFrame, gcr_df: pd.DataFrame):
    """Wilcoxon по per-sample метрикам RAGAS."""
    logger.info("=" * 70)
    logger.info("ШАГ 6: Статистические тесты")
    logger.info("=" * 70)

    results = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        b = baseline_df[metric].values
        g = gcr_df[metric].values

        mask = ~(pd.isna(b) | pd.isna(g))
        b, g = b[mask], g[mask]

        if len(b) < 5:
            results[metric] = {"stat": None, "p": None, "n": len(b)}
            continue

        try:
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
        except Exception as e:
            logger.warning("Wilcoxon для %s не удался: %s", metric, e)
            results[metric] = {"stat": None, "p": None, "n": len(b)}

    # Дополнительно считаем rejection stats
    results["rejection"] = {
        "baseline_insufficient": sum(
            1 for a in baseline_df.get("answer", []) if "insufficient" in str(a).lower()
        ) if "answer" in baseline_df.columns else 0,
        "gcr_insufficient": sum(
            1 for a in gcr_df.get("answer", []) if "insufficient" in str(a).lower()
        ) if "answer" in gcr_df.columns else 0,
    }

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

    # Красивый вывод Wilcoxon
    for metric, res in stats_result.items():
        if metric == "rejection":
            continue
        if res.get("p") is not None:
            print(f"Wilcoxon {metric}: p={res['p']:.4f} (n={res['n']})")

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
    try:
        step0_calibrate_gcr(driver, qa_pairs)

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
    baseline_df, gcr_df, baseline_metrics, gcr_metrics = step5_evaluate(
        qa_pairs, baseline_data, gcr_data
    )

    # Шаг 6: статистика
    stats_result = step6_statistical_tests(baseline_df, gcr_df)

    # Шаг 7: сохранение
    step7_save_results(baseline_metrics, gcr_metrics, baseline_data, gcr_data, stats_result)

    logger.info("Эксперимент завершён за %.1f сек", time.time() - t_start)


if __name__ == "__main__":
    main()