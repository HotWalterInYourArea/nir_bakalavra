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


def step1_setup_graph():
    logger.info("=" * 70)
    logger.info("ШАГ 1: Загрузка DBpedia в Neo4j + векторный индекс")
    logger.info("=" * 70)

    loader = HybridDBpediaLoader(entity_limit=1000)
    try:
        loader.run()
    finally:
        loader.close()



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

    return baseline_metrics, gcr_metrics


def step6_statistical_tests(baseline_data, gcr_data):
    logger.info("=" * 70)
    logger.info("ШАГ 6: Статистические тесты")
    logger.info("=" * 70)

    baseline_lens = [len(a) for a in baseline_data["answers"]]
    gcr_lens = [len(a) for a in gcr_data["answers"]]

    try:
        stat, p = stats.wilcoxon(baseline_lens, gcr_lens)
        logger.info("Wilcoxon (длина ответов): stat=%.2f, p=%.4f", stat, p)
    except Exception as e:
        logger.warning("Wilcoxon не удался: %s", e)
        stat, p = None, None

    return {
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
    }

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
    if stats_result.get("wilcoxon_p") is not None:
        print(f"Wilcoxon (длина ответов): stat={stats_result['wilcoxon_stat']:.2f}, p={stats_result['wilcoxon_p']:.4f}")
    else:
        print("Wilcoxon: не удалось вычислить")

    comparison.to_csv(f"{CONFIG.results_dir}/comparison_llama.csv", index=False)

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
    with open(f"{CONFIG.results_dir}/full_results_llama_3.2.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Результаты сохранены в %s/", CONFIG.results_dir)


def main():
    t_start = time.time()
    step1_setup_graph()

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

        b_answers, b_contexts, b_meta, b_time = step3_run_baseline(driver, qa_pairs)
        baseline_data = {
            "answers": b_answers,
            "contexts": b_contexts,
            "metadata": b_meta,
            "time": b_time,
        }

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

    baseline_metrics, gcr_metrics = step5_evaluate(qa_pairs, baseline_data, gcr_data)

    stats_result = step6_statistical_tests(baseline_data, gcr_data)

    step7_save_results(baseline_metrics, gcr_metrics, baseline_data, gcr_data, stats_result)

    logger.info("Эксперимент завершён за %.1f сек", time.time() - t_start)


if __name__ == "__main__":
    main()