from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import neo4j
import pandas as pd

from config import CONFIG
from dbpedia_loader import HybridDBpediaLoader
from eval_dataset import EvalDatasetBuilder
from gcr_graph_rag import GCRGraphRAG
from evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("experiment")


# ----------------------------------------------------------------------
# Шаг 1: Загрузка графа
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
# Шаг 2: QA-датасет
# ----------------------------------------------------------------------
def step2_build_dataset():
    logger.info("=" * 70)
    logger.info("ШАГ 2: Генерация QA-датасета")
    logger.info("=" * 70)

    dataset_path = "eval_dataset.json"
    if Path(dataset_path).exists():
        logger.info("Загрузка существующего датасета: %s", dataset_path)
        pairs = EvalDatasetBuilder.load(dataset_path)
        if len(pairs) >= 20:
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
        abstracts = builder.fetch_abstracts(limit=1000)
        logger.info("Найдено %d abstracts", len(abstracts))
        pairs = builder.generate_qa_pairs(abstracts, n=CONFIG.eval_dataset_size)
        builder.save(pairs, dataset_path)
        return pairs
    finally:
        driver.close()


# ----------------------------------------------------------------------
# Шаг 3: ТОЛЬКО GCR (без baseline)
# ----------------------------------------------------------------------
def step3_run_gcr_only(driver, qa_pairs):
    logger.info("=" * 70)
    logger.info("ШАГ 3: GCR GraphRAG (без baseline)")
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
# Шаг 4: RAGAS-оценка только GCR
# ----------------------------------------------------------------------
def step4_evaluate_gcr(qa_pairs, gcr_data):
    logger.info("=" * 70)
    logger.info("ШАГ 4: RAGAS-оценка GCR")
    logger.info("=" * 70)

    questions = [p.question for p in qa_pairs]
    ground_truths = [p.ground_truth for p in qa_pairs]

    evaluator = Evaluator()

    logger.info("Оценка GCR...")
    gcr_metrics = evaluator.evaluate(
        questions,
        gcr_data["answers"],
        gcr_data["contexts"],
        ground_truths,
    )
    logger.info("GCR: %s", gcr_metrics.to_dict())

    return gcr_metrics



def step5_save_results(gcr_metrics, gcr_data):
    Path(CONFIG.results_dir).mkdir(exist_ok=True)

    # Таблица с метриками GCR
    results_df = pd.DataFrame({
        "Metric": ["Faithfulness", "Answer Relevancy", "Context Precision", "Hallucination Rate"],
        "GCR (GraphRAG)": [
            gcr_metrics.faithfulness,
            gcr_metrics.answer_relevancy,
            gcr_metrics.context_precision,
            gcr_metrics.hallucination_rate,
        ],
    })

    print("\n" + "=" * 75)
    print("РЕЗУЛЬТАТЫ GCR (без baseline)")
    print("=" * 75)
    print(results_df.to_string(index=False))
    print("=" * 75)
    print(f"\nGCR time:      {gcr_data['time']:.1f}s")
    print(f"GCR отбраковано: {gcr_data['rejected']}/{len(gcr_data['answers'])}")

    results_df.to_csv(f"{CONFIG.results_dir}/gcr_only.csv", index=False)

    full_results = {
        "gcr_metrics": gcr_metrics.to_dict(),
        "gcr_time": gcr_data["time"],
        "gcr_rejected": gcr_data["rejected"],
        "gcr_metadata": gcr_data["metadata"],
    }
    with open(f"{CONFIG.results_dir}/gcr_only_12.json", "w", encoding="utf-8") as f:
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
        g_answers, g_contexts, g_meta, g_time, rejected = step3_run_gcr_only(driver, qa_pairs)
        gcr_data = {
            "answers": g_answers,
            "contexts": g_contexts,
            "metadata": g_meta,
            "time": g_time,
            "rejected": rejected,
        }
    finally:
        driver.close()

    gcr_metrics = step4_evaluate_gcr(qa_pairs, gcr_data)
    step5_save_results(gcr_metrics, gcr_data)

    logger.info("Эксперимент завершён за %.1f сек", time.time() - t_start)


if __name__ == "__main__":
    main()