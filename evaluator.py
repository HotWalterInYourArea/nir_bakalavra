from __future__ import annotations

import logging
from dataclasses import dataclass

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    hallucination_rate: float
    n_samples: int

    def to_dict(self):
        return self.__dict__


class Evaluator:
    def __init__(self):
        # Локальная judge-LLM через Ollama
        self.llm = LangchainLLMWrapper(
            ChatOllama(
                model=CONFIG.llm.eval_model,
                base_url=CONFIG.llm.ollama_base_url,
                temperature=0,
            )
        )
        # Локальные эмбеддинги
        self.embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=CONFIG.embedding.model)
        )
        # Ограничиваем параллелизм, чтобы не перегружать Ollama
        self.run_config = RunConfig(
            timeout=180,
            max_workers=2,
            max_retries=3,
        )

    def evaluate(
        self,
        questions: list,
        answers: list,
        contexts: list,
        ground_truths: list,
    ) -> EvalResult:
        # RAGAS требует, чтобы contexts[i] был непустым списком
        # Если контекст пустой — заменяем на пустую строку
        contexts = [c if c else [""] for c in contexts]

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        logger.info("RAGAS-оценка на %d примерах...", len(dataset))
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=self.llm,
            embeddings=self.embeddings,
            run_config=self.run_config,
            raise_exceptions=False,
        )

        df = result.to_pandas()

        faith = float(df["faithfulness"].mean(skipna=True))
        rel = float(df["answer_relevancy"].mean(skipna=True))
        prec = float(df["context_precision"].mean(skipna=True))

        return EvalResult(
            faithfulness=faith,
            answer_relevancy=rel,
            context_precision=prec,
            hallucination_rate=1.0 - faith,
            n_samples=len(df),
        )