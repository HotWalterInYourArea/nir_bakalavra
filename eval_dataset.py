from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass

import neo4j

from config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    question: str
    ground_truth: str
    source_uri: str
    source_text: str


GENERATION_PROMPT = """Based ONLY on the text below, generate 1 high-quality question-answer pair.

Requirements:
- The question must be answerable from the text.
- The answer must be concise (1-2 sentences) and factual.
- Return ONLY valid JSON, no explanations: {{"question": "...", "answer": "..."}}

Text:
{text}

JSON:"""


class EvalDatasetBuilder:
    def __init__(self, driver: neo4j.Driver, llm):
        self.driver = driver
        self.llm = llm

    def fetch_abstracts(self, limit: int = 500) -> list:
        query = """
        MATCH (n:Resource)
        WHERE coalesce(n.text, n.abstract) IS NOT NULL
          AND size(coalesce(n.text, n.abstract)) > 200
        RETURN coalesce(n.uri, toString(id(n))) AS uri,
               coalesce(n.text, n.abstract) AS text
        ORDER BY rand()
        LIMIT $limit
        """
        with self.driver.session(database=CONFIG.neo4j.database) as s:
            return [r.data() for r in s.run(query, limit=limit)]

    def generate_qa_pairs(self, abstracts: list, n: int = 100) -> list:
        random.seed(CONFIG.seed)
        random.shuffle(abstracts)
        pairs = []

        for i, abstract in enumerate(abstracts):
            if len(pairs) >= n:
                break

            prompt = GENERATION_PROMPT.format(text=abstract["text"][:2000])
            try:
                resp = self.llm.invoke(prompt)
                content = resp.content if hasattr(resp, "content") else str(resp)
                content = self._extract_json(content)

                data = json.loads(content)
                if "question" not in data or "answer" not in data:
                    continue

                pairs.append(QAPair(
                    question=data["question"].strip(),
                    ground_truth=data["answer"].strip(),
                    source_uri=abstract["uri"],
                    source_text=abstract["text"][:2000],
                ))
                logger.info("[%d/%d] %s", len(pairs), n, data["question"][:60])
            except Exception as e:
                logger.debug("Ошибка генерации: %s", e)

        logger.info("Сгенерировано %d QA-пар", len(pairs))
        return pairs

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
        # Ищем первую { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start:end + 1]
        return text

    @staticmethod
    def save(pairs: list, path: str = "eval_dataset.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([p.__dict__ for p in pairs], f, ensure_ascii=False, indent=2)
        logger.info("Сохранено: %s", path)

    @staticmethod
    def load(path: str = "eval_dataset.json") -> list:
        with open(path, "r", encoding="utf-8") as f:
            return [QAPair(**d) for d in json.load(f)]