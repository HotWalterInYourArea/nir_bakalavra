from __future__ import annotations

import logging
from dataclasses import dataclass

import neo4j
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import VectorRetriever

from config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class RAGOutput:
    answer: str
    contexts: list
    retriever_metadata: dict


PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the context is insufficient to answer, respond exactly with: "Insufficient context."
Do not use any outside knowledge.

Context:
{context}

Question: {question}

Answer:"""


class BaselineVectorRAG:
    def __init__(self, driver: neo4j.Driver):
        self.driver = driver
        self.embedder = SentenceTransformerEmbeddings(
            model=CONFIG.embedding.model,
        )
        self.llm = OllamaLLM(
            model_name=CONFIG.llm.model,
            model_params={"temperature": CONFIG.llm.temperature},
            host=CONFIG.llm.ollama_base_url,
        )
        self.retriever = VectorRetriever(
            driver=driver,
            index_name=CONFIG.embedding.index,
            embedder=self.embedder,
            return_properties=["uri", "text"],
        )

    def search(self, query: str, top_k: int = 5) -> RAGOutput:
        result = self.retriever.search(query_text=query, top_k=top_k)
        contexts = [item.content for item in result.items]
        contexts = [c for c in contexts if c]

        if not contexts:
            return RAGOutput(
                answer="Insufficient context.",
                contexts=[],
                retriever_metadata={"retriever": "vector", "empty": True},
            )

        prompt = PROMPT_TEMPLATE.format(
            context="\n\n---\n\n".join(contexts),
            question=query,
        )
        resp = self.llm.invoke(prompt)
        answer = resp.content if hasattr(resp, "content") else str(resp)

        return RAGOutput(
            answer=answer.strip(),
            contexts=contexts,
            retriever_metadata={"retriever": "vector"},
        )