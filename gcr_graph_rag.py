"""Experimental: GraphRAG + GCR с локальными моделями."""
from __future__ import annotations

import logging

import neo4j

from config import CONFIG
from gcr import Neo4jGraphLoader
from gcr_retriever import GCRFilterRetriever
from baseline_vector_rag import RAGOutput, BaselineVectorRAG, PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class GCRGraphRAG:
    def __init__(self, driver: neo4j.Driver):
        self.driver = driver
        self.baseline = BaselineVectorRAG(driver)

        self.graph_loader = Neo4jGraphLoader(
            uri=CONFIG.neo4j.uri,
            auth=(CONFIG.neo4j.user, CONFIG.neo4j.password),
            database=CONFIG.neo4j.database,
        )
        self.gcr_retriever = GCRFilterRetriever(
            vector_retriever=self.baseline.retriever,
            graph_loader=self.graph_loader,
            gcr_threshold=CONFIG.gcr.threshold,
            enable_expansion=CONFIG.gcr.enable_expansion,
            expansion_hops=CONFIG.gcr.expansion_hops,
            max_expanded_nodes=CONFIG.gcr.max_expanded_nodes,
        )
        self.llm = self.baseline.llm

    def search(self, query: str, top_k: int = 5) -> RAGOutput:
        result = self.gcr_retriever.search(query_text=query, top_k=top_k)
        contexts = [item.content for item in result.items if item.content]

        if not contexts:
            return RAGOutput(
                answer="Insufficient context.",
                contexts=[],
                retriever_metadata={
                    "retriever": "gcr",
                    "rejected": True,
                    "gcr": result.metadata.get("gcr"),
                },
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
            retriever_metadata={
                "retriever": "gcr",
                "rejected": False,
                "gcr": result.metadata.get("gcr"),
                "gcr_se": result.metadata.get("gcr_se"),
                "gcr_sc": result.metadata.get("gcr_sc"),
                "gcr_ph": result.metadata.get("gcr_ph"),
                "expanded": result.metadata.get("expanded", False),
            },
        )

    def close(self):
        self.graph_loader.close()