from __future__ import annotations

import logging
import re
from typing import Any, Optional

import networkx as nx
import neo4j
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem

from gcr import GraphContextRelevance, GCRConfig, Neo4jGraphLoader

logger = logging.getLogger(__name__)

_URI_RE = re.compile(r"http://dbpedia\.org/resource/[\w\(\)\,\.\-]+")


class GCRFilterRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_loader: Neo4jGraphLoader,
        gcr_threshold: float = 0.35,
        gcr_config: Optional[GCRConfig] = None,
        enable_expansion: bool = True,
        expansion_hops: int = 1,
        max_expanded_nodes: int = 20,
        node_text_property: str = "abstract",
    ):
        self.vector_retriever = vector_retriever
        self.graph_loader = graph_loader
        self.gcr_threshold = gcr_threshold
        self.enable_expansion = enable_expansion
        self.expansion_hops = expansion_hops
        self.max_expanded_nodes = max_expanded_nodes
        self.node_text_property = node_text_property

        logger.info("Загрузка графа для GCR...")
        self.graph = graph_loader.load_graph(node_label="Resource")
        self.gcr = GraphContextRelevance(
            graph=self.graph,
            config=gcr_config,
            node_text_attr="text",
        )
        stats = self.gcr.hub_stats()
        logger.info(
            "Граф: %d узлов, %d рёбер, порог хабов %.2f",
            stats["n_nodes"], stats["n_edges"], stats["hub_threshold"],
        )

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        query_params: Optional[dict] = None,
        **kwargs,
    ) -> RetrieverResult:
        vector_result = self.vector_retriever.search(
            query_text=query_text, top_k=top_k, **kwargs,
        )
        # === ДИАГНОСТИКА ===
        logger.info("=== VECTOR RESULT ===")
        logger.info("Items count: %d", len(vector_result.items))
        for i, item in enumerate(vector_result.items[:2]):
            logger.info("Item %d metadata: %s", i, item.metadata)
            logger.info("Item %d content: %s", i, item.content[:100] if item.content else None)
        # ===================

        retrieved_uris = self._extract_uris(vector_result.items)
        logger.info("Extracted URIs: %s", retrieved_uris)
        logger.info("Graph has %d nodes", self.graph.number_of_nodes())
        # Какие из них есть в графе?
        found = [u for u in retrieved_uris if u in self.graph]
        logger.info("URIs found in graph: %d / %d", len(found), len(retrieved_uris))
        if not vector_result.items:
            return vector_result

        retrieved_uris = self._extract_uris(vector_result.items)
        if not retrieved_uris:
            logger.warning("Не удалось извлечь URI из результатов.")
            return vector_result

        gcr_result = self.gcr.score(
            retrieved_nodes=retrieved_uris,
            query=query_text,
        )
        logger.info(
            "GCR=%.4f (Se=%.3f, Sc=%.3f, Ph=%.3f, хабов=%d)",
            gcr_result.gcr, gcr_result.entity_overlap,
            gcr_result.structural_connectivity,
            gcr_result.hub_noise_penalty, gcr_result.n_hub_nodes,
        )

        if gcr_result.gcr >= self.gcr_threshold:
            return vector_result

        logger.warning(
            "GCR %.4f < порога %.2f. Попытка расширения.",
            gcr_result.gcr, self.gcr_threshold,
        )

        if not self.enable_expansion:
            return RetrieverResult(
                items=[], metadata={"gcr": gcr_result.gcr, "rejected": True},
            )

        expanded_uris = self._expand_context(retrieved_uris)
        expanded_gcr = self.gcr.score(
            retrieved_nodes=expanded_uris, query=query_text,
        )
        logger.info("GCR после расширения: %.4f", expanded_gcr.gcr)

        if expanded_gcr.gcr < self.gcr_threshold:
            logger.warning("Даже после расширения GCR ниже порога.")
            return RetrieverResult(
                items=[],
                metadata={"gcr": expanded_gcr.gcr, "rejected": True},
            )

        return self._build_expanded_result(expanded_uris, expanded_gcr)

    def _extract_uris(self, items) -> list:
        """Извлекает elementId узлов из результатов VectorRetriever."""
        eids = []
        for item in items:
            if item.metadata and "id" in item.metadata:
                eids.append(item.metadata["id"])
            elif item.metadata and "elementId" in item.metadata:
                eids.append(item.metadata["elementId"])
        return eids

    def _expand_context(self, seed_uris: list) -> list:
        expanded = set(seed_uris)
        valid_seeds = [u for u in seed_uris if u in self.graph]

        if not valid_seeds:
            return list(expanded)

        for seed in valid_seeds:
            neighbors = nx.single_source_shortest_path_length(
                self.graph, seed, cutoff=self.expansion_hops,
            )
            expanded.update(neighbors.keys())
            if len(expanded) >= self.max_expanded_nodes:
                break

        result = list(valid_seeds)
        for u in expanded:
            if u not in result and len(result) < self.max_expanded_nodes:
                result.append(u)

        logger.info("Расширение: %d seed → %d узлов", len(valid_seeds), len(result))
        return result

    def _build_expanded_result(self, eids: list, gcr_result) -> RetrieverResult:
        """Формирует RetrieverResult из расширенного подграфа."""
        items = []
        for eid in eids:
            if eid not in self.graph:
                continue

            node_data = self.graph.nodes[eid]
            text = self.graph.nodes[eid].get("text", "")
            if isinstance(text, list):
                text = " ".join(str(x) for x in text if x)
            text = text or eid
            uri = node_data.get("uri", eid)

            if len(text) > 2000:
                text = text[:2000] + "..."

            items.append(
                RetrieverResultItem(
                    content=text,
                    metadata={
                        "uri": uri,
                        "elementId": eid,
                        "source": "gcr_expansion",
                    },
                )
            )

        return RetrieverResult(
            items=items,
            metadata={
                "retriever": "GCRFilterRetriever",
                "gcr": gcr_result.gcr,
                "gcr_se": gcr_result.entity_overlap,
                "gcr_sc": gcr_result.structural_connectivity,
                "gcr_ph": gcr_result.hub_noise_penalty,
                "expanded": True,
            },
        )