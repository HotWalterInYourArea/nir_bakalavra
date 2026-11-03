from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import neo4j


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

logger = logging.getLogger(__name__)

@dataclass
class GCRConfig:
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2
    hub_sigma: float = 2.0
    min_token_length: int = 2
    stopwords: set = field(default_factory=lambda: {
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and",
        "or", "is", "are", "was", "were", "be", "been", "by", "with",
        "as", "from", "that", "this", "it", "its", "which", "who",
        "what", "when", "where", "why", "how",
    })


def tokenize(text: str, cfg: GCRConfig) -> set[str]:
    if not text:
        return set()
    tokens = _TOKEN_RE.findall(text.lower())
    return {
        t for t in tokens
        if len(t) >= cfg.min_token_length and t not in cfg.stopwords
    }


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class GCRResult:
    gcr: float
    entity_overlap: float
    structural_connectivity: float
    hub_noise_penalty: float
    hub_nodes: list
    largest_component_ratio: float
    n_retrieved: int
    n_hub_nodes: int

    def explain(self) -> str:
        return (
            f"GCR = {self.gcr:.4f}\n"
            f"  Se = {self.entity_overlap:.4f}\n"
            f"  Sc = {self.structural_connectivity:.4f}\n"
            f"  Ph = {self.hub_noise_penalty:.4f}\n"
            f"  Узлов: {self.n_retrieved}, хабов: {self.n_hub_nodes}"
        )


class GraphContextRelevance:
    def __init__(
        self,
        graph: nx.Graph,
        config: Optional[GCRConfig] = None,
        node_text_attr: str = "text",
    ):
        self.graph = graph
        self.cfg = config or GCRConfig()
        self.node_text_attr = node_text_attr

        degrees = np.array([d for _, d in graph.degree()], dtype=float)
        self._mu = degrees.mean() if len(degrees) else 0.0
        self._sigma = degrees.std() if len(degrees) else 0.0
        self._hub_threshold = self._mu + self.cfg.hub_sigma * self._sigma

    def score(
        self,
        retrieved_nodes: Iterable[str],
        query: str,
        expected_answer: str = "",
    ) -> GCRResult:
        retrieved = [n for n in retrieved_nodes if n in self.graph]
        if not retrieved:
            return GCRResult(
                gcr=0.0, entity_overlap=0.0, structural_connectivity=0.0,
                hub_noise_penalty=0.0, hub_nodes=[],
                largest_component_ratio=0.0,
                n_retrieved=0, n_hub_nodes=0,
            )

        se, hub_nodes = self._entity_overlap(retrieved, query, expected_answer)
        sc, lcc_ratio = self._structural_connectivity(retrieved)
        ph = len(hub_nodes) / len(retrieved) if retrieved else 0.0

        gcr = (
            self.cfg.alpha * se
            + self.cfg.beta * sc
            - self.cfg.gamma * ph
        )
        gcr = max(0.0, min(1.0, gcr))

        return GCRResult(
            gcr=gcr,
            entity_overlap=se,
            structural_connectivity=sc,
            hub_noise_penalty=ph,
            hub_nodes=hub_nodes,
            largest_component_ratio=lcc_ratio,
            n_retrieved=len(retrieved),
            n_hub_nodes=len(hub_nodes),
        )

    def _entity_overlap(self, retrieved, query, expected_answer):
        query_tokens = tokenize(query, self.cfg) | tokenize(expected_answer, self.cfg)
        scores, hub_nodes = [], []

        for node in retrieved:
            text = self.graph.nodes[node].get(self.node_text_attr, "") or str(node)
            node_tokens = tokenize(text, self.cfg)
            scores.append(jaccard(query_tokens, node_tokens))

            if self.graph.degree(node) > self._hub_threshold:
                hub_nodes.append(str(node))

        return (float(np.mean(scores)) if scores else 0.0), hub_nodes

    def _structural_connectivity(self, retrieved):
        subgraph = self.graph.subgraph(retrieved)
        n = subgraph.number_of_nodes()
        if n == 0:
            return 0.0, 0.0

        components = list(nx.connected_components(subgraph))
        if not components:
            return 0.0, 0.0

        lcc = max(len(c) for c in components)
        ratio = lcc / n
        return ratio, ratio

    def hub_stats(self):
        return {
            "mean_degree": self._mu,
            "std_degree": self._sigma,
            "hub_threshold": self._hub_threshold,
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
        }

def _as_str(v):
        """Приводит значение из Neo4j к строке: str | list | None."""
        if v is None:
            return ""
        if isinstance(v, list):
            return " ".join(str(x) for x in v if x)
        return str(v)
class Neo4jGraphLoader:
    """Загружает граф из Neo4j в networkx."""

    def __init__(self, uri: str, auth: tuple, database: str = "neo4j"):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=auth)
        self.database = database

    def close(self):
        self.driver.close()


    def load_graph(self, node_label: str = "Resource") -> nx.Graph:
        G = nx.Graph()

        node_query = f"""
        MATCH (n:{node_label})
        RETURN elementId(n) AS eid,
            n.uri AS uri,
            coalesce(n.text, n.abstract, n.label, n.uri, '') AS text,
            n.label AS label
        """
        with self.driver.session(database=self.database) as session:
            for rec in session.run(node_query):
                eid = rec["eid"]
                G.add_node(
                    eid,
                    uri=_as_str(rec["uri"]),
                    text=_as_str(rec["text"]),
                    label=_as_str(rec["label"]),
                )

        edge_query = f"""
        MATCH (a:{node_label})-[r]->(b:{node_label})
        RETURN elementId(a) AS src, elementId(b) AS dst, type(r) AS rel
        """
        with self.driver.session(database=self.database) as session:
            for rec in session.run(edge_query):
                if rec["src"] and rec["dst"]:
                    G.add_edge(rec["src"], rec["dst"], rel=rec["rel"])

        logger.info(
            "Граф загружен: %d узлов, %d рёбер",
            G.number_of_nodes(), G.number_of_edges(),
        )
        return G