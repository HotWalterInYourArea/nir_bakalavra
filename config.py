from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")


@dataclass
class EmbeddingConfig:
    """Локальные эмбеддинги через sentence-transformers."""
    provider: str = "sentence-transformers"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384
    index: str = "chunk_embeddings"


@dataclass
class LLMConfig:
    """Локальная LLM через Ollama."""
    provider: str = "ollama"
    model: str = os.getenv("LLM_MODEL", "llama3.1")
    temperature: float = 0.0
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )


@dataclass
class GCRConfig:
    alpha: float = 0.5          # было 0.4 — увеличиваем вес Se
    beta: float = 0.3           # было 0.4 — уменьшаем вес Sc
    gamma: float = 0.2
    hub_sigma: float = 3.0      # было 2.0 — строже к хабам
    threshold: float = 0.08     # было 0.35 — реалистичный порог
    enable_expansion: bool = True
    expansion_hops: int = 1
    max_expanded_nodes: int = 20

@dataclass
class ExperimentConfig:
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    gcr: GCRConfig = field(default_factory=GCRConfig)
    eval_dataset_size: int = 100
    top_k: int = 5
    results_dir: str = "results"
    seed: int = 42


CONFIG = ExperimentConfig()