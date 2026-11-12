"""
Grid Search for GCR Weights (Alpha, Beta, Gamma).
Optimized for high-memory: loads graph and vector results once.
"""
import logging
import itertools
import pandas as pd
import numpy as np
import neo4j
from sentence_transformers import SentenceTransformer
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

from config import CONFIG
from gcr import GraphContextRelevance, GCRConfig, Neo4jGraphLoader
from eval_dataset import EvalDatasetBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Define the grid of weights to test
# Alpha (Se), Beta (Sc), Gamma (Ph)
WEIGHT_GRID = {
    "alpha": [0.2, 0.4, 0.6],
    "beta":  [0.2, 0.4, 0.6],
    "gamma": [0.1, 0.2, 0.3],
    "threshold": [0.10, 0.15, 0.20]
}

def run_sensitivity_analysis():
    # 1. Load Data & Graph
    logger.info("Loading dataset...")
    qa_pairs = EvalDatasetBuilder.load("eval_dataset.json")
    questions = [p.question for p in qa_pairs]

    logger.info("Loading Neo4j graph into memory...")
    driver = neo4j.GraphDatabase.driver(
        CONFIG.neo4j.uri, auth=(CONFIG.neo4j.user, CONFIG.neo4j.password)
    )
    graph_loader = Neo4jGraphLoader(
        uri=CONFIG.neo4j.uri,
        auth=(CONFIG.neo4j.user, CONFIG.neo4j.password),
        database=CONFIG.neo4j.database
    )
    G = graph_loader.load_graph(node_label="Resource")
    logger.info("Graph loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # 2. Run Vector Search ONCE (Cache results)
    logger.info("Running vector search for all questions (this takes a moment)...")
    embedder = SentenceTransformerEmbeddings(model=CONFIG.embedding.model)
    vector_retriever = VectorRetriever(
        driver=driver,
        index_name=CONFIG.embedding.index,
        embedder=embedder,
        return_properties=["uri", "text"],
    )

    cached_retrievals = []
    for i, q in enumerate(questions):
        result = vector_retriever.search(query_text=q, top_k=10)
        # Extract elementIds from the search results
        eids = [item.metadata.get("id") for item in result.items if item.metadata and "id" in item.metadata]
        cached_retrievals.append(eids)
        if (i + 1) % 20 == 0:
            logger.info("Cached retrieval %d/%d", i + 1, len(questions))

    # 3. Grid Search over Weights
    logger.info("Starting Grid Search...")
    results = []
    
    # Pre-instantiate GCR with a dummy config, we will override weights later
    gcr_engine = GraphContextRelevance(G, config=GCRConfig())

    combinations = list(itertools.product(
        WEIGHT_GRID["alpha"], 
        WEIGHT_GRID["beta"], 
        WEIGHT_GRID["gamma"], 
        WEIGHT_GRID["threshold"]
    ))
    
    for alpha, beta, gamma, threshold in combinations:
        accepted_count = 0
        total_gcr = 0.0
        total_se = 0.0
        total_sc = 0.0
        total_ph = 0.0

        # Override weights in the engine
        gcr_engine.cfg.alpha = alpha
        gcr_engine.cfg.beta = beta
        gcr_engine.cfg.gamma = gamma

        for i, q in enumerate(questions):
            seed_eids = cached_retrievals[i]
            if not seed_eids:
                continue

            # Compute initial GCR
            res = gcr_engine.score(seed_eids, q)
            
            # If below threshold, simulate expansion (1-hop)
            if res.gcr < threshold:
                # Expand context in memory using networkx
                expanded_eids = set(seed_eids)
                for seed in seed_eids:
                    if seed in G:
                        expanded_eids.update(G.neighbors(seed))
                # Limit expansion size
                expanded_eids = list(expanded_eids)[:20]
                
                # Recompute GCR on expanded
                res = gcr_engine.score(expanded_eids, q)

            total_gcr += res.gcr
            total_se += res.entity_overlap
            total_sc += res.structural_connectivity
            total_ph += res.hub_noise_penalty
            
            if res.gcr >= threshold:
                accepted_count += 1

        n = len(questions)
        results.append({
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "threshold": threshold,
            "avg_gcr": total_gcr / n,
            "avg_se": total_se / n,
            "avg_sc": total_sc / n,
            "avg_ph": total_ph / n,
            "acceptance_rate": accepted_count / n,
            "accepted_queries": accepted_count
        })

    # 4. Save Results
    df = pd.DataFrame(results)
    df.to_csv("results/gcr_grid_search.csv", index=False)
    logger.info("Grid search complete. Saved to results/gcr_grid_search.csv")
    
    # Print best combination (highest acceptance rate without sacrificing GCR)
    best = df.sort_values(by=["acceptance_rate", "avg_gcr"], ascending=[False, False]).iloc[0]
    logger.info("Best Configuration:\n%s", best)

    graph_loader.close()
    driver.close()

if __name__ == "__main__":
    run_sensitivity_analysis()