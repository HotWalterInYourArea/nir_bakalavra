from __future__ import annotations

import json
import logging
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import neo4j
import requests
from sentence_transformers import SentenceTransformer

from config import CONFIG
from entity_fetcher import EntityFetcher

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Кэш сущностей: чтобы не дёргать DBpedia при каждом запуске
# ----------------------------------------------------------------------
ENTITIES_CACHE = "entities.json"


def get_target_entities(limit: int = 100) -> list[str]:
    """
    Возвращает список title'ов сущностей для загрузки.
    Кэширует результат в entities.json.
    """
    if Path(ENTITIES_CACHE).exists():
        logger.info("Загрузка сущностей из кэша: %s", ENTITIES_CACHE)
        with open(ENTITIES_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        titles = [d["title"] for d in data]
        logger.info("Загружено %d сущностей из кэша", len(titles))
        return titles[:limit]

    logger.info("Запрос сущностей из DBpedia...")
    fetcher = EntityFetcher()
    entities = fetcher.fetch_mixed(limit=limit)

    if not entities:
        logger.error("DBpedia не вернула сущностей. Использую fallback.")
        return _fallback_entities()[:limit]

    with open(ENTITIES_CACHE, "w", encoding="utf-8") as f:
        json.dump(
            [{"uri": e.uri, "name": e.name, "title": e.title} for e in entities],
            f, ensure_ascii=False, indent=2,
        )
    logger.info("Сохранено %d сущностей в %s", len(entities), ENTITIES_CACHE)

    return [e.title for e in entities][:limit]


def _fallback_entities() -> list[str]:
    """Резервный список, если DBpedia недоступна."""
    return [
        "Albert_Einstein", "Marie_Curie", "Isaac_Newton", "Nikola_Tesla",
        "Charles_Darwin", "Ada_Lovelace", "Leonardo_da_Vinci", "Galileo_Galilei",
        "Stephen_Hawking", "Richard_Feynman", "Niels_Bohr", "Max_Planck",
        "Erwin_Schrödinger", "Werner_Heisenberg", "Michael_Faraday",
        "James_Clerk_Maxwell", "Louis_Pasteur", "Gregor_Mendel",
        "Alan_Turing", "John_von_Neumann",
    ]


# ----------------------------------------------------------------------
# SPARQL CONSTRUCT для структурных связей
# ----------------------------------------------------------------------
def build_sparql_query(titles: list) -> str:
    """Строит SPARQL CONSTRUCT для списка сущностей."""
    # Ограничиваем до 150 сущностей — публичный эндпоинт не любит
    # слишком большие VALUES-блоки
    titles = titles[:150]
    values = " ".join(f"dbr:{t}" for t in titles)
    return f"""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

CONSTRUCT {{
    ?entity a ?type .
    ?entity rdfs:label ?name .
    ?entity dbo:birthPlace ?place .
    ?entity dbo:birthDate ?birthDate .
    ?entity dbo:deathDate ?deathDate .
    ?entity dbo:occupation ?occupation .
    ?entity dbo:field ?field .
    ?entity dbo:nationality ?nationality .
    ?place rdfs:label ?placeName .
    ?occupation rdfs:label ?occName .
}}
WHERE {{
    VALUES ?entity {{ {values} }}
    ?entity a ?type .
    OPTIONAL {{ ?entity rdfs:label ?name . FILTER(LANG(?name) = "en") }}
    OPTIONAL {{
        ?entity dbo:birthPlace ?place .
        ?place rdfs:label ?placeName .
        FILTER(LANG(?placeName) = "en")
    }}
    OPTIONAL {{ ?entity dbo:birthDate ?birthDate . }}
    OPTIONAL {{ ?entity dbo:deathDate ?deathDate . }}
    OPTIONAL {{
        ?entity dbo:occupation ?occupation .
        ?occupation rdfs:label ?occName .
        FILTER(LANG(?occName) = "en")
    }}
    OPTIONAL {{ ?entity dbo:field ?field . }}
    OPTIONAL {{ ?entity dbo:nationality ?nationality . }}
}}
"""


# ----------------------------------------------------------------------
# Wikipedia REST API
# ----------------------------------------------------------------------
WIKI_REST_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_HEADERS = {
    "User-Agent": "BachelorThesis-RAG/1.0 (educational use; contact@example.com)",
    "Accept": "application/json",
}


def fetch_wikipedia_extract(title: str, timeout: int = 15) -> str:
    url = WIKI_REST_URL.format(title=urllib.parse.quote(title, safe=""))
    try:
        r = requests.get(url, headers=WIKI_HEADERS, timeout=timeout)
        if r.status_code != 200:
            return ""
        return r.json().get("extract", "") or ""
    except Exception as e:
        logger.debug("Wikipedia %s: %s", title, e)
        return ""


def fetch_all_extracts(titles: list, max_workers: int = 6) -> dict:
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_wikipedia_extract, t): t for t in titles}
        for fut in as_completed(futures):
            title = futures[fut]
            try:
                results[title] = fut.result()
            except Exception as e:
                logger.warning("Ошибка для %s: %s", title, e)
                results[title] = ""
    return results


# ----------------------------------------------------------------------
# Основной загрузчик
# ----------------------------------------------------------------------
@dataclass
class LoadStats:
    triples_imported: int = 0
    nodes_with_text: int = 0
    nodes_with_embeddings: int = 0
    wikipedia_hits: int = 0
    wikipedia_misses: int = 0
    total_entities: int = 0


class HybridDBpediaLoader:
    def __init__(self, entity_limit: int = 100):
        self.driver = neo4j.GraphDatabase.driver(
            CONFIG.neo4j.uri,
            auth=(CONFIG.neo4j.user, CONFIG.neo4j.password),
        )
        self.endpoint = "https://dbpedia.org/sparql"
        self.stats = LoadStats()
        self.entity_limit = entity_limit
        self.entities: list[str] = []

    def close(self):
        self.driver.close()

    def _run(self, query, **params):
        with self.driver.session(database=CONFIG.neo4j.database) as s:
            return [r.data() for r in s.run(query, **params)]

    # ------------------------------------------------------------------
    def setup_n10s(self):
        logger.info("Инициализация n10s...")
        try:
            self._run("""
                CALL n10s.graphconfig.init({
                    handleVocabUris: "IGNORE",
                    handleMultival: "ARRAY",
                    keepLangTag: false,
                    applyNeo4jNaming: false
                })
            """)
            logger.info("n10s.graphconfig инициализирован")
        except Exception as e:
            logger.warning("n10s.graphconfig: %s (уже инициализирован)", e)

        try:
            self._run("""
                CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS
                FOR (r:Resource) REQUIRE r.uri IS UNIQUE
            """)
            logger.info("Constraint готов")
        except Exception as e:
            logger.warning("Constraint: %s", e)

    # ------------------------------------------------------------------
    def import_structural_triples(self):
        query = build_sparql_query(self.entities)
        encoded = urllib.parse.quote(query)

        logger.info("Импорт структурных троек для %d сущностей...", len(self.entities))
        cypher = f"""
        CALL n10s.rdf.import.fetch(
            "{self.endpoint}", "Turtle",
            {{
                headerParams: {{ Accept: "application/turtle" }},
                payload: "query={encoded}"
            }}
        )
        YIELD triplesLoaded, extraInfo
        RETURN triplesLoaded, extraInfo
        """
        try:
            result = self._run(cypher)
            if result:
                loaded = result[0].get("triplesLoaded", 0)
                self.stats.triples_imported = loaded
                logger.info("Импортировано %d триплетов", loaded)
                if loaded == 0:
                    logger.warning("extraInfo: %s", result[0].get("extraInfo", ""))
            return self.stats.triples_imported
        except Exception as e:
            logger.error("Ошибка импорта: %s", e)
            return 0

    # ------------------------------------------------------------------
    def attach_wikipedia_extracts(self):
        logger.info("Загрузка abstracts из Wikipedia для %d статей...",
                    len(self.entities))
        extracts = fetch_all_extracts(self.entities, max_workers=6)

        hits = sum(1 for v in extracts.values() if v)
        self.stats.wikipedia_hits = hits
        self.stats.wikipedia_misses = len(self.entities) - hits
        logger.info("Wikipedia: %d hits, %d misses", hits, self.stats.wikipedia_misses)

        updated = 0
        with self.driver.session(database=CONFIG.neo4j.database) as s:
            for title, text in extracts.items():
                if not text:
                    continue
                result = s.run(
                    """
                    MATCH (n:Resource)
                    WHERE n.uri ENDS WITH $suffix
                       OR n.uri = $full
                    SET n.text = $text, n.abstract = $text
                    RETURN count(n) AS cnt
                    """,
                    suffix=f"/{title}",
                    full=f"http://dbpedia.org/resource/{title}",
                    text=text[:5000],
                ).single()
                if result and result["cnt"] > 0:
                    updated += result["cnt"]

        self.stats.nodes_with_text = updated
        logger.info("Записано текстов в %d узлов", updated)

    # ------------------------------------------------------------------
    def build_vector_index(self):
        embedder = SentenceTransformer(CONFIG.embedding.model)

        try:
            self._run(f"DROP INDEX {CONFIG.embedding.index} IF EXISTS")
        except Exception:
            pass

        self._run(f"""
            CREATE VECTOR INDEX {CONFIG.embedding.index} IF NOT EXISTS
            FOR (n:Resource) ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {CONFIG.embedding.dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """)
        logger.info("Векторный индекс создан")

        rows = self._run("""
            MATCH (n:Resource)
            WHERE n.embedding IS NULL AND n.text IS NOT NULL
              AND size(n.text) > 0
            RETURN elementId(n) AS nid, n.text AS txt
        """)

        if not rows:
            logger.info("Нет узлов, требующих эмбеддингов")
            return

        logger.info("Считаем эмбеддинги для %d узлов...", len(rows))
        texts = [r["txt"] for r in rows]
        embeddings = embedder.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=True, batch_size=32,
        )

        with self.driver.session(database=CONFIG.neo4j.database) as s:
            for r, emb in zip(rows, embeddings):
                s.run(
                    "MATCH (n) WHERE elementId(n) = $nid SET n.embedding = $emb",
                    nid=r["nid"], emb=emb.tolist(),
                )

        self.stats.nodes_with_embeddings = len(rows)
        logger.info("Обновлено %d эмбеддингов", len(rows))

    # ------------------------------------------------------------------
    def diagnose(self):
        logger.info("=" * 60)
        logger.info("ДИАГНОСТИКА ГРАФА")
        logger.info("=" * 60)
        stats = {
            "total": self._run("MATCH (n) RETURN count(n) AS c")[0]["c"],
            "resources": self._run("MATCH (n:Resource) RETURN count(n) AS c")[0]["c"],
            "with_text": self._run(
                "MATCH (n:Resource) WHERE n.text IS NOT NULL RETURN count(n) AS c"
            )[0]["c"],
            "with_emb": self._run(
                "MATCH (n:Resource) WHERE n.embedding IS NOT NULL RETURN count(n) AS c"
            )[0]["c"],
            "rels": self._run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"],
        }
        for k, v in stats.items():
            logger.info("  %-20s %d", k, v)

    # ------------------------------------------------------------------
    def _ensure_target_nodes(self):
        with self.driver.session(database=CONFIG.neo4j.database) as s:
            for title in self.entities:
                uri = f"http://dbpedia.org/resource/{title}"
                s.run(
                    """
                    MERGE (n:Resource {uri: $uri})
                    ON CREATE SET n.created_by = 'hybrid_loader',
                                  n.title = $title
                    """,
                    uri=uri, title=title,
                )
        logger.info("Гарантированы узлы для %d сущностей", len(self.entities))

    # ------------------------------------------------------------------
    def run(self):
        logger.info("=" * 60)
        logger.info("ГИБРИДНАЯ ЗАГРУЗКА DBpedia + Wikipedia")
        logger.info("=" * 60)

        t0 = time.time()

        # 1. Получаем список сущностей
        self.entities = get_target_entities(limit=self.entity_limit)
        self.stats.total_entities = len(self.entities)
        if not self.entities:
            logger.error("Список сущностей пуст. Выход.")
            return self.stats

        # 2. n10s setup
        self.setup_n10s()

        # 3. Импорт структурных троек
        self.import_structural_triples()

        # 4. Гарантируем узлы (если n10s что-то пропустил)
        self._ensure_target_nodes()

        # 5. Тексты из Wikipedia
        self.attach_wikipedia_extracts()

        # 6. Векторный индекс
        self.build_vector_index()

        # 7. Диагностика
        self.diagnose()

        logger.info("Загрузка завершена за %.1f сек", time.time() - t0)
        return self.stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    loader = HybridDBpediaLoader(entity_limit=100)
    try:
        stats = loader.run()
        print("\n" + "=" * 60)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 60)
        print(f"  Сущностей в списке:      {stats.total_entities}")
        print(f"  Триплетов импортировано: {stats.triples_imported}")
        print(f"  Узлов с текстом:         {stats.nodes_with_text}")
        print(f"  Узлов с эмбеддингами:    {stats.nodes_with_embeddings}")
        print(f"  Wikipedia hits:          {stats.wikipedia_hits}")
        print(f"  Wikipedia misses:        {stats.wikipedia_misses}")
    finally:
        loader.close()