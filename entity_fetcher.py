from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "BachelorThesis-RAG/1.0 (educational use)",
}


# ----------------------------------------------------------------------
# SPARQL: получить учёных с наибольшим числом связей
# ----------------------------------------------------------------------
SPARQL_SCIENTISTS = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT DISTINCT ?person ?name WHERE {
    ?person a dbo:Scientist .
    ?person rdfs:label ?name .
    ?person dbo:birthDate ?birthDate .
    ?person dbo:birthPlace ?birthPlace .
    FILTER(LANG(?name) = "en")
}
ORDER BY ?name
LIMIT 200
"""


# ----------------------------------------------------------------------
# Альтернативные запросы для расширения домена
# ----------------------------------------------------------------------
SPARQL_PHYSICISTS = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?person ?name WHERE {
    ?person a dbo:Person .
    ?person dbo:occupation <http://dbpedia.org/resource/Physicist> .
    ?person rdfs:label ?name .
    ?person dbo:birthDate ?birthDate .
    ?person dbo:birthPlace ?birthPlace .
    FILTER(LANG(?name) = "en")
}
ORDER BY ?name
LIMIT 150
"""


SPARQL_MATHEMATICIANS = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?person ?name WHERE {
    ?person a dbo:Person .
    ?person dbo:occupation <http://dbpedia.org/resource/Mathematician> .
    ?person rdfs:label ?name .
    ?person dbo:birthDate ?birthDate .
    FILTER(LANG(?name) = "en")
}
ORDER BY ?name
LIMIT 150
"""


SPARQL_MIXED = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?person ?name WHERE {
    VALUES ?occ {
        dbr:Physicist dbr:Chemist dbr:Mathematician
        dbr:Biologist dbr:Astronomer dbr:Inventor
        dbr:Engineer dbr:Computer_scientist
    }
    ?person a dbo:Person .
    ?person dbo:occupation ?occ .
    ?person rdfs:label ?name .
    ?person dbo:birthDate ?birthDate .
    ?person dbo:birthPlace ?birthPlace .
    FILTER(LANG(?name) = "en")
}
ORDER BY ?name
LIMIT 3000
"""


@dataclass
class Entity:
    uri: str
    name: str
    title: str      # часть URI после /resource/ — для Wikipedia API


class EntityFetcher:
    """Получает список сущностей из DBpedia по SPARQL."""

    def __init__(self, endpoint: str = DBPEDIA_ENDPOINT):
        self.endpoint = endpoint

    def fetch(self, query: str) -> list[Entity]:
        """Выполняет SPARQL SELECT и парсит результаты."""
        logger.info("Запрос к DBpedia...")
        try:
            r = requests.get(
                self.endpoint,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.error("Ошибка SPARQL: %s", e)
            return []

        entities = []
        for binding in data.get("results", {}).get("bindings", []):
            uri = binding["person"]["value"]
            name = binding["name"]["value"]

            # Извлекаем title для Wikipedia API: часть после /resource/
            if "/resource/" not in uri:
                continue
            title = uri.split("/resource/")[-1]

            entities.append(Entity(uri=uri, name=name, title=title))

        logger.info("Получено %d сущностей", len(entities))
        return entities

    def fetch_mixed(self, limit: int = 200) -> list[Entity]:
        """Смешанный список: физики, химики, математики, биологи и т.д."""
        entities = self.fetch(SPARQL_MIXED)
        # Дедупликация по URI
        seen = set()
        unique = []
        for e in entities:
            if e.uri not in seen:
                seen.add(e.uri)
                unique.append(e)
            if len(unique) >= limit:
                break
        logger.info("Уникальных сущностей: %d", len(unique))
        return unique


# ----------------------------------------------------------------------
# CLI: посмотреть, кого вернул запрос
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = EntityFetcher()
    entities = fetcher.fetch_mixed(limit=1000)

    print(f"\n{'=' * 70}")
    print(f"Найдено {len(entities)} сущностей")
    print(f"{'=' * 70}")
    for e in entities[:20]:
        print(f"  {e.name:40s} → {e.title}")

    # Сохраняем в JSON для последующего использования
    import json
    with open("entities.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"uri": e.uri, "name": e.name, "title": e.title} for e in entities],
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nСохранено в entities.json")