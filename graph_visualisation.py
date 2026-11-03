import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import neo4j

from config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Цвета по типам сущностей (аналог легенды на вашей картинке)
TYPE_COLORS = {
    "Person": "#E53935",           # красный
    "Place": "#1E88E5",            # синий
    "Organisation": "#43A047",     # зелёный
    "Work": "#FB8C00",             # оранжевый
    "Event": "#8E24AA",            # фиолетовый
    "Concept": "#00ACC1",          # голубой
    "Other": "#757575",            # серый
}


def classify_node(node_labels: list) -> str:
    """Определяет тип узла по его Neo4j-меткам."""
    labels_lower = {l.lower() for l in node_labels}

    if any("person" in l for l in labels_lower):
        return "Person"
    if any(x in labels_lower for x in ["place", "city", "country", "location"]):
        return "Place"
    if any(x in labels_lower for x in ["organisation", "organization", "company"]):
        return "Organisation"
    if any(x in labels_lower for x in ["work", "film", "book", "album"]):
        return "Work"
    if any(x in labels_lower for x in ["event", "war", "battle"]):
        return "Event"
    if any(x in labels_lower for x in ["concept", "field", "science"]):
        return "Concept"
    return "Other"


def load_graph_from_neo4j(driver, database="neo4j"):
    """Загружает граф из Neo4j с типами узлов."""
    G = nx.Graph()

    # Узлы с типами
    with driver.session(database=database) as s:
        for rec in s.run("""
            MATCH (n:Resource)
            WHERE n.uri IS NOT NULL
            RETURN elementId(n) AS eid,
                   n.uri AS uri,
                   coalesce(n.text, n.label, n.uri) AS label,
                   labels(n) AS labels
        """):
            node_type = classify_node(rec["labels"])
            G.add_node(
                rec["eid"],
                uri=rec["uri"],
                label=_short_label(rec["label"]),
                type=node_type,
            )

        # Рёбра
        for rec in s.run("""
            MATCH (a:Resource)-[r]->(b:Resource)
            WHERE a.uri IS NOT NULL AND b.uri IS NOT NULL
            RETURN elementId(a) AS src, elementId(b) AS dst, type(r) AS rel
        """):
            G.add_edge(rec["src"], rec["dst"], rel=rec["rel"])

    logger.info("Загружено: %d узлов, %d рёбер", G.number_of_nodes(), G.number_of_edges())
    return G


def _short_label(text: str) -> str:
    """Обрезает длинный текст до 30 символов."""
    if not text:
        return "?"
    text = str(text).split(".")[0]  # первое предложение
    return text[:30] + ("..." if len(text) > 30 else "")


def visualize(G: nx.Graph, output_path: str = "graph_visualization.png"):
    """Строит force-directed визуализацию с цветами по типам."""
    plt.figure(figsize=(20, 14))

    # Позиции узлов (spring layout = force-directed)
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)

    # Группируем узлы по типам
    nodes_by_type = defaultdict(list)
    for node, data in G.nodes(data=True):
        nodes_by_type[data.get("type", "Other")].append(node)

    # Размер узла = degree (число связей), с базовым минимумом
    degrees = dict(G.degree())
    node_sizes = [max(300, degrees[n] * 400) for n in G.nodes()]

    # Рисуем рёбра
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.15, width=0.5, edge_color="#999999",
    )

    # Рисуем узлы по группам
    for node_type, nodes in nodes_by_type.items():
        color = TYPE_COLORS.get(node_type, TYPE_COLORS["Other"])
        sizes = [max(300, degrees[n] * 400) for n in nodes]

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=color,
            node_size=sizes,
            alpha=0.85,
            edgecolors="white",
            linewidths=1.5,
        )

    # Подписи только для крупных узлов (топ-30 по degree)
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:30]
    labels = {n: G.nodes[n].get("label", "?")[:20] for n in top_nodes}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=7,
        font_color="#222222",
    )

    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TYPE_COLORS[t], label=t)
        for t in TYPE_COLORS
        if t in nodes_by_type
    ]
    plt.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.title(
        "Граф знаний DBpedia: учёные и их связи",
        fontsize=16, pad=20,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info("Сохранено: %s", output_path)
    plt.show()


if __name__ == "__main__":
    driver = neo4j.GraphDatabase.driver(
        CONFIG.neo4j.uri,
        auth=(CONFIG.neo4j.user, CONFIG.neo4j.password),
    )
    try:
        G = load_graph_from_neo4j(driver, CONFIG.neo4j.database)
        visualize(G, "results/graph_visualization.png")
    finally:
        driver.close()