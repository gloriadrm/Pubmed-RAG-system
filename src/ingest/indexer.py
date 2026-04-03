from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from uuid import uuid5, NAMESPACE_URL
from src.services.embeddings import embedding_list

from src.config import QDRANT_URL, COLLECTION_NAME, VECTOR_SIZE

client = QdrantClient(url=QDRANT_URL)


def create_collection(recreate: bool = False):
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists. Skipping.")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"Collection '{COLLECTION_NAME}' created.")


def _make_id(document_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, document_id))


def upsert_pubmed_articles(articles: list[dict]):
    texts = [a["embedding_text"] for a in articles]
    print(f"Generando embeddings para {len(texts)} artículos PubMed...")
    vectors = embedding_list(texts)
    print(f"Embeddings listos. Subiendo a Qdrant...")

    points = []
    for article, vector in zip(articles, vectors):
        payload = {
            "embedding_text": article["embedding_text"],
            # LangChain QdrantVectorStore lee "metadata" como doc.metadata
            "metadata": {
                "document_id":      article["document_id"],
                "source":           "pubmed",
                "pm_id":            article["pm_id"],
                "pmc_id":           article.get("pmc_id"),
                "doi":              article["doi"],
                "title":            article["title"],
                "authors":          article["metadata"]["authors"],
                "journal":          article["metadata"]["journal"],
                "language":         article["metadata"]["language"],
                "keywords":         article["metadata"]["keywords"],
                "pub_date":         article["metadata"]["pub_date"],
                "date_revised":     article["metadata"]["date_revised"],
                "license":          article["metadata"].get("license"),
                "pmc_last_updated": article["metadata"].get("pmc_last_updated"),
            },
        }
        points.append(PointStruct(
            id=_make_id(article["document_id"]), # el upsert no duplica points con mismo document_id
            vector=vector,
            payload=payload,
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} PubMed points.")


def upsert_pmc_chunks(chunks: list[dict]):
    texts = [c["embedding_text"] for c in chunks]
    print(f"Generando embeddings para {len(texts)} chunks PMC...")
    vectors = embedding_list(texts)
    print(f"Embeddings listos. Subiendo a Qdrant...")

    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "embedding_text": chunk["embedding_text"],
            # LangChain QdrantVectorStore lee "metadata" como doc.metadata
            "metadata": {
                "document_id":    chunk["chunk_id"],
                "source":         "pmc",
                "pmc_id":         chunk["pmc_id"],
                "pm_id":          chunk.get("pm_id"),
                "doi":            chunk.get("doi"),
                "title":          chunk["title"],
                "section":        chunk["section"],
                "chunk_index":    chunk["chunk_index"],
                "total_chunks":   chunk["total_chunks"],
                "abstract":       chunk["metadata"].get("abstract"),
                "authors":        chunk["metadata"]["authors"],
                "journal":        chunk["metadata"]["journal"],
                "language":       chunk["metadata"].get("language"),
                "keywords":       chunk["metadata"]["keywords"],
                "article_type":   chunk["metadata"].get("article_type"),
                "pub_date":       chunk["metadata"]["pub_date"],
                "license":        chunk["metadata"].get("license"),
                "pmc_last_updated": chunk["metadata"].get("pmc_last_updated"),
            },
        }
        points.append(PointStruct(
            id=_make_id(chunk["chunk_id"]),
            vector=vector,
            payload=payload,
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} PMC chunks.")


def search(query: str, limit: int = 3, source: str = None, pmc_id: str = None):
    """
    Búsqueda semántica en Qdrant.

    Args:
        query:   texto de la consulta
        limit:   número de resultados
        source:  filtrar por fuente ("pubmed" o "pmc")
        pmc_id:  filtrar por artículo concreto (solo tiene sentido con source="pmc")
    """
    from src.services.embeddings import embedding_text as _embed
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    vector = _embed(query)

    conditions = []
    if source:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))
    if pmc_id:
        conditions.append(FieldCondition(key="pmc_id", match=MatchValue(value=pmc_id)))

    query_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit,
        query_filter=query_filter,
    )

    for r in results.points:
        meta    = r.payload.get("metadata", {})
        src     = meta.get("source", "?")
        section = meta.get("section", "")
        print(f"[{src.upper()}] {meta.get('title', '')[:70]}")
        if section:
            print(f"  sección : {section}")
        print(f"  score   : {r.score:.4f}")
        print()
