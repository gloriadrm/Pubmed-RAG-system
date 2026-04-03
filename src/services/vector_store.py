"""
vector_store.py
---------------
Conexión a la colección Qdrant para búsqueda semántica dentro de la chain RAG.

Usa QdrantVectorStore de langchain-qdrant con nuestro modelo bge-m3 y la
colección 'biomedical_rag'. El campo content_payload_key="embedding_text"
hace que doc.page_content contenga el texto ya formateado con título y sección,
que es lo que pasamos al LLM como contexto.
"""

from langchain_qdrant import QdrantVectorStore
from src.services.embeddings import embeddings_langchain  # instancia de BGE_m3_Embeddings
from src.ingest.indexer import COLLECTION_NAME, QDRANT_URL

qdrant_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings_langchain,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    content_payload_key="embedding_text",   # campo del payload de Qdrant que se convierte en el contexto que el LLM va a leer
)
