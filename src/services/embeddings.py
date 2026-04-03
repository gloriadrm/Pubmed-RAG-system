from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

from src.config import EMBEDDING_MODEL as MODEL_NAME

embedding_model = SentenceTransformer(MODEL_NAME, device="cpu")

# Procesa una lista y devuelve una lista de vectores (INGESTA)
def embedding_list(texts: list[str]) -> list[list[float]]:
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.tolist()

# Procesa un texto suelto y devuelve un vector (CONSULTAS)
def embedding_text(text: str) -> list[float]:
    return embedding_list([text])[0]


# Wrapper LangChain para reutilizar el mismo modelo en QdrantVectorStore
class BGE_m3_Embeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embedding_list(texts)

    def embed_query(self, text: str) -> list[float]:
        return embedding_text(text)

embeddings_langchain = BGE_m3_Embeddings()