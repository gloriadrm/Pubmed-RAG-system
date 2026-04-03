from fastapi import APIRouter

from src.api.schema import RAGRequest  # schema → validación de inputs (Pydantic)
from src.processes.langchain_chain.chain import rag_chain # → pipeline LLM
from src.services.vector_store import qdrant_langchain # qdrant_langchain es la conexión al vector store (VectorStore de LangChain)

router = APIRouter()

@router.post("/search")
async def search(query: str):
    found_docs = await qdrant_langchain.asimilarity_search(query, k=5)
    return found_docs

@router.post("/rag")
async def rag_endpoint(request: RAGRequest):
    """
    Endpoint to interact with the RAG system.
    """
    result = await rag_chain.ainvoke({"question": request.question})
    return {
        "question": result["question"],
        "answer": result["answer"],
        "source": result["source"].selection if result.get('source') else None,
        "source_reason": result["source"].reason if result.get('source') else None
    }