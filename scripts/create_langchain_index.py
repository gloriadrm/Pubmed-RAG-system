""" 
PDFs en una carpeta → carga de documentos → ajuste de metadatos → conexión a Qdrant → creación de colección → vectorización + inserción de documentos
"""


import sys 
from dotenv import load_dotenv

# Añade la carpeta actual al path de Python para poder importar módulos propios del proyecto
sys.path.append('.')
load_dotenv()

from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFDirectoryLoader # para cargar todos los PDFs de una carpeta
from qdrant_client.models import VectorParams, Distance # para definir cómo será la colección en Qdrant (tamaño del vector y la métrica de similitud)
from langchain_qdrant import QdrantVectorStore
from src.services.embeddings import embeddings_model_langchain
import os 
from uuid import uuid4 # para generar IDs únicos por documento
import time 
import warnings
warnings.filterwarnings("ignore")
 
#CARGA DOCUMENTOS
data_path = 'data/optimized_chunks'
loader = PyPDFDirectoryLoader(data_path)
documents = loader.load()

for doc in documents:
    source_path = doc.metadata.get('source','') # coge doc.metadata['source'] --> data/optimized_chunks/Presentación.pdf  
    file_name = os.path.basename(source_path) # extrae el nombre del archivo 
    source_name = os.path.splitext(file_name)[0] # elimina la extensión .pdf
    doc.metadata['source'] = source_name

# QDRANT CONFIGURATION
collection_name = "langchain_index"
qdrant_url = "http://localhost:6333"

client = QdrantClient(url=qdrant_url)

# BORRADO PREVIO DE COLECCIÓN
try:
    client.get_collection(collection_name)
    client.delete_collection(collection_name)
except:
    pass 

# CREACIÓN DE COLECCIÓN
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=3072, # DIMENSIÓN EMBEDDINGS 
                                distance=Distance.COSINE), # MÉTRICA DISTANCIA 
)

# ALTERNATIVA EN BATCH   
#QdrantVectorStore.from_documents(
#    documents=documents,
#    embedding=embeddings_model_langchain,
#    collection_name=collection_name,
#    url=qdrant_url,
#    prefer_grpc=True,
#    force_recreate=True
#)

# CREACION DEL VECTOR STORE
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings_model_langchain
)

# INSERCION DE DOCUMENTOS EN LA VECTOR STORE
for document in documents:
    doc_id = str(uuid4())
    vector_store.add_documents(documents=[document],ids=[doc_id])
    time.sleep(3) # para evitar problemas de rate limit o saturación