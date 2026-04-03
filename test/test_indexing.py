import sys
sys.path.append('.')

from dotenv import load_dotenv
load_dotenv()

from src.data_actualization.oa_updater import get_oa_df
from src.ingest.indexer import create_collection, search
from src.ingest.pipeline import ingest_query

# Cargar OA CSV 
print("Cargando OA file list...")
oa_df = get_oa_df()
print(f"  → {len(oa_df):,} entradas en el OA CSV")

# Ingestar 
create_collection(recreate=True)
query = (
    '("gut microbiota" OR microbiome) '
    'AND ("leaky gut" OR "intestinal permeability") '
    'AND inflammation'
)
ingest_query(query, n=10, oa_df=oa_df)

