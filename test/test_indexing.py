"""
test_indexing.py
-----------------
Script manual de siembra inicial de la colección: recrea la colección desde
cero (create_collection(recreate=True) — DESTRUCTIVO) y la puebla con una
query fija.

Nunca debe ejecutarse importando el módulo (p. ej. al recolectar tests con
pytest sobre todo el directorio test/) — solo invocando este archivo
directamente. Todo el código va dentro de main(), guardado tras
if __name__ == "__main__".

Uso:
    python test/test_indexing.py
"""

import sys
from pathlib import Path

# Ver test_actualizations.py: insertar al principio, no anexar al final,
# para no resolver src.* contra la copia instalada en site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.data_actualization.oa_updater import get_oa_df
from src.ingest.indexer import create_collection, search
from src.ingest.pipeline import ingest_query


def main():
    load_dotenv()

    print("Cargando OA file list...")
    oa_df = get_oa_df()
    print(f"  → {len(oa_df):,} entradas en el OA CSV")

    create_collection(recreate=True)
    query = (
        '("gut microbiota" OR microbiome) '
        'AND ("leaky gut" OR "intestinal permeability") '
        'AND inflammation'
    )
    ingest_query(query, n=10, oa_df=oa_df)


if __name__ == "__main__":
    main()
