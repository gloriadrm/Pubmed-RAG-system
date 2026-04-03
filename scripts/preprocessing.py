"""Divide un PDF en varios PDFs usando sus marcadores (índice)
    chunking semántico basado en estructura del documento. """

# Para leer/escribir PDFs
from pypdf import PdfReader, PdfWriter
# Destination = enlace a una sección del PDF
# es un objeto que representa un marcador (bookmark) del PDF
# Contiene  title, page_number, y otras propiedades que permiten navegar a esa sección del PDF
from pypdf.generic import Destination 
# Tipado
from typing import Dict, List, Any
import os
import warnings
warnings.filterwarnings("ignore")

reader = PdfReader('data/informe_mercado_de_trabajo_jovenes.pdf')

# LISTA DE TITULOS QUE NOS INSTERESAN
chunk_level_info = [
'Presentación',
'Índices',
'Objetivos',
'Metodología',
'Fuentes',
'Indicadores de los colectivos de atención prioritaria para el empleo. Comparativa',
'Los jóvenes menores de 30 años en el mercado laboral',
'1. Población y actividad laboral',
'2. El empleo',
'3. El desempleo',
'4. Ocupaciones',
'Glosario de términos'
]

# OBTIENE EL ÍNDICE DEL PDF
bookmarks = reader.outline

all_bookmark_info: List[Dict[str,Any]] = []

for item in bookmarks:
    # BOOKMARK SIMPLE 
    if isinstance(item, Destination):
        title = item.title # TITULO
        page_number = reader.get_destination_page_number(item) # PAGINA DE INICIO (ojo: base 0)
        all_bookmark_info.append({"title": title, "start_page": page_number }) 
    
    # LISTA ESTRUCTURADA (SUBSECCIONES DEL INDICE)
    elif isinstance(item, list):
        parent_bookmark = item[0] 
        title = parent_bookmark.title # TITULO NIVEL SUPERIOR
        page_number = reader.get_destination_page_number(parent_bookmark) # PAGINA DE INICIO NIVEL SUPERIOR
        all_bookmark_info.append({"title": title, "start_page": page_number })

# FILTRA SOLO SECCIONES QUE NOS INTERESAN (LAS QUE ESTÁN EN chunk_level_info)
all_filtered_bookmarks_info: List[Dict[str,Any]] = []
for item in all_bookmark_info:
    if item['title'] in chunk_level_info:
        all_filtered_bookmarks_info.append(item)

final_pages_division_info: List[Dict[str, Any]] = []
final_page: int = len(reader.pages)

# CALCULA PÁGINAS DE CADA SECCIÓN 
for item, next_item in zip(all_filtered_bookmarks_info, all_filtered_bookmarks_info[1:]+[None]):
# lista = [A, B, C] --> lista[1:] → [B, C] --> [B, C] + [None] → [B, C, None]
# El None añade un elemento extra al final de la lista 
    if next_item:
        final_pages_division_info.append({
            **item,
            'final_page': next_item['start_page'] -1
        })
    else:
        final_pages_division_info.append({
            **item,
            'final_page': final_page -1
        })

# CREA LOS NUEVOS PDFs CON LAS SECCIONES DIVIDIDAS
for section in final_pages_division_info:
    writer = PdfWriter()
    name = section['title']
    start_page = section['start_page']
    end_page = section['final_page']

    # copia páginas al nuevo PDF
    if end_page > start_page:
        for page_num in range(start_page,end_page):
            writer.add_page(reader.pages[page_num]) 
    elif end_page == start_page:
        writer.add_page(reader.pages[start_page])

    # CARPETA DESTINO
    output_filename = f"{name}.pdf"
    output_path = os.path.join('data','optimized_chunks',output_filename)

    # GUARDA PDFs
    os.makedirs('data/optimized_chunks', exist_ok=True)
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)