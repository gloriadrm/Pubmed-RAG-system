import os
import requests
import tiktoken
from xml.etree import ElementTree as ET
from dotenv import load_dotenv

load_dotenv()
ncbi_api_key = os.getenv('NCBI_API_KEY')

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
from src.config import CHUNK_TOKEN_LIMIT, CHUNK_SIZE, CHUNK_OVERLAP

tokenizer = tiktoken.get_encoding("cl100k_base")


# -------------- FETCH --------------

def fetch_pmc_full_text(pmc_ids: list[str]) -> str:
    params = {
        "db": "pmc",
        "id": ",".join(pmc_ids),
        "retmode": "xml",
        "api_key": ncbi_api_key,
    }
    response = requests.get(BASE_URL + "efetch.fcgi", params=params, timeout=60)
    response.raise_for_status()
    return response.text


# -------------- JATS XML PARSERS --------------

def _text(elem, path, default=None):
    node = elem.find(path)
    if node is not None:
        return "".join(node.itertext()).strip() or default
    return default


def parse_pmc_metadata(article_elem):
    language     = article_elem.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
    article_type = article_elem.attrib.get("article-type")

    front = article_elem.find("front")
    if front is None:
        return {}

    meta = front.find("article-meta")
    if meta is None:
        return {}

    title = _text(meta, ".//title-group/article-title")

    pmc_id = pmid = doi = None
    for aid in meta.findall(".//article-id"):
        id_type = aid.attrib.get("pub-id-type")
        val = "".join(aid.itertext()).strip()
        if id_type in ("pmc", "pmcid"):   # NCBI usa ambos según el artículo
            pmc_id = val
        elif id_type == "pmid":
            pmid = val
        elif id_type == "doi":
            doi = val

    authors = []
    for contrib in meta.findall(".//contrib[@contrib-type='author']"):
        # .//surname cubre <name name-style="western"><surname>...</surname></name>
        surname = _text(contrib, ".//surname")
        given   = _text(contrib, ".//given-names")
        if surname:
            authors.append({"last_name": surname, "name": given})

    journal_meta = front.find("journal-meta")
    journal = _text(journal_meta, ".//journal-title") if journal_meta else None

    pub_date_val = None
    for pub_date in meta.findall(".//pub-date"):
        year  = _text(pub_date, "year")
        month = _text(pub_date, "month")
        day   = _text(pub_date, "day")
        if year:
            parts = [year]
            if month: parts.append(month.zfill(2))
            if day:   parts.append(day.zfill(2))
            pub_date_val = "-".join(parts)
            break

    keywords = []
    for kw in meta.findall(".//kwd-group/kwd"):
        text = "".join(kw.itertext()).strip()
        if text:
            keywords.append(text)

    abstract_node = meta.find(".//abstract")
    abstract = "".join(abstract_node.itertext()).strip() if abstract_node else None

    return {
        "pmc_id":       f"PMC{pmc_id}" if pmc_id and not str(pmc_id).startswith("PMC") else pmc_id,
        "pm_id":        pmid,
        "doi":          doi,
        "title":        title,
        "abstract":     abstract,
        "authors":      authors,
        "journal":      journal,
        "pub_date":     pub_date_val,
        "keywords":     keywords,
        "language":     language,
        "article_type": article_type,
    }


# Tags dentro de <p> que no son texto narrativo y deben omitirse
_SKIP_TAGS_IN_P = {"table-wrap", "fig", "supplementary-material", "media",
                   "inline-supplementary-material", "boxed-text"}

# sec-type que no aportan contenido útil para RAG
_SKIP_SEC_TYPES = {"supplementary-material", "supplementary-data",
                   "additional-information", "notes"}


def _paragraph_text(p_elem) -> str:
    """
    Extrae el texto narrativo de un <p>, ignorando tablas, figuras
    y otros bloques que no son texto corrido.

    Estrategia: recorre sólo los hijos directos de <p>.
    - Si el hijo es un tag a saltar (tabla, figura…), ignoramos su subárbol
      pero conservamos el .tail (texto que viene justo después).
    - Para el resto (xref, italic, bold, sup…) usamos itertext() de ese hijo.
    """
    parts = []
    if p_elem.text:
        parts.append(p_elem.text.strip())

    for child in p_elem:
        if child.tag in _SKIP_TAGS_IN_P:
            # No incluimos el contenido del bloque, pero sí el texto tras él
            if child.tail:
                parts.append(child.tail.strip())
        else:
            # Inline tag (xref, italic, bold, sup, sub, named-content…)
            inline = "".join(child.itertext()).strip()
            if inline:
                parts.append(inline)
            if child.tail:
                parts.append(child.tail.strip())

    return " ".join(p for p in parts if p)


def parse_section(sec_elem, parent_label=None):
    # Omitir secciones de material suplementario
    sec_type = sec_elem.attrib.get("sec-type", "")
    if sec_type in _SKIP_SEC_TYPES:
        return []

    title_node = sec_elem.find("title")
    title_text = "".join(title_node.itertext()).strip() if title_node is not None else ""

    # Etiqueta del nodo: título propio o heredado del padre
    label = title_text or parent_label or "Unknown"

    paragraphs = []
    for child in sec_elem:
        if child.tag == "p":
            text = _paragraph_text(child)
            if text:
                paragraphs.append(text)

    sections = []
    if paragraphs:
        sections.append({"section": label, "text": " ".join(paragraphs)})

    for subsec in sec_elem.findall("sec"):
        sections.extend(parse_section(subsec, parent_label=label))

    return sections


def parse_pmc_sections(article_elem):
    body = article_elem.find("body")
    if body is None:
        return []
    sections = []
    for sec in body.findall("sec"):
        sections.extend(parse_section(sec))
    return sections


# -------------- ADAPTIVE CHUNKING --------------

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def chunk_text(text: str) -> list[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunks.append(tokenizer.decode(tokens[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def adaptive_chunk_section(section: dict) -> list[dict]:
    text  = section["text"]
    label = section["section"]

    if count_tokens(text) <= CHUNK_TOKEN_LIMIT:
        return [{"section": label, "text": text, "chunk_index": 0, "total_chunks": 1}]

    raw_chunks = chunk_text(text)
    return [
        {"section": label, "text": chunk, "chunk_index": i, "total_chunks": len(raw_chunks)}
        for i, chunk in enumerate(raw_chunks)
    ]


# -------------- BUILD CHUNK DICTS --------------

def build_pmc_chunks_from_article(article_elem) -> list[dict]:
    meta     = parse_pmc_metadata(article_elem)
    sections = parse_pmc_sections(article_elem)

    pmc_id = meta.get("pmc_id") or "unknown"
    title  = meta.get("title") or ""

    # Recolectar todos los chunks del artículo antes de numerarlos
    # para que chunk_index sea global dentro del artículo, no local por sección.
    # Sin esto, cada sección empieza en chunk_index=0 y los IDs colisionan en Qdrant.
    raw_chunks = []
    for sec in sections:
        for chunk in adaptive_chunk_section(sec):
            raw_chunks.append(chunk)

    total = len(raw_chunks)
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunk_id = f"pmc_{pmc_id}_c{i}"
        chunks.append({
            "chunk_id":     chunk_id,
            "source":       "pmc",
            "pmc_id":       pmc_id,
            "pm_id":        meta.get("pm_id"),
            "doi":          meta.get("doi"),
            "title":        title,
            "section":      chunk["section"],
            "text":         chunk["text"],
            "chunk_index":  i,
            "total_chunks": total,
            "embedding_text": f"{title} | {chunk['section']} | {chunk['text']}",
            "metadata": {
                "abstract":          meta.get("abstract"),
                "authors":           meta.get("authors", []),
                "journal":           meta.get("journal"),
                "language":          meta.get("language"),
                "article_type":      meta.get("article_type"),
                "keywords":          meta.get("keywords", []),
                "pub_date":          meta.get("pub_date"),
                "pmc_last_updated":  None,   # rellenar con enrich_with_oa_metadata()
                "license":           None,   # rellenar con enrich_with_oa_metadata()
            }
        })

    return chunks


def parse_pmc_xml_to_chunks(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    all_chunks = []
    for article_elem in root.findall(".//article"):
        all_chunks.extend(build_pmc_chunks_from_article(article_elem))
    return all_chunks


# -------------- OA ENRICHMENT --------------

def enrich_with_oa_metadata(chunks: list[dict], oa_index: dict) -> list[dict]:
    """
    Enriquece los chunks con last_updated y license del OA file list.

    oa_index: {pmc_id: {"last_updated": "2024-05-22 15:25:16", "license": "CC BY"}}
    Construir con lookup_oa_csv() de oa_updater.py.
    """
    for chunk in chunks:
        oa = oa_index.get(chunk["pmc_id"], {})
        chunk["metadata"]["pmc_last_updated"] = oa.get("last_updated")
        chunk["metadata"]["license"]      = oa.get("license")
    return chunks